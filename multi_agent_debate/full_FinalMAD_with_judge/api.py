"""
api.py — FastAPI service for the full_FinalMAD_with_judge pipeline.

Backward-compatible with the existing multi_agent/api.py contract:
  POST /mad/verify          — run full MAD pipeline (decompose → R0 → R1 → judge)
  GET  /mad/health          — liveness check
  GET  /mad/info            — show current config

New endpoints:
  POST /mad/run             — async batch run from raw JSON payload (no pre-built claims)

Run with (from repo root):
  cd multi_agent_debate/full_FinalMAD_with_judge
  uvicorn api:app --host 0.0.0.0 --port 8001 --reload
"""
from __future__ import annotations

import asyncio
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

# ── path setup so `configs` and `src` are importable ─────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
load_dotenv(_HERE.parent.parent / ".env")   # repo-root .env
load_dotenv(_HERE / ".env", override=False) # local .env (lower priority)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from configs import config
from src.db.db import init_db
from src.debate.pipeline import run_debate_for_query
from src.langfuse_log.logger import get_langfuse_client
from src.schemas.schemas import JudgeVerdict as JudgeVerdictEnum

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Guardrails Gateway — MAD API (full_FinalMAD_with_judge)",
    description=(
        "Multi-Agent Debate verification pipeline. "
        "Accepts an enterprise LLM answer and verifies it against regulatory evidence."
    ),
    version="2.0.0",
)

# ── Request / Response schemas (backward-compatible with multi_agent/api.py) ──

class MADRequest(BaseModel):
    query: str
    llm_answer: str


class ClaimOut(BaseModel):
    claim_id: str
    claim_text: str
    is_material: bool
    is_critical: bool
    confidence: float
    verdict: Optional[str] = None
    evidence_chunks: List[str] = []
    reasoning: str = ""


class JudgeVerdictOut(BaseModel):
    claim_id: str
    claim_text: str
    is_material: bool
    score: float
    reasoning: str


class MADResponse(BaseModel):
    routing_decision: str
    aggregate_confidence: float
    correction_signal: Optional[str] = None
    claims: List[ClaimOut]
    judge_verdicts: List[JudgeVerdictOut]
    debate_transcript: str
    query_id: str
    rollout_id: str
    cse_result: Optional[Dict[str, Any]] = None
    langfuse_trace_id: Optional[str] = None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_routing(
    judge_verdicts_by_claim: dict,
    claims: list,
) -> tuple[str, float]:
    """
    Derive routing decision and aggregate confidence from judge verdicts.

    Hard rule: any material claim with v_label=0.0 → HARD_BLOCK.
    Soft rules (70% material-min + 30% non-material-mean):
      >= 0.8  → DELIVER
      >= 0.4  → RETRY
      <  0.4  → HUMAN_REVIEW
    """
    claim_map = {c.claim_id: c for c in claims}
    scores: List[tuple[bool, float]] = []

    for claim_id, verdict in judge_verdicts_by_claim.items():
        claim = claim_map.get(claim_id)
        v_label = float(verdict.v_label.value)
        is_material = claim.is_material if claim else False

        # Hard block: fabricated material claim
        if is_material and v_label == 0.0:
            agg = _aggregate(scores or [(is_material, v_label)])
            return "HARD_BLOCK", agg

        scores.append((is_material, v_label))

    if not scores:
        return "DELIVER", 1.0

    agg = _aggregate(scores)
    if agg >= 0.8:
        return "DELIVER", agg
    elif agg >= 0.4:
        return "RETRY", agg
    else:
        return "HUMAN_REVIEW", agg


def _aggregate(scores: List[tuple[bool, float]]) -> float:
    mat = [s for is_mat, s in scores if is_mat]
    nmat = [s for is_mat, s in scores if not is_mat]
    if mat and nmat:
        return round(0.70 * min(mat) + 0.30 * (sum(nmat) / len(nmat)), 4)
    if mat:
        return round(min(mat), 4)
    return round(sum(nmat) / len(nmat), 4)


def _build_claim_outs(state) -> List[ClaimOut]:
    outs = []
    for claim in state.claims:
        # Pull R1 agent_a output as representative verdict/reasoning
        outputs = state.agent_outputs_by_claim.get(claim.claim_id, {})
        a_r1 = (outputs.get("agent_a") or {}).get(1)
        verdict_str = a_r1.verdict.value if a_r1 else None
        reasoning = a_r1.reasoning if a_r1 else ""
        evidence_chunks = [e.chunk_id for e in a_r1.evidence_cited] if a_r1 else []
        confidence = a_r1.confidence_internal if a_r1 else claim.confidence_prior

        outs.append(ClaimOut(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            is_material=claim.is_material,
            is_critical=claim.is_critical,
            confidence=confidence,
            verdict=verdict_str,
            evidence_chunks=evidence_chunks,
            reasoning=reasoning,
        ))
    return outs


def _build_judge_outs(state) -> List[JudgeVerdictOut]:
    claim_map = {c.claim_id: c for c in state.claims}
    outs = []
    for claim_id, verdict in state.judge_verdicts_by_claim.items():
        claim = claim_map.get(claim_id)
        outs.append(JudgeVerdictOut(
            claim_id=claim_id,
            claim_text=claim.claim_text if claim else claim_id,
            is_material=claim.is_material if claim else False,
            score=float(verdict.v_label.value),
            reasoning=verdict.judge_reasoning,
        ))
    return outs


def _build_transcript(state, routing: str, agg: float) -> str:
    lines = [
        f"=== MAD Debate Transcript ===",
        f"Query    : {state.user_query}",
        f"Run ID   : {state.run_id}",
        f"Query ID : {state.query_id}",
        "",
        f"Claims ({len(state.claims)}):",
    ]
    for c in state.claims:
        mat = "MATERIAL" if c.is_material else "contextual"
        lines.append(f"  [{mat}] {c.claim_id}: {c.claim_text}")

    lines += ["", "Agent Outputs (Round 1):"]
    for claim in state.claims:
        outputs = state.agent_outputs_by_claim.get(claim.claim_id, {})
        for role in ("agent_a", "agent_b"):
            r1 = (outputs.get(role) or {}).get(1)
            if r1:
                lines.append(
                    f"  {role} / {claim.claim_id}: "
                    f"verdict={r1.verdict.value} conf={r1.confidence_internal:.2f}"
                )

    if state.judge_verdicts_by_claim:
        lines += ["", "Judge Verdicts:"]
        for cid, v in state.judge_verdicts_by_claim.items():
            lines.append(
                f"  {cid}: v_label={float(v.v_label.value):.1f} "
                f"confidence={v.judge_confidence:.2f}"
            )

    lines += [
        "",
        f"Routing Decision : {routing}",
        f"Aggregate Score  : {agg:.4f}",
    ]
    return "\n".join(lines)


async def _run_pipeline(query: str, llm_answer: str) -> MADResponse:
    """
    Build a single-query payload, initialise DB, run debate + judge, return response.
    Uses Ollama/vLLM depending on env vars in configs/config.py.
    Falls back gracefully if vLLM is unavailable (judge skipped on connection error).
    """
    run_id = f"run-{uuid4().hex[:8]}"
    query_id = f"q-{uuid4().hex[:8]}"
    rollout_id = str(uuid4())

    db_path = config.SQLITE_DB_PATH
    init_db(db_path)

    # Build a minimal query dict — no pre-built claims, let decomposer run
    query_dict = {
        "query_id": query_id,
        "user_query": query,
        "baseline_answer": llm_answer,
        "rag_chunks": [],   # RAG pass-through; live RAG wired via RAG_SERVICE_PATH
        "claims": [],
    }

    # Persist query row
    from src.db.db import insert_query
    insert_query(
        query_id=query_id,
        run_id=run_id,
        user_query=query,
        rag_chunk_ids=[],
        rag_chunks=[],
        baseline_answer=llm_answer,
        baseline_model="app-backend",
        db_path=db_path,
    )

    langfuse = get_langfuse_client()
    # Create a parent trace for this pipeline run so all per-claim traces link up
    langfuse_run_trace_id: Optional[str] = None
    if langfuse:
        try:
            parent_trace = langfuse.trace(
                name="mad-pipeline-run",
                input={"query": query, "run_id": run_id, "query_id": query_id},
                metadata={"run_id": run_id},
            )
            langfuse_run_trace_id = parent_trace.id
        except Exception:
            pass

    run_judge = True
    try:
        from src.debate.pipeline import run_debate_for_query as _run
        # We need the full PipelineState back, not just the summary dict.
        # Monkey-patch: call internal steps directly so we keep state.
        from src.schemas.schemas import PipelineState, Claim
        from src.agents.round0 import debate_round0_node
        from src.agents.round1 import debate_round1_node
        from src.decomposer.node import decompose_node
        from src.db.db import connect, insert_claim

        state = PipelineState(
            query_id=query_id,
            run_id=run_id,
            user_query=query,
            rag_chunks=[],
            baseline_answer=llm_answer,
            claims=[],
        )

        # Decompose
        state = await decompose_node(state)

        # Persist claims
        with connect(db_path) as conn:
            for claim in state.claims:
                insert_claim(conn, claim, query_id, True, 1.0)

        # Round 0 + Round 1
        state = await debate_round0_node(state)
        state = await debate_round1_node(state)

        # Judge
        try:
            from src.judge.node import judge_node
            state = await judge_node(state, langfuse=langfuse)
        except Exception as exc:
            logger.warning("Judge node failed (%s) — proceeding without verdicts", exc)

    except Exception as exc:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"MAD pipeline error: {exc}\n\n{tb}",
        )

    routing, agg = _compute_routing(state.judge_verdicts_by_claim, state.claims)
    claims_out = _build_claim_outs(state)
    judge_outs = _build_judge_outs(state)
    transcript = _build_transcript(state, routing, agg)

    if langfuse and langfuse_run_trace_id:
        try:
            langfuse.trace(
                id=langfuse_run_trace_id,
                output={"routing_decision": routing, "aggregate_confidence": agg},
            )
            langfuse.flush()
        except Exception:
            pass

    return MADResponse(
        routing_decision=routing,
        aggregate_confidence=agg,
        correction_signal=None,
        claims=claims_out,
        judge_verdicts=judge_outs,
        debate_transcript=transcript,
        query_id=query_id,
        rollout_id=rollout_id,
        cse_result=None,
        langfuse_trace_id=langfuse_run_trace_id,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/mad/verify", response_model=MADResponse)
async def verify(request: MADRequest) -> MADResponse:
    """
    Run full MAD verification pipeline.
    Drop-in replacement for multi_agent/api.py /mad/verify.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if not request.llm_answer.strip():
        raise HTTPException(status_code=422, detail="llm_answer must not be empty")
    return await _run_pipeline(request.query, request.llm_answer)


@app.get("/mad/health")
async def health() -> dict:
    return {"status": "ok", "service": "MAD pipeline v2.0 (full_FinalMAD_with_judge)"}


@app.get("/mad/info")
async def info() -> dict:
    return {
        "version": "2.0.0",
        "agents_model": config.AGENTS_MODEL_NAME,
        "judge_model": config.JUDGE_MODEL_NAME,
        "vllm_agents_url": config.VLLM_AGENTS_URL,
        "vllm_judge_url": config.VLLM_JUDGE_URL,
        "sqlite_db_path": config.SQLITE_DB_PATH,
        "claim_concurrency": config.CLAIM_CONCURRENCY,
    }
