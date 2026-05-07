"""
api.py — FastAPI service for the MAD pipeline.

Endpoints:
  POST /mad/verify     — run full MAD pipeline on a query + LLM answer
  GET  /mad/health     — liveness check
  GET  /mad/info       — show current config (model, thresholds)

Run with (from repo root):
  PYTHONPATH=. uvicorn multi_agent_debate.multi_agent.api:app --host 0.0.0.0 --port 8001 --reload
"""
from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .mad_tracing import flush as lf_flush
from .mad_tracing import observe as lf_observe
from .mad_tracing import observability_status as lf_observability_status
from .mad_tracing import ping as lf_ping
from .mad_tracing import shutdown as lf_shutdown
from .mad_pipeline import run_mad
from .config import (
    AGENT_MODEL, JUDGE_MODEL,
    MAX_CYCLES,
    CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_LOW,
    OLLAMA_BASE_URL,
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    lf_shutdown()


app = FastAPI(
    title="Guardrails Gateway — MAD API",
    description=(
        "Multi-Agent Debate verification pipeline. "
        "Accepts an enterprise LLM answer and verifies it against regulatory evidence."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class MADRequest(BaseModel):
    query:      str
    llm_answer: str


class ClaimOut(BaseModel):
    claim_id:       int
    claim_text:     str
    is_material:    bool
    confidence:     float
    verdict:        Optional[str]
    evidence_chunks: List[str]
    reasoning:      str


class JudgeVerdictOut(BaseModel):
    claim_id:   int
    claim_text: str
    is_material: bool
    score:      float
    reasoning:  str


class MADResponse(BaseModel):
    routing_decision:     str
    aggregate_confidence: float
    correction_signal:    Optional[str]
    claims:               List[ClaimOut]
    judge_verdicts:       List[JudgeVerdictOut]
    debate_transcript:    str
    query_id:             str   # UUID — use to query SQLite for this run
    rollout_id:           str   # UUID — for GRPO multi-rollout comparison


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/mad/verify", response_model=MADResponse)
async def verify(request: MADRequest) -> MADResponse:
    """
    Run the full MAD verification pipeline.

    Input:  { "query": "...", "llm_answer": "..." }
    Output: routing decision, judge verdicts, correction signal, full transcript
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if not request.llm_answer.strip():
        raise HTTPException(status_code=422, detail="llm_answer must not be empty")

    try:
        q, a = request.query.strip(), request.llm_answer.strip()
        trace_out: Optional[Dict[str, Any]] = None
        with lf_observe(
            "span",
            "mad.http.verify",
            input_payload={
                "query_chars": len(q),
                "answer_chars": len(a),
                "endpoint": "/mad/verify",
            },
        ) as lf_root:
            result = run_mad(query=q, llm_answer=a)
            trace_out = {
                "routing_decision": result.routing_decision,
                "aggregate_confidence": result.aggregate_confidence,
                "query_id": result.query_id,
                "rollout_id": result.rollout_id,
                "claims_count": len(result.claims),
            }
            if lf_root is not None:
                try:
                    lf_root.update(output=trace_out)
                except Exception:
                    pass
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"MAD pipeline error: {str(e)}\n\n{tb}",
        )
    finally:
        lf_flush()

    return MADResponse(
        routing_decision=result.routing_decision,
        aggregate_confidence=result.aggregate_confidence,
        correction_signal=result.correction_signal,
        claims=[
            ClaimOut(
                claim_id=c.claim_id,
                claim_text=c.claim_text,
                is_material=c.is_material,
                confidence=c.confidence,
                verdict=c.verdict.value if c.verdict else None,
                evidence_chunks=c.evidence_chunks,
                reasoning=c.reasoning,
            )
            for c in result.claims
        ],
        judge_verdicts=[
            JudgeVerdictOut(
                claim_id=jv.claim_id,
                claim_text=jv.claim_text,
                is_material=jv.is_material,
                score=jv.score,
                reasoning=jv.reasoning,
            )
            for jv in result.judge_verdicts
        ],
        debate_transcript=result.debate_transcript,
        query_id=result.query_id,
        rollout_id=result.rollout_id,
    )


@app.get("/mad/health")
async def health() -> dict:
    return {"status": "ok", "service": "MAD pipeline"}


@app.get("/mad/observability/status")
async def mad_observability_status() -> dict:
    """Langfuse config snapshot (no secrets). Use when traces do not appear in the UI."""
    return lf_observability_status()


@app.post("/mad/observability/ping")
async def mad_observability_ping() -> dict:
    """
    Emit a minimal Langfuse trace and flush. Open the Langfuse UI Traces tab after calling.
    Does not run the full MAD pipeline (no Ollama/Anthropic required).
    """
    return lf_ping()


@app.get("/mad/info")
async def info() -> dict:
    return {
        "agent_model":          AGENT_MODEL,
        "judge_model":          JUDGE_MODEL,
        "ollama_base_url":      OLLAMA_BASE_URL,
        "max_cycles":           MAX_CYCLES,
        "threshold_high":       CONFIDENCE_THRESHOLD_HIGH,
        "threshold_low":        CONFIDENCE_THRESHOLD_LOW,
    }
