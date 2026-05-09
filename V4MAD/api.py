"""
Single-query V4MAD runner for the backend app.

This module wraps the existing V4MAD nodes without changing the agent prompts,
schemas, parsers, or debate behavior. The backend passes the already-generated
LLM answer as the baseline answer, then V4MAD decomposes claims, retrieves live
NewRAG v2 evidence per claim, runs the two debate rounds, and judges.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

_V4MAD_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _V4MAD_ROOT.parent
for _path in (str(_V4MAD_ROOT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import AGENT_A_MODEL, AGENT_B_MODEL, BASELINE_MODEL, DB_PATH
from db import (
    get_agent_output,
    get_claim_chunks,
    get_claims_for_query,
    get_db_conn,
    init_db,
    insert_query,
)
from nodes.claim_rag_node import claim_rag_node
from nodes.debate_r0_node import debate_r0_node
from nodes.debate_r1_node import debate_r1_node
from nodes.decompose_node import decompose_node
from nodes.judge_node import judge_node
from schemas import MADState


class ClaimOut(BaseModel):
    claim_id: str
    claim_text: str
    is_material: bool
    is_critical: bool
    confidence: float
    verdict: Optional[str] = None
    evidence_chunks: list[str] = Field(default_factory=list)
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
    claims: list[ClaimOut]
    judge_verdicts: list[JudgeVerdictOut]
    debate_transcript: str
    query_id: str
    rollout_id: str
    cse_result: Optional[dict[str, Any]] = None
    langfuse_trace_id: Optional[str] = None


def _aggregate(scores: list[tuple[bool, float]]) -> float:
    material = [score for is_material, score in scores if is_material]
    non_material = [score for is_material, score in scores if not is_material]
    if material and non_material:
        return round(0.70 * min(material) + 0.30 * (sum(non_material) / len(non_material)), 4)
    if material:
        return round(min(material), 4)
    if non_material:
        return round(sum(non_material) / len(non_material), 4)
    return 1.0


def _compute_routing(claims: list[dict], judge_verdicts: dict[str, dict]) -> tuple[str, float]:
    if claims and not judge_verdicts:
        return "HUMAN_REVIEW", 0.0

    claim_map = {claim["claim_id"]: claim for claim in claims}
    scores: list[tuple[bool, float]] = []

    for claim_id, verdict in judge_verdicts.items():
        claim = claim_map.get(claim_id, {})
        is_material = bool(claim.get("is_material", True))
        score = float(verdict.get("v_label", 0.5))
        if is_material and score == 0.0:
            return "HARD_BLOCK", _aggregate(scores or [(is_material, score)])
        scores.append((is_material, score))

    aggregate = _aggregate(scores)
    if aggregate >= 0.8:
        return "DELIVER", aggregate
    if aggregate >= 0.4:
        return "RETRY", aggregate
    return "HUMAN_REVIEW", aggregate


def _build_claim_outs(claims: list[dict], claim_chunks: dict[str, dict],
                      agent_outputs: dict[str, dict]) -> list[ClaimOut]:
    outs: list[ClaimOut] = []
    for claim in claims:
        claim_id = claim["claim_id"]
        r1 = agent_outputs.get(claim_id, {}).get("agent_a", {}).get(1) or {}
        evidence = r1.get("evidence_cited", [])
        if not evidence:
            evidence = [
                {"chunk_id": chunk.get("chunk_id", "")}
                for chunk in claim_chunks.get(claim_id, {}).get("judge", [])
            ]
        outs.append(
            ClaimOut(
                claim_id=claim_id,
                claim_text=claim.get("claim_text", ""),
                is_material=bool(claim.get("is_material", True)),
                is_critical=bool(claim.get("is_critical", False)),
                confidence=float(claim.get("confidence_prior", 0.75)),
                verdict=r1.get("verdict"),
                evidence_chunks=[item.get("chunk_id", "") for item in evidence if item.get("chunk_id")],
                reasoning=r1.get("reasoning", ""),
            )
        )
    return outs


def _build_judge_outs(claims: list[dict], judge_verdicts: dict[str, dict]) -> list[JudgeVerdictOut]:
    outs: list[JudgeVerdictOut] = []
    for claim in claims:
        claim_id = claim["claim_id"]
        verdict = judge_verdicts.get(claim_id, {})
        outs.append(
            JudgeVerdictOut(
                claim_id=claim_id,
                claim_text=claim.get("claim_text", ""),
                is_material=bool(claim.get("is_material", True)),
                score=float(verdict.get("v_label", 0.5)),
                reasoning=verdict.get("judge_reasoning", ""),
            )
        )
    return outs


def _build_transcript(query: str, claims: list[dict], routing: str, aggregate: float) -> str:
    lines = [
        "V4MAD single-query run",
        f"Query: {query}",
        f"Routing: {routing}",
        f"Aggregate confidence: {aggregate:.4f}",
        f"Claims: {len(claims)}",
    ]
    for claim in claims:
        lines.append(f"- {claim.get('claim_id')}: {claim.get('claim_text')}")
    return "\n".join(lines)


async def run_v4mad(
    query: str,
    llm_answer: str,
    *,
    query_id: Optional[str] = None,
    rollout_id: Optional[str] = None,
    run_debate: bool = True,
    run_judge: bool = True,
    max_claims: Optional[int] = None,
    claims_override: Optional[list[dict]] = None,
) -> MADResponse:
    """
    Run V4MAD for one backend query.

    The backend LLM answer is treated as the baseline answer. If run_debate and
    run_judge are true, vLLM-compatible servers must be reachable at the V4MAD
    configured ports. tests may pass claims_override and disable debate to verify
    the NewRAG claim retrieval path without launching agent models.
    """
    init_db(str(DB_PATH))
    conn = get_db_conn()

    query_id = query_id or f"app_{uuid.uuid4().hex}"
    rollout_id = rollout_id or f"v4mad_{uuid.uuid4().hex}"

    insert_query(
        conn,
        query_id,
        rollout_id,
        query,
        llm_answer,
        BASELINE_MODEL,
        0,
        0,
        0,
    )

    state: MADState = {
        "query_id": query_id,
        "run_id": rollout_id,
        "user_query": query,
        "baseline_answer": llm_answer,
        "claims": claims_override or [],
        "query_chunks": [],
        "claim_chunks": {},
        "agent_outputs": {},
        "judge_verdicts": {},
        "errors": [],
    }

    if claims_override:
        from db import insert_claim

        for index, claim in enumerate(claims_override):
            claim.setdefault("claim_id", f"{query_id}_claim_{index}")
            claim.setdefault("claim_index", index)
            claim.setdefault("is_material", True)
            claim.setdefault("is_critical", False)
            claim.setdefault("confidence_prior", 0.75)
            insert_claim(conn, claim, query_id)
    else:
        state.update(await decompose_node(state))
        state["claims"] = get_claims_for_query(conn, query_id)

    if max_claims is not None and max_claims > 0:
        state["claims"] = state.get("claims", [])[:max_claims]

    state.update(await claim_rag_node(state))

    if run_debate:
        state.update(await debate_r0_node(state))
        state.update(await debate_r1_node(state))

    if run_debate and run_judge:
        state.update(await judge_node(state))

    claims = state.get("claims", [])
    claim_chunks = state.get("claim_chunks", {})
    agent_outputs = state.get("agent_outputs", {})
    judge_verdicts = state.get("judge_verdicts", {})

    # Rehydrate from DB when a node wrote data but returned a partial state.
    for claim in claims:
        claim_id = claim["claim_id"]
        claim_chunks.setdefault(claim_id, get_claim_chunks(conn, claim_id))
        agent_outputs.setdefault(claim_id, {"agent_a": {}, "agent_b": {}})
        for agent_role in ("agent_a", "agent_b"):
            for round_num in (0, 1):
                output = get_agent_output(conn, claim_id, agent_role, round_num)
                if output:
                    agent_outputs[claim_id].setdefault(agent_role, {})[round_num] = output

    routing, aggregate = _compute_routing(claims, judge_verdicts)
    return MADResponse(
        routing_decision=routing,
        aggregate_confidence=aggregate,
        correction_signal=None,
        claims=_build_claim_outs(claims, claim_chunks, agent_outputs),
        judge_verdicts=_build_judge_outs(claims, judge_verdicts),
        debate_transcript=_build_transcript(query, claims, routing, aggregate),
        query_id=query_id,
        rollout_id=rollout_id,
        cse_result=None,
        langfuse_trace_id=None,
    )
