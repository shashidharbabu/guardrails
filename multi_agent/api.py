"""
api.py — FastAPI service for the MAD pipeline.

Endpoints:
  POST /mad/verify          — run full MAD pipeline on a query + LLM answer
  GET  /mad/cse/{query_id}  — retrieve CSE breakdown for a completed run
  GET  /mad/health          — liveness check
  GET  /mad/info            — show current config (model, thresholds)

Run with:
  uvicorn multi_agent.api:app --host 0.0.0.0 --port 8001 --reload
"""
from __future__ import annotations

import json
import sqlite3
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from multi_agent.mad_pipeline import run_mad
from multi_agent.config import (
    AGENT_MODEL, JUDGE_MODEL,
    MAX_CYCLES,
    CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_LOW,
    OLLAMA_BASE_URL,
    DB_PATH,
)

app = FastAPI(
    title="Guardrails Gateway — MAD API",
    description=(
        "Multi-Agent Debate verification pipeline. "
        "Accepts an enterprise LLM answer and verifies it against regulatory evidence."
    ),
    version="0.1.0",
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
    query_id:             str
    rollout_id:           str
    cse_result:           Optional[Dict[str, Any]] = None


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
        result = run_mad(query=request.query, llm_answer=request.llm_answer)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"MAD pipeline error: {str(e)}\n\n{tb}",
        )

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
        cse_result=result.cse_result,
    )


@app.get("/mad/cse/{query_id}")
async def get_cse(query_id: str) -> dict:
    """
    Retrieve the full CSE breakdown for a completed MAD run by query_id.

    Returns four component scores, the final weighted score, routing decision,
    CSE version, and the full cse_breakdown JSON blob (scoring_mode, explanation,
    triggered_flags, top_failed_claims) when available.
    """
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        row = con.execute(
            """SELECT query_id, rollout_id, final_cse_score, routing_decision,
                      cse_f_llm, cse_h_llm, cse_relevancy, cse_judge_eval,
                      cse_version, cse_breakdown, timestamp
               FROM queries
               WHERE query_id = ?
               ORDER BY timestamp DESC
               LIMIT 1""",
            (query_id,),
        ).fetchone()
        con.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}")

    if row is None:
        raise HTTPException(status_code=404, detail=f"query_id {query_id!r} not found")

    breakdown = None
    if row["cse_breakdown"]:
        try:
            breakdown = json.loads(row["cse_breakdown"])
        except (json.JSONDecodeError, TypeError):
            breakdown = None

    return {
        "query_id":         row["query_id"],
        "rollout_id":       row["rollout_id"],
        "final_score":      row["final_cse_score"],
        "routing_decision": row["routing_decision"],
        "components": {
            "f_llm":      row["cse_f_llm"],
            "h_llm":      row["cse_h_llm"],
            "relevancy":  row["cse_relevancy"],
            "judge_eval": row["cse_judge_eval"],
        },
        "version":      row["cse_version"],
        "timestamp":    row["timestamp"],
        "cse_breakdown": breakdown,
    }


@app.get("/mad/health")
async def health() -> dict:
    return {"status": "ok", "service": "MAD pipeline"}


@app.get("/mad/info")
async def info() -> dict:
    return {
        "agent_model":     AGENT_MODEL,
        "judge_model":     JUDGE_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "max_cycles":      MAX_CYCLES,
        "threshold_high":  CONFIDENCE_THRESHOLD_HIGH,
        "threshold_low":   CONFIDENCE_THRESHOLD_LOW,
    }
