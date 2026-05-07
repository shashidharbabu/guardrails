"""Sessions router — list, retrieve, pipeline, CSE, events."""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.backend import db
from app.backend.pipeline import run_pipeline

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
def list_sessions(
    limit: int = 100,
    status: Optional[str] = None,
    tenant_id: Optional[str] = None,
):
    return db.get_sessions(limit=limit, status=status, tenant_id=tenant_id)


@router.get("/{session_id}")
def get_session(session_id: str):
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


@router.get("/{session_id}/events")
def get_session_events(session_id: str):
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return db.get_session_events(session_id)


@router.get("/{session_id}/cse")
def get_session_cse(session_id: str):
    """
    Return the CSE breakdown for a session.

    NOTE: db._deserialize_session() stores mad_output_json as both
    mad_output_json (parsed dict) AND mad_output (alias). We read from
    mad_output here which is always present after deserialization.
    """
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")

    # mad_output is the deserialized form of mad_output_json (see db._deserialize_session)
    mad_output = s.get("mad_output")
    if not mad_output:
        raise HTTPException(
            404,
            "MAD not completed for this session — "
            "CSE data will be available after MAD finishes. Try again shortly.",
        )

    # cse_result_json is stored as a separate column but also may be inside mad_output
    cse = s.get("cse_result_json") or (
        mad_output.get("cse_result") if isinstance(mad_output, dict) else None
    )

    if not cse:
        raise HTTPException(
            404,
            "CSE data not available — MAD may have run in judge-only mode without CSE.",
        )

    # Normalize CSE to the production schema
    return {
        "session_id": session_id,
        "mad_query_id": s.get("mad_query_id"),
        "cse": _normalize_cse(cse),
    }


def _normalize_cse(raw: dict) -> dict:
    """Ensure CSE response always has the full production schema."""
    return {
        "final_cse_score": raw.get("final_cse_score") or raw.get("aggregate_score"),
        "routing_decision": raw.get("routing_decision") or raw.get("route"),
        "scoring_mode": raw.get("scoring_mode", "full"),
        "cse_version": raw.get("cse_version") or raw.get("version", "1.0.0"),
        "score_breakdown": raw.get("score_breakdown") or raw.get("components", {}),
        "triggered_flags": raw.get("triggered_flags") or raw.get("flags", []),
        "explanation": raw.get("explanation") or raw.get("summary"),
        "retry_reasons": raw.get("retry_reasons", []),
        "review_reasons": raw.get("review_reasons", []),
        "block_reasons": raw.get("block_reasons", []),
        "top_failed_claims": raw.get("top_failed_claims") or raw.get("failed_claims", []),
        "metadata": raw.get("metadata", {}),
    }


# ── POST /api/query ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    llm_model: Optional[str] = None


query_router = APIRouter(prefix="/api", tags=["query"])


@query_router.post("/query")
async def submit_query(body: QueryRequest, request: Request):
    if not body.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    session = await run_pipeline(
        query=body.query.strip(),
        llm_model=body.llm_model,
        request_id=request_id,
    )

    db.insert_audit_log(
        action="query_submitted",
        resource_type="session",
        resource_id=session.get("id"),
        session_id=session.get("id"),
        request_id=request_id,
        after={"query_truncated": body.query.strip()[:200]},
    )
    return session
