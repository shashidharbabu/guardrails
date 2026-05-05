"""Sessions router — list, retrieve, and run pipeline sessions."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.backend import db
from app.backend.pipeline import run_pipeline

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
def list_sessions(limit: int = 100):
    return db.get_sessions(limit=limit)


@router.get("/{session_id}")
def get_session(session_id: str):
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


# ── POST /api/query  (mounted at app level, not /api/sessions) ────────────────

class QueryRequest(BaseModel):
    query: str
    llm_model: Optional[str] = "qwen2.5:7b"


query_router = APIRouter(prefix="/api", tags=["query"])


@query_router.post("/query")
async def submit_query(body: QueryRequest):
    """
    Run a query through the full pipeline:
    Gateway → LLM (if not blocked) → MAD → store session → return.
    """
    if not body.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    session = await run_pipeline(body.query.strip(), body.llm_model)
    return session
