#!/usr/bin/env python3
"""FastAPI wrapper for NewRAG v2 retrieval.

Run on Colab/AWS/5090 and expose with ngrok. The app backend can call
POST /retrieve instead of loading Qwen3-Embedding-4B locally.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from rag_v2.rag_service import get_service


app = FastAPI(title="NewRAG v2 Retrieval Service")
_service = None


class RetrieveRequest(BaseModel):
    query: str
    k: int = 5
    session_id: str = "remote"
    agent_id: str = "shared"
    round_num: int = 0
    question_type: str = ""
    domain_filter: Optional[str] = None


@app.on_event("startup")
def startup() -> None:
    global _service
    _service = get_service()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service_loaded": _service is not None}


@app.post("/retrieve")
def retrieve(body: RetrieveRequest) -> dict:
    service = _service or get_service()
    return service.retrieve_for_cod(
        query=body.query,
        session_id=body.session_id,
        agent_id=body.agent_id,
        round_num=body.round_num,
        question_type=body.question_type,
        k=body.k,
        domain_filter=body.domain_filter,
    )
