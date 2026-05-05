"""
pipeline.py — Full end-to-end pipeline orchestrator.

Flow: Gateway (:8080) → Ollama LLM → MAD (background) → SQLite
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.backend import db

GATEWAY_URL   = "http://localhost:8080"
OLLAMA_URL    = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You are an enterprise compliance assistant. Answer questions about regulatory "
    "requirements accurately and concisely based on your knowledge of healthcare, "
    "banking, and legal regulations. Be concise — 2-4 sentences maximum."
)


async def run_pipeline(query: str, llm_model: str = DEFAULT_MODEL) -> dict:
    """
    Run the full pipeline for a query.

    Gateway + LLM run synchronously and are returned immediately.
    MAD runs as a background asyncio task and updates the session row when done.
    """
    session_id = str(uuid.uuid4())
    t0 = time.time()

    # ── 1. GATEWAY ────────────────────────────────────────────────────────────
    gw_payload = await _call_gateway(query)
    gateway_decision = gw_payload["decision"]
    gateway_score    = gw_payload["gateway_score"]

    llm_answer = None

    # ── 2. LLM  (skip if gateway BLOCK) ──────────────────────────────────────
    if gateway_decision != "BLOCK":
        try:
            llm_answer = await _call_llm(query, llm_model)
        except Exception as e:
            llm_answer = f"[LLM error: {e}]"

    duration_ms = int((time.time() - t0) * 1000)

    # ── 3. PERSIST initial session (MAD fields null, will be updated later) ──
    session = {
        "id":                   session_id,
        "created_at":           datetime.now(timezone.utc).isoformat(),
        "query":                query,
        "gateway_decision":     gateway_decision,
        "gateway_score":        gateway_score,
        "gateway_payload":      json.dumps(gw_payload),
        "llm_answer":           llm_answer,
        "mad_routing":          None,
        "mad_confidence":       None,
        "mad_output_json":      None,
        "mad_query_id":         None,
        "mad_rollout_id":       None,
        "langfuse_trace_id":    None,
        "pipeline_duration_ms": duration_ms,
    }
    db.insert_session(session)

    # ── 4. KICK OFF MAD in background (only if we have a valid LLM answer) ───
    if llm_answer and not llm_answer.startswith("[LLM error"):
        asyncio.create_task(_run_mad_background(session_id, query, llm_answer, t0))

    return db.get_session(session_id)


async def _run_mad_background(session_id: str, query: str, llm_answer: str, t0: float):
    """Background task: run MAD, then patch the session row."""
    try:
        loop = asyncio.get_event_loop()
        mad_out = await loop.run_in_executor(None, _run_mad_sync, query, llm_answer)
        if mad_out:
            # routing_decision is already a plain str in MADOutput
            mad_routing    = mad_out.routing_decision
            mad_confidence = mad_out.aggregate_confidence
            mad_output     = mad_out.model_dump(mode="json")
            duration_ms    = int((time.time() - t0) * 1000)
            db.update_session_mad(
                session_id           = session_id,
                mad_routing          = mad_routing,
                mad_confidence       = mad_confidence,
                mad_output_json      = json.dumps(mad_output),
                mad_query_id         = mad_out.query_id,
                mad_rollout_id       = mad_out.rollout_id,
                pipeline_duration_ms = duration_ms,
            )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("MAD background task failed: %s", exc, exc_info=True)


async def _call_gateway(query: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{GATEWAY_URL}/validate", json={"text": query})
        r.raise_for_status()
        return r.json()


async def _call_llm(query: str, model: str) -> str:
    payload = {
        "model":    model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ],
        "stream":     False,
        "max_tokens": 512,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


def _run_mad_sync(query: str, llm_answer: str):
    """Called in a thread via run_in_executor — run_mad is synchronous."""
    try:
        from multi_agent.mad_pipeline import run_mad
        return run_mad(query, llm_answer)
    except Exception:
        return None
