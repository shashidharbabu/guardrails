"""
pipeline.py — Full end-to-end pipeline orchestrator.

Flow: Gateway → LLM → MAD (background asyncio task) → DB
All service URLs come from the settings module — no hardcoded localhost.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.backend import db
from app.backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = (
    "You are an enterprise compliance assistant. Answer questions about regulatory "
    "requirements accurately and concisely based on your knowledge of healthcare, "
    "banking, and legal regulations. Be concise — 2-4 sentences maximum."
)


async def run_pipeline(
    query: str,
    llm_model: Optional[str] = None,
    request_id: Optional[str] = None,
) -> dict:
    """
    Run the full pipeline for a query.
    Gateway + LLM run synchronously and are returned immediately.
    MAD runs as a background asyncio task and updates the session row when done.
    """
    session_id = str(uuid.uuid4())
    model = llm_model or settings.DEFAULT_LLM_MODEL
    t0 = time.time()

    # ── 1. GATEWAY ────────────────────────────────────────────────────────────
    db.insert_session_event(
        session_id=session_id,
        stage="gateway",
        to_status="GATEWAY_RUNNING",
        request_id=request_id,
    )
    gw_payload = await _call_gateway(query)
    gateway_decision = gw_payload["decision"]
    gateway_score = gw_payload.get("gateway_score", gw_payload.get("threat_score", 0.0))

    gateway_status = {
        "PASS": "GATEWAY_PASSED",
        "BLOCK": "GATEWAY_BLOCKED",
        "ESCALATE": "GATEWAY_ESCALATED",
    }.get(gateway_decision, "GATEWAY_PASSED")

    llm_answer = None

    # ── 2. LLM (skip if gateway BLOCK) ────────────────────────────────────────
    if gateway_decision != "BLOCK":
        db.insert_session_event(
            session_id=session_id,
            stage="llm",
            to_status="LLM_RUNNING",
            from_status=gateway_status,
            request_id=request_id,
        )
        try:
            llm_answer = await _call_llm(query, model)
            db.insert_session_event(
                session_id=session_id,
                stage="llm",
                to_status="LLM_COMPLETED",
                from_status="LLM_RUNNING",
                request_id=request_id,
            )
        except Exception as e:
            llm_answer = f"[LLM error: {e}]"
            db.insert_session_event(
                session_id=session_id,
                stage="llm",
                to_status="FAILED",
                from_status="LLM_RUNNING",
                error_message=str(e),
                request_id=request_id,
            )

    duration_ms = int((time.time() - t0) * 1000)
    initial_status = gateway_status if gateway_decision == "BLOCK" else "LLM_COMPLETED"

    # ── 3. PERSIST initial session ────────────────────────────────────────────
    session = {
        "id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "status": initial_status,
        "gateway_decision": gateway_decision,
        "gateway_score": gateway_score,
        "gateway_payload": json.dumps(gw_payload),
        "llm_answer": llm_answer,
        "llm_model": model,
        "mad_routing": None,
        "mad_confidence": None,
        "mad_output_json": None,
        "mad_query_id": None,
        "mad_rollout_id": None,
        "langfuse_trace_id": None,
        "pipeline_duration_ms": duration_ms,
        "cse_result_json": None,
        "final_route": gateway_status if gateway_decision == "BLOCK" else None,
    }
    db.insert_session(session)

    # ── 4. KICK OFF MAD in background ─────────────────────────────────────────
    if llm_answer and not llm_answer.startswith("[LLM error"):
        db.insert_session_event(
            session_id=session_id,
            stage="mad",
            to_status="MAD_QUEUED",
            from_status="LLM_COMPLETED",
            request_id=request_id,
        )
        asyncio.create_task(
            _run_mad_background(session_id, query, llm_answer, t0, request_id)
        )

    return db.get_session(session_id)


async def _run_mad_background(
    session_id: str,
    query: str,
    llm_answer: str,
    t0: float,
    request_id: Optional[str] = None,
):
    db.insert_session_event(
        session_id=session_id,
        stage="mad",
        to_status="MAD_RUNNING",
        from_status="MAD_QUEUED",
        request_id=request_id,
    )
    try:
        mad_out = await _run_mad_async(query, llm_answer)
        if mad_out:
            mad_routing = mad_out.routing_decision
            mad_confidence = mad_out.aggregate_confidence
            mad_output = mad_out.model_dump(mode="json")
            cse_result = getattr(mad_out, "cse_result", None)
            duration_ms = int((time.time() - t0) * 1000)
            db.update_session_mad(
                session_id=session_id,
                mad_routing=mad_routing,
                mad_confidence=mad_confidence,
                mad_output_json=json.dumps(mad_output),
                mad_query_id=getattr(mad_out, "query_id", "") or "",
                mad_rollout_id=getattr(mad_out, "rollout_id", "") or "",
                pipeline_duration_ms=duration_ms,
                cse_result_json=json.dumps(cse_result) if isinstance(cse_result, dict) else None,
            )
            db.insert_session_event(
                session_id=session_id,
                stage="mad",
                to_status="MAD_COMPLETED",
                from_status="MAD_RUNNING",
                message=f"MAD routing: {mad_routing}",
                request_id=request_id,
            )
            logger.info(
                "mad_completed",
                extra={"session_id": session_id, "mad_routing": mad_routing},
            )
    except Exception as exc:
        logger.error("mad_background_failed", extra={"session_id": session_id, "error": str(exc)}, exc_info=True)
        db.insert_session_event(
            session_id=session_id,
            stage="mad",
            to_status="FAILED",
            from_status="MAD_RUNNING",
            error_message=str(exc),
            request_id=request_id,
        )
        db.update_session_status(session_id, "FAILED", error_message=str(exc))


async def _call_gateway(query: str) -> dict:
    url = f"{settings.GATEWAY_URL}/validate"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json={"text": query})
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        logger.error("gateway_unreachable", extra={"url": url})
        # Fail open with ESCALATE so LLM still runs but is flagged
        return {
            "decision": "ESCALATE",
            "gateway_score": 0.5,
            "error": "gateway_unreachable",
        }


async def _call_llm(query: str, model: str) -> str:
    llm_url = f"{settings.LLM_PROVIDER_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "stream": False,
        "max_tokens": 512,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(llm_url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


async def _run_mad_async(query: str, llm_answer: str):
    """
    Run the full_FinalMAD_with_judge pipeline asynchronously.
    Falls back to the legacy Ollama-based multi_agent pipeline if the new one
    is unavailable (e.g. vLLM not configured), so the app always degrades gracefully.
    """
    if settings.MAD_MODE == "disabled":
        return None
    try:
        import sys
        from pathlib import Path
        _mad_root = Path(__file__).resolve().parent.parent.parent / "multi_agent_debate" / "full_FinalMAD_with_judge"
        if str(_mad_root) not in sys.path:
            sys.path.insert(0, str(_mad_root))
        from api import _run_pipeline  # type: ignore[import]
        return await _run_pipeline(query, llm_answer)
    except Exception as exc:
        logger.warning("new_mad_failed, trying legacy", extra={"error": str(exc)})
        # Legacy fallback: Ollama-based synchronous pipeline
        try:
            loop = asyncio.get_event_loop()
            try:
                from multi_agent.mad_pipeline import run_mad  # type: ignore[import]
            except ImportError:
                from multi_agent_debate.multi_agent.mad_pipeline import run_mad  # type: ignore[import]
            return await loop.run_in_executor(None, run_mad, query, llm_answer)
        except Exception as exc2:
            logger.warning("legacy_mad_also_failed", extra={"error": str(exc2)})
            return None
