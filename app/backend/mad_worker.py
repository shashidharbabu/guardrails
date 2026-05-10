"""Subprocess worker for V4MAD session updates.

Keeping V4MAD outside the FastAPI process avoids blocking UI polling while
local embedding/reranker models load and keeps Apple MPS work on a main thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from app.backend import db
from app.backend.pipeline import _run_mad_async, _sanitise_json_str

logger = logging.getLogger(__name__)


def _duration_ms(created_at: str) -> int:
    try:
        started = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return int((datetime.now(started.tzinfo) - started).total_seconds() * 1000)
    except Exception:
        return 0


async def _run(session_id: str, request_id: str | None) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=False)

    session = db.get_session(session_id)
    if not session:
        raise RuntimeError(f"Session not found: {session_id}")

    db.insert_session_event(
        session_id=session_id,
        stage="mad",
        to_status="MAD_RUNNING",
        from_status="MAD_QUEUED",
        request_id=request_id,
    )

    try:
        mad_out = await _run_mad_async(session["query"], session["llm_answer"])
        if not mad_out:
            return

        mad_routing = mad_out.routing_decision
        mad_confidence = mad_out.aggregate_confidence
        mad_output = mad_out.model_dump(mode="json")
        cse_result = getattr(mad_out, "cse_result", None)
        db.update_session_mad(
            session_id=session_id,
            mad_routing=mad_routing,
            mad_confidence=mad_confidence,
            mad_output_json=_sanitise_json_str(json.dumps(mad_output)),
            mad_query_id=getattr(mad_out, "query_id", "") or "",
            mad_rollout_id=getattr(mad_out, "rollout_id", "") or "",
            pipeline_duration_ms=_duration_ms(session.get("created_at", "")),
            cse_result_json=json.dumps(cse_result) if isinstance(cse_result, dict) else None,
            langfuse_trace_id=getattr(mad_out, "langfuse_trace_id", None),
        )
        db.insert_session_event(
            session_id=session_id,
            stage="mad",
            to_status="MAD_COMPLETED",
            from_status="MAD_RUNNING",
            message=f"MAD routing: {mad_routing}",
            request_id=request_id,
        )
    except Exception as exc:
        db.insert_session_event(
            session_id=session_id,
            stage="mad",
            to_status="FAILED",
            from_status="MAD_RUNNING",
            error_message=str(exc),
            request_id=request_id,
        )
        db.update_session_status(session_id, "FAILED", error_message=str(exc))
        raise


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m app.backend.mad_worker <session_id> [request_id]", file=sys.stderr)
        return 2
    session_id = sys.argv[1]
    request_id = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    t0 = time.time()
    try:
        asyncio.run(_run(session_id, request_id))
        print(f"MAD worker completed {session_id} in {time.time() - t0:.1f}s", flush=True)
        return 0
    except Exception as exc:
        print(f"MAD worker failed {session_id}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
