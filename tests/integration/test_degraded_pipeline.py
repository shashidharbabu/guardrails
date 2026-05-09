from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_mad_none_marks_session_unavailable():
    from app.backend import db
    from app.backend.pipeline import _run_mad_background

    db.init_db()
    session_id = "test-mad-unavailable"
    db.insert_session(
        {
            "id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query": "What does HIPAA require?",
            "status": "LLM_COMPLETED",
            "gateway_decision": "PASS",
            "gateway_score": 0.0,
            "gateway_payload": json.dumps({"decision": "PASS"}),
            "llm_answer": "HIPAA requires safeguards for protected health information.",
            "llm_model": "test-model",
            "mad_routing": None,
            "mad_confidence": None,
            "mad_output_json": None,
            "pipeline_duration_ms": 10,
            "cse_result_json": None,
            "final_route": None,
        }
    )

    with patch("app.backend.pipeline._run_mad_async", new=AsyncMock(return_value=None)):
        await _run_mad_background(
            session_id=session_id,
            query="What does HIPAA require?",
            llm_answer="HIPAA requires safeguards for protected health information.",
            t0=0,
            request_id="test-request",
        )

    session = db.get_session(session_id)
    events = db.get_session_events(session_id)

    assert session["status"] == "MAD_UNAVAILABLE"
    assert any(e["to_status"] == "MAD_UNAVAILABLE" for e in events)
