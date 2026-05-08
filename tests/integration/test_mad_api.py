"""Integration tests for the MAD API (full_FinalMAD_with_judge).

Uses FastAPI TestClient. LLM calls are mocked — no Ollama needed.
Tests verify API contract and response shape.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_MAD_ROOT = Path(__file__).resolve().parents[2] / "multi_agent_debate" / "full_FinalMAD_with_judge"
sys.path.insert(0, str(_MAD_ROOT))


@pytest.fixture(scope="module")
def mad_client():
    from fastapi.testclient import TestClient
    import api as mad_api
    return TestClient(mad_api.app)


def _fake_mad_response(routing: str = "DELIVER"):
    """Build a valid MADResponse using the real field names from api.py."""
    import api as mad_api
    return mad_api.MADResponse(
        routing_decision=routing,
        aggregate_confidence=0.85,
        claims=[],
        judge_verdicts=[],
        debate_transcript="No debate (mocked).",
        query_id="test-qid",
        rollout_id="test-rid",
    )


class TestMADHealth:
    def test_health_ok(self, mad_client):
        r = mad_client.get("/mad/health")
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "ok"

    def test_info_returns_version(self, mad_client):
        r = mad_client.get("/mad/info")
        assert r.status_code == 200


class TestMADVerifyContract:
    def test_verify_missing_body_returns_422(self, mad_client):
        r = mad_client.post("/mad/verify", json={})
        assert r.status_code == 422

    def test_verify_response_has_required_fields(self, mad_client):
        """Mock the pipeline — verify the API response contract."""
        with patch("api._run_pipeline", new=AsyncMock(return_value=_fake_mad_response())):
            r = mad_client.post(
                "/mad/verify",
                json={
                    "query": "What does the Security Rule require?",
                    "llm_answer": "The Security Rule requires technical safeguards.",
                },
            )
        assert r.status_code == 200
        body = r.json()
        assert "routing_decision" in body
        assert "aggregate_confidence" in body
        assert "claims" in body
        assert body["routing_decision"] in ("DELIVER", "RETRY", "HUMAN_REVIEW", "HARD_BLOCK")

    @pytest.mark.parametrize("routing", ["DELIVER", "HARD_BLOCK", "RETRY", "HUMAN_REVIEW"])
    def test_all_routing_outcomes_return_200(self, mad_client, routing):
        with patch("api._run_pipeline", new=AsyncMock(return_value=_fake_mad_response(routing))):
            r = mad_client.post("/mad/verify", json={"query": "q", "llm_answer": "a"})
        assert r.status_code == 200
        assert r.json()["routing_decision"] == routing

    def test_aggregate_confidence_is_float_in_range(self, mad_client):
        with patch("api._run_pipeline", new=AsyncMock(return_value=_fake_mad_response())):
            r = mad_client.post("/mad/verify", json={"query": "q", "llm_answer": "a"})
        body = r.json()
        assert 0.0 <= float(body["aggregate_confidence"]) <= 1.0
