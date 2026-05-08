"""Integration tests for the MAD API (full_FinalMAD_with_judge).

These tests call the FastAPI app directly via TestClient. They mock the
LLM calls so no Ollama server is needed. Tests verify the API contract
and pipeline wiring, not LLM output quality.
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
    """Verify /mad/verify response shape without running the full pipeline."""

    def test_verify_missing_body_returns_422(self, mad_client):
        r = mad_client.post("/mad/verify", json={})
        assert r.status_code == 422

    def test_verify_response_shape(self, mad_client):
        """Mock the pipeline so we can test the API contract without Ollama."""
        from api import MADResponse

        fake_response = MADResponse(
            query="test",
            llm_answer="test answer",
            routing="DELIVER",
            agg_confidence=0.85,
            claims=[],
            debate_rounds=[],
            judge_verdicts=[],
            run_id="test-run",
        )

        with patch("api._run_pipeline", new=AsyncMock(return_value=fake_response)):
            r = mad_client.post(
                "/mad/verify",
                json={
                    "query": "What does HIPAA require?",
                    "llm_answer": "HIPAA requires privacy safeguards.",
                },
            )
        assert r.status_code == 200
        body = r.json()
        assert "routing" in body
        assert "agg_confidence" in body
        assert "claims" in body

    @pytest.mark.parametrize("routing,expected_code", [
        ("DELIVER", 200),
        ("HARD_BLOCK", 200),
        ("RETRY", 200),
    ])
    def test_all_routing_outcomes_return_200(self, mad_client, routing, expected_code):
        from api import MADResponse

        fake = MADResponse(
            query="q", llm_answer="a", routing=routing,
            agg_confidence=0.5, claims=[], debate_rounds=[],
            judge_verdicts=[], run_id="r",
        )
        with patch("api._run_pipeline", new=AsyncMock(return_value=fake)):
            r = mad_client.post("/mad/verify", json={"query": "q", "llm_answer": "a"})
        assert r.status_code == expected_code
