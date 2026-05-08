"""Smoke tests — hit live services and check health endpoints.

Run ONLY when all services are up:
    ./start.sh all
    pytest tests/smoke/ -v

These tests are deliberately excluded from CI (see pytest marks below).
"""
from __future__ import annotations

import pytest
import httpx

pytestmark = pytest.mark.smoke  # `pytest -m "not smoke"` skips this file


SERVICES = {
    "gateway":  "http://localhost:8080",
    "mad":      "http://localhost:8001",
    "backend":  "http://localhost:8000",
    "feedback": "http://localhost:8002",
    "frontend": "http://localhost:5173",
}

HEALTH_PATHS = {
    "gateway":  "/health",
    "mad":      "/mad/health",
    "backend":  "/healthz",
    "feedback": "/health",
    "frontend": "/",
}


@pytest.fixture(scope="module")
def http():
    return httpx.Client(timeout=10.0)


@pytest.mark.parametrize("service", SERVICES.keys())
def test_service_health(http, service):
    url = SERVICES[service] + HEALTH_PATHS[service]
    try:
        r = http.get(url)
        assert r.status_code in (200, 404)  # 404 for / on frontend is still "up"
    except httpx.ConnectError:
        pytest.fail(f"{service} not reachable at {url}")


def test_backend_readyz(http):
    r = http.get(f"{SERVICES['backend']}/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_gateway_processes_clean_query(http):
    r = http.post(
        f"{SERVICES['gateway']}/process",
        json={"query": "What does HIPAA 45 CFR §164 require for data security?"},
        timeout=30.0,
    )
    assert r.status_code == 200
    body = r.json()
    assert "decision" in body or "is_allowed" in body


def test_mad_verify_smoke(http):
    r = http.post(
        f"{SERVICES['mad']}/mad/verify",
        json={
            "query": "What does HIPAA require?",
            "llm_answer": "HIPAA requires healthcare organisations to implement safeguards.",
        },
        timeout=120.0,
    )
    assert r.status_code == 200
    body = r.json()
    assert "routing" in body
    assert body["routing"] in ("DELIVER", "RETRY", "HUMAN_REVIEW", "HARD_BLOCK")


def test_backend_submit_query(http):
    """End-to-end: submit a query through the full stack via the backend."""
    # First get or create a session
    sessions = http.get(f"{SERVICES['backend']}/api/sessions").json()
    if isinstance(sessions, list) and sessions:
        session_id = sessions[0]["session_id"]
    else:
        s = http.post(f"{SERVICES['backend']}/api/sessions", json={"title": "smoke"})
        session_id = s.json()["session_id"]

    r = http.post(
        f"{SERVICES['backend']}/api/query",
        json={
            "session_id": session_id,
            "query": "What are HIPAA's main privacy requirements?",
        },
        timeout=180.0,
    )
    assert r.status_code in (200, 201, 202)
