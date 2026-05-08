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
    "backend":  "http://localhost:8000",
    "feedback": "http://localhost:8002",
    "frontend": "http://localhost:5173",
}

# MAD now runs in-process inside the backend; standalone service on :8001 is optional.
# Point to backend's /mad/health which proxies through if the standalone is up,
# or use backend health as fallback.
MAD_STANDALONE_URL = "http://localhost:8001"

HEALTH_PATHS = {
    "gateway":  "/health",
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
    """Hits standalone MAD service on :8001. Skipped when not running."""
    try:
        r = http.post(
            f"{MAD_STANDALONE_URL}/mad/verify",
            json={
                "query": "What does HIPAA require?",
                "llm_answer": "HIPAA requires healthcare organisations to implement safeguards.",
            },
            timeout=120.0,
        )
    except httpx.ConnectError:
        pytest.skip("standalone MAD service not running on :8001")
    assert r.status_code == 200
    body = r.json()
    assert "routing_decision" in body
    assert body["routing_decision"] in ("DELIVER", "RETRY", "HUMAN_REVIEW", "HARD_BLOCK")


def test_backend_submit_query(http):
    """End-to-end: submit a query through the full stack via the backend."""
    r = http.post(
        f"{SERVICES['backend']}/api/query",
        json={
            "query": "What are HIPAA's main privacy requirements?",
        },
        timeout=180.0,
    )
    assert r.status_code in (200, 201, 202)
