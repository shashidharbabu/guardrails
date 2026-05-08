"""Integration tests for the App Backend FastAPI application.

Uses FastAPI TestClient — no real HTTP server needed. DB uses an in-memory
SQLite so tests are fully isolated.
"""
from __future__ import annotations

import os
import pytest

# Set env vars BEFORE importing the app so Settings picks them up.
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("DISABLE_AUTH", "true")
os.environ.setdefault("SQLITE_DB_PATH", ":memory:")
os.environ.setdefault("GATEWAY_URL", "http://localhost:8080")
os.environ.setdefault("MAD_API_URL", "http://localhost:8001")
os.environ.setdefault("FEEDBACK_API_URL", "http://localhost:8002")

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.backend.main import app
    from app.backend import db
    db.init_db()
    return TestClient(app)


class TestHealthEndpoints:
    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_livez(self, client):
        r = client.get("/livez")
        assert r.status_code == 200

    def test_readyz(self, client):
        r = client.get("/readyz")
        assert r.status_code == 200


class TestSessionsRouter:
    def test_list_sessions_empty(self, client):
        r = client.get("/api/sessions")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, (list, dict))

    def test_get_nonexistent_session_404(self, client):
        r = client.get("/api/sessions/nonexistent-id")
        assert r.status_code == 404


class TestAnalyticsRouter:
    def test_summary_returns_200(self, client):
        r = client.get("/api/analytics/summary")
        assert r.status_code == 200

    def test_analytics_body_is_dict(self, client):
        r = client.get("/api/analytics/summary")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, dict)


class TestAuditRouter:
    def test_audit_logs_returns_200(self, client):
        # Actual path is /api/audit/logs
        r = client.get("/api/audit/logs")
        assert r.status_code == 200

    def test_audit_logs_is_list(self, client):
        r = client.get("/api/audit/logs")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, (list, dict))


class TestSystemRouter:
    def test_system_health_returns_200(self, client):
        r = client.get("/api/system/health")
        assert r.status_code == 200

    def test_system_health_shape(self, client):
        r = client.get("/api/system/health")
        body = r.json()
        assert isinstance(body, dict)


class TestHumanReviewRouter:
    def test_review_queue_returns_200(self, client):
        # Actual path is /api/human-review/queue
        r = client.get("/api/human-review/queue")
        assert r.status_code == 200

    def test_reviews_list_returns_200(self, client):
        r = client.get("/api/human-review/reviews")
        assert r.status_code == 200


class TestGatewayRouter:
    def test_gateway_health_returns_502_when_down(self, client):
        # Gateway not running in test — expect proxy error or cached result
        r = client.get("/api/gateway/health")
        assert r.status_code in (200, 502, 503, 504)

    def test_gateway_stats_returns_200_or_502(self, client):
        r = client.get("/api/gateway/stats")
        assert r.status_code in (200, 502, 503, 504)

    def test_gateway_config_returns_200_or_502(self, client):
        r = client.get("/api/gateway/config")
        assert r.status_code in (200, 502, 503, 504)


class TestRLHFRouter:
    def test_rlhf_health_returns_502_when_feedback_down(self, client):
        r = client.get("/api/rlhf/health")
        assert r.status_code in (200, 502)

    def test_review_next_returns_502_when_feedback_down(self, client):
        r = client.get("/api/rlhf/review/next")
        assert r.status_code in (200, 502)


class TestFeedbackRouter:
    def test_feedback_list_returns_200(self, client):
        # /api/feedback (GET)
        r = client.get("/api/feedback")
        assert r.status_code == 200
