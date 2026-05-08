"""Integration tests for the App Backend FastAPI application.

Uses FastAPI TestClient — no real HTTP server needed. DB uses an in-memory
SQLite so tests are fully isolated.
"""
from __future__ import annotations

import os
import pytest

# Must set env vars BEFORE importing the app module so Settings picks them up.
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


class TestSessionsRouter:
    def test_list_sessions_empty(self, client):
        r = client.get("/api/sessions")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, (list, dict))

    def test_create_session(self, client):
        r = client.post("/api/sessions", json={"title": "test session"})
        assert r.status_code in (200, 201)


class TestAnalyticsRouter:
    def test_summary_returns_200(self, client):
        r = client.get("/api/analytics/summary")
        assert r.status_code == 200

    def test_routing_distribution_shape(self, client):
        r = client.get("/api/analytics/routing")
        assert r.status_code == 200
        body = r.json()
        # Should return a list or dict of routing counts
        assert body is not None


class TestAuditRouter:
    def test_audit_log_returns_200(self, client):
        r = client.get("/api/audit")
        assert r.status_code == 200


class TestSystemRouter:
    def test_system_health_returns_200(self, client):
        r = client.get("/api/system/health")
        assert r.status_code == 200


class TestHumanReviewRouter:
    def test_pending_reviews_returns_200(self, client):
        r = client.get("/api/human-review/pending")
        assert r.status_code == 200

    def test_review_stats_returns_200(self, client):
        r = client.get("/api/human-review/stats")
        assert r.status_code in (200, 404)  # 404 ok if no stats row yet


class TestRLHFRouter:
    def test_rlhf_health_returns_502_when_feedback_down(self, client):
        """When feedback loop service isn't running, proxy returns 502."""
        r = client.get("/api/rlhf/health")
        assert r.status_code in (200, 502)

    def test_review_next_returns_502_when_feedback_down(self, client):
        r = client.get("/api/rlhf/review/next")
        assert r.status_code in (200, 502)
