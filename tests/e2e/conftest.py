"""
tests/e2e/conftest.py — Shared fixtures for E2E tests.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import pytest


def wait_for_mad(
    client,
    session_id: str,
    timeout: int = 120,
    poll_interval: float = 2.0,
) -> Optional[dict]:
    """
    Poll GET /api/sessions/{session_id} until mad_routing is set or timeout.
    Returns the final session dict, or None if timed out.

    Usage:
        session = wait_for_mad(backend_client, session_id, timeout=60)
        assert session["mad_routing"] == "DELIVER"
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/sessions/{session_id}")
        if r.status_code == 200:
            body = r.json()
            if body.get("mad_routing") is not None:
                return body
        time.sleep(poll_interval)
    return None


@pytest.fixture
def wait_for_mad_fixture(backend_client):
    """Fixture version — injects backend_client automatically."""
    def _wait(session_id: str, timeout: int = 120) -> Optional[dict]:
        return wait_for_mad(backend_client, session_id, timeout=timeout)
    return _wait
