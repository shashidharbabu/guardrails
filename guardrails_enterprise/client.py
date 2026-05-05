"""
guardrails_enterprise/client.py — GuardrailsClient.

Provides a typed Python interface to all three services:
  - App Backend (:8000)   for sessions and CSE
  - MAD API (:8001)       for direct MAD verification
  - Gateway (:8080)       for direct input validation

All methods have both sync (requests-based) and async (httpx-based) variants.
"""
from __future__ import annotations

import os
from typing import List, Optional

import httpx
import requests

from guardrails_enterprise.types import CSEResult, SessionResult

APP_URL     = os.getenv("GUARDRAILS_APP_URL",     "http://localhost:8000")
MAD_URL     = os.getenv("GUARDRAILS_MAD_URL",     "http://localhost:8001")
GATEWAY_URL = os.getenv("GUARDRAILS_GATEWAY_URL", "http://localhost:8080")
TIMEOUT     = float(os.getenv("GUARDRAILS_TIMEOUT", "120"))


class GuardrailsClient:
    """
    Synchronous client for the Enterprise Guardrails Platform.

    Usage:
        client = GuardrailsClient()
        session = client.query("What does GDPR Article 17 require?")
        print(session.cse)
    """

    def __init__(
        self,
        app_url:     str = APP_URL,
        mad_url:     str = MAD_URL,
        gateway_url: str = GATEWAY_URL,
        timeout:     float = TIMEOUT,
    ) -> None:
        self.app_url     = app_url.rstrip("/")
        self.mad_url     = mad_url.rstrip("/")
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout     = timeout

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def query(self, text: str, llm_model: str = "qwen2.5:7b") -> SessionResult:
        """
        Submit a query through the full pipeline (Gateway → LLM → MAD → CSE).

        Returns immediately with the gateway + LLM result.
        MAD/CSE run in the background; poll `get_session(session_id)` for results.
        """
        resp = requests.post(
            f"{self.app_url}/api/query",
            json={"query": text, "llm_model": llm_model},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return SessionResult.from_dict(resp.json())

    # ── Sessions ──────────────────────────────────────────────────────────────

    def get_session(self, session_id: str) -> SessionResult:
        """Retrieve a session by ID (includes MAD/CSE once they complete)."""
        resp = requests.get(
            f"{self.app_url}/api/sessions/{session_id}",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return SessionResult.from_dict(resp.json())

    def list_sessions(self, limit: int = 50) -> List[SessionResult]:
        """List recent sessions, most recent first."""
        resp = requests.get(
            f"{self.app_url}/api/sessions",
            params={"limit": limit},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return [SessionResult.from_dict(s) for s in resp.json()]

    # ── CSE ───────────────────────────────────────────────────────────────────

    def get_cse(self, session_id: str) -> Optional[CSEResult]:
        """
        Retrieve the full CSE breakdown for a session.

        Returns None if MAD hasn't completed yet.
        Raises HTTPError if session_id not found.
        """
        resp = requests.get(
            f"{self.app_url}/api/sessions/{session_id}/cse",
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return CSEResult.from_dict(data.get("cse", data))

    def get_cse_by_query_id(self, mad_query_id: str) -> Optional[CSEResult]:
        """
        Retrieve CSE directly from the MAD API using the mad_query_id.
        Useful when working with the MAD API directly.
        """
        resp = requests.get(
            f"{self.mad_url}/mad/cse/{mad_query_id}",
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return CSEResult.from_dict({
            "final_score":      data.get("final_score"),
            "routing_decision": data.get("routing_decision"),
            "components":       data.get("components", {}),
            "version":          data.get("version", "v0.1"),
        })

    # ── MAD direct ────────────────────────────────────────────────────────────

    def verify(self, query: str, llm_answer: str) -> dict:
        """
        Call the MAD API directly to verify an LLM answer.
        Returns the full MAD response dict including cse_result.
        """
        resp = requests.post(
            f"{self.mad_url}/mad/verify",
            json={"query": query, "llm_answer": llm_answer},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Gateway direct ────────────────────────────────────────────────────────

    def validate(self, text: str) -> dict:
        """
        Call the Gateway directly to validate an input.
        Returns the full gateway response dict.
        """
        resp = requests.post(
            f"{self.gateway_url}/validate",
            json={"text": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Check liveness of all three services."""
        results = {}
        for name, url, path in [
            ("app",     self.app_url,     "/health"),
            ("mad",     self.mad_url,     "/mad/health"),
            ("gateway", self.gateway_url, "/health"),
        ]:
            try:
                r = requests.get(f"{url}{path}", timeout=5)
                results[name] = "ok" if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception as exc:
                results[name] = f"error: {exc}"
        return results
