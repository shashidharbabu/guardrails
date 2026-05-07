"""
guardrails_enterprise/client.py — GuardrailsClient (HTTP client mode).

Talks to the three deployed services:
  App Backend  (:8000)  — sessions, CSE, analytics
  MAD API      (:8001)  — direct MAD verification
  Gateway      (:8080)  — direct input validation

Use GuardrailsPipeline instead if you want everything in-process (no services).
"""
from __future__ import annotations

import os
from typing import List, Optional

import httpx
import requests

from guardrails_enterprise.types import CSEResult, GatewayResult, PipelineResult, SessionResult, MADSummary, GatewayResult

APP_URL     = os.getenv("GUARDRAILS_APP_URL",     "http://localhost:8000")
MAD_URL     = os.getenv("GUARDRAILS_MAD_URL",     "http://localhost:8001")
GATEWAY_URL = os.getenv("GUARDRAILS_GATEWAY_URL", "http://localhost:8080")
TIMEOUT     = float(os.getenv("GUARDRAILS_TIMEOUT", "120"))


class GuardrailsClient:
    """
    HTTP client for the Enterprise Guardrails Platform.

    Calls the three deployed services rather than running logic in-process.
    Both GuardrailsClient and GuardrailsPipeline return PipelineResult / SessionResult
    with the same CSEResult type, so your application code stays the same regardless
    of which mode you use.

    Usage
    -----
        client = GuardrailsClient()
        result = client.run("What does GDPR Article 17 require?")
        print(result.cse.final_score)
    """

    def __init__(
        self,
        app_url:     str   = APP_URL,
        mad_url:     str   = MAD_URL,
        gateway_url: str   = GATEWAY_URL,
        timeout:     float = TIMEOUT,
    ) -> None:
        self.app_url     = app_url.rstrip("/")
        self.mad_url     = mad_url.rstrip("/")
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout     = timeout

    # ── Primary API ───────────────────────────────────────────────────────────

    def run(self, query: str, llm_model: str = "qwen2.5:7b") -> SessionResult:
        """
        Submit a query through the full pipeline (Gateway → LLM → MAD → CSE).

        Equivalent to GuardrailsPipeline.run() but via HTTP.
        Returns immediately; MAD/CSE may still be running in the background.
        Poll get_session(session_id) until cse is populated.
        """
        return self.query(query, llm_model)

    def query(self, text: str, llm_model: str = "qwen2.5:7b") -> SessionResult:
        """Submit a query and return the initial session result."""
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
        Retrieve the full CSE breakdown for a completed session.
        Returns None if MAD hasn't completed yet for this session.
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
        Retrieve CSE directly from the MAD API using mad_query_id.
        Useful when you have the MAD query_id but not the session_id.
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
        Returns the raw MAD response dict including cse_result.
        """
        resp = requests.post(
            f"{self.mad_url}/mad/verify",
            json={"query": query, "llm_answer": llm_answer},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Gateway direct ────────────────────────────────────────────────────────

    def validate(self, text: str) -> GatewayResult:
        """
        Call the Gateway directly to validate an input.
        Returns a typed GatewayResult.
        """
        resp = requests.post(
            f"{self.gateway_url}/validate",
            json={"text": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return GatewayResult.from_dict(resp.json())

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Check liveness of all three services. Returns dict of service → status."""
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
