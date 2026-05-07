"""
FastAPI gateway server.

Start with: uvicorn gateway.server:app --reload --port 8080
  OR from inside gateway/: uvicorn server:app --reload --port 8080
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Optional APM: ddtrace.auto can fail on some Python/ddtrace pairs (e.g. 3.9 + recent ddtrace).
if os.getenv("DD_TRACE_AUTO_INSTRUMENT", "true").lower() in ("1", "true", "yes"):
    if os.getenv("DD_TRACE_ENABLED", "true").lower() in ("1", "true", "yes"):
        try:
            import ddtrace.auto  # noqa: F401 — FastAPI/Starlette patches when ddtrace is installed
        except Exception as exc:
            print(f"[Gateway] ddtrace.auto not loaded (APM optional): {exc}", flush=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gateway import logger as event_logger
from gateway.gateway import GuardrailGateway
from gateway import telemetry as gw_telemetry

app = FastAPI(
    title="Guardrail Gateway",
    description=(
        "Enterprise AI security gateway. "
        "Detects PII exfiltration, jailbreak attempts, and prompt injection "
        "before requests reach downstream systems."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_gateway = GuardrailGateway(
    pii_model_path=os.environ.get("PII_MODEL_PATH"),
    threat_model_path=os.environ.get("THREAT_MODEL_PATH"),
    pi_model_path=os.environ.get("PROMPT_INJECTION_MODEL_PATH"),
)


class ValidateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="User input to validate")
    user_id: Optional[str] = Field(None, description="Optional user identifier for logs")
    session_id: Optional[str] = Field(None, description="Optional session identifier")


class ScoreBreakdown(BaseModel):
    pii: float
    jailbreak: float
    prompt_injection: float


class ValidateResponse(BaseModel):
    decision: str
    allowed: bool
    gateway_score: float
    scores: ScoreBreakdown
    pii_entities: List[dict]
    threat_types: List[str]
    blocked_reason: Optional[str]


@app.post("/validate", response_model=ValidateResponse)
def validate_input(request: ValidateRequest):
    """Main gateway endpoint."""
    trace_hint = request.session_id or request.user_id or str(uuid.uuid4())
    with gw_telemetry.span(
        "gateway.http.validate",
        trace_id=trace_hint,
        session_id=request.session_id or "",
        user_id=request.user_id or "",
        text_len=len(request.text),
    ):
        result = _gateway.process(request.text, trace_id=trace_hint)
        resp = ValidateResponse(
            decision=result.decision.value,
            allowed=result.is_allowed,
            gateway_score=result.gateway_score,
            scores=ScoreBreakdown(
                pii=result.pii_score,
                jailbreak=result.jb_score,
                prompt_injection=result.pi_score,
            ),
            pii_entities=result.pii_entities,
            threat_types=result.threat_types,
            blocked_reason=result.blocked_reason,
        )
        gw_telemetry.tag_current_span(
            decision=result.decision.value,
            gateway_score=result.gateway_score,
            allowed=result.is_allowed,
            pii_score=result.pii_score,
            jb_score=result.jb_score,
            pi_score=result.pi_score,
        )
        return resp


@app.get("/health")
def health():
    from gateway.validators.pii_validator import CustomPIIValidator
    from gateway.validators.threat_validator import CustomThreatValidator
    from gateway.validators.pi_validator import CustomPIValidator
    return {
        "status": "ok",
        "models_loaded": {
            "pii": bool(CustomPIIValidator._PIPELINE_CACHE),
            "jailbreak": bool(CustomThreatValidator._PIPELINE_CACHE),
            "prompt_injection": bool(CustomPIValidator._PIPELINE_CACHE)
            or bool(CustomPIValidator._CAUSAL_CACHE),
        },
    }


@app.get("/health/trace-ping")
def trace_ping():
    """
    Always emits a gateway trace span.
    Use this when validating Datadog APM wiring without running heavy model inference.
    """
    with gw_telemetry.span("gateway.observability.ping", trace_id="gateway-ping"):
        gw_telemetry.tag_current_span(ping=True)
    return {"ok": True, "emitted": "gateway.observability.ping"}


@app.get("/health/observability")
def health_observability():
    """Datadog-oriented diagnostics when APM traces do not appear in the UI."""
    try:
        import ddtrace

        dd_version = getattr(ddtrace, "__version__", "unknown")
    except ImportError:
        dd_version = None
    return {
        "dd_service": os.getenv("DD_SERVICE", "gateway"),
        "dd_env": os.getenv("DD_ENV", ""),
        "dd_trace_enabled": os.getenv("DD_TRACE_ENABLED", "true"),
        "dd_trace_auto_instrument": os.getenv("DD_TRACE_AUTO_INSTRUMENT", "true"),
        "dd_agent_host": os.getenv("DD_AGENT_HOST", ""),
        "dd_trace_agent_url": os.getenv("DD_TRACE_AGENT_URL", ""),
        "gateway_telemetry_stdout": os.getenv("GATEWAY_TELEMETRY_STDOUT", "false"),
        "manual_tracer_active": gw_telemetry.ddtrace_active(),
        "ddtrace_version": dd_version,
        "hint": (
            "APM needs the Datadog Agent listening for traces (usually localhost:8126). "
            "Start the gateway with: ddtrace-run python -m uvicorn gateway.server:app --host 127.0.0.1 --port 8080. "
            "In Datadog: APM → Traces, filter service:<DD_SERVICE> env:<DD_ENV>. "
            "If still empty, set DD_TRACE_AGENT_URL=http://127.0.0.1:8126 or run DD_TRACE_AGENT_URL with your Agent URL."
        ),
    }


@app.get("/logs")
def get_logs(limit: int = 20):
    events = event_logger.get_recent_events(limit=limit)
    return {"events": events, "count": len(events)}


@app.get("/stats")
def get_stats():
    return event_logger.get_stats()
