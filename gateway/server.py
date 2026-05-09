"""
FastAPI gateway server.

Start with: uvicorn gateway.server:app --reload --port 8080
  OR from inside gateway/: uvicorn server:app --reload --port 8080
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gateway import logger as event_logger
from gateway.gateway import GuardrailGateway

app = FastAPI(
    title="Guardrail Gateway",
    description=(
        "Enterprise AI security gateway. "
        "Detects PII exfiltration, jailbreak attempts, and prompt injection "
        "before requests reach downstream systems."
    ),
    version="1.0.0",
)

_cors_origins = [
    o.strip()
    for o in os.environ.get("GATEWAY_CORS_ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)

# ---------------------------------------------------------------------------
# In-memory config state — survives requests, resets on server restart
# ---------------------------------------------------------------------------

_DEFAULT_MODELS = {
    "pii":              os.environ.get("PII_MODEL_PATH", "iiiorg/piiranha-v1-detect-personal-information"),
    "jailbreak":        os.environ.get("THREAT_MODEL_PATH", "jackhhao/jailbreak-classifier"),
    "prompt_injection": os.environ.get("PROMPT_INJECTION_MODEL_PATH", "protectai/deberta-v3-base-prompt-injection-v2"),
}

_DEFAULT_THRESHOLDS = {
    "pii_threshold":         0.5,
    "jb_threshold":          0.4,
    "pi_threshold":          0.4,
    "pass_threshold":        0.3,
    "block_threshold":       0.7,
    "jb_override_threshold": 0.7,
    "pi_override_threshold": 0.7,
    "pii_override_threshold":0.9,
}

_config: Dict = {
    "models":     dict(_DEFAULT_MODELS),
    "thresholds": dict(_DEFAULT_THRESHOLDS),
}


def _build_gateway() -> GuardrailGateway:
    """Create a GuardrailGateway from the current in-memory config."""
    return GuardrailGateway(
        pii_model_path=_config["models"]["pii"],
        threat_model_path=_config["models"]["jailbreak"],
        pi_model_path=_config["models"]["prompt_injection"],
        **_config["thresholds"],
    )


_gateway = _build_gateway()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ValidateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


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
    duration_ms: Optional[int] = None
    judge_verdict: Optional[str] = None
    judge_reason: Optional[str] = None
    judge_threat_type: Optional[str] = None


class ConfigUpdateRequest(BaseModel):
    models: Optional[Dict[str, str]] = None
    thresholds: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/validate", response_model=ValidateResponse)
def validate_input(request: ValidateRequest):
    """Main gateway endpoint — runs all three validators + decision engine."""
    t0 = time.time()
    result = _gateway.process(request.text)
    duration_ms = int((time.time() - t0) * 1000)
    return ValidateResponse(
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
        duration_ms=duration_ms,
        judge_verdict=result.judge_verdict,
        judge_reason=result.judge_reason,
        judge_threat_type=result.judge_threat_type,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "validators": "hf-serverless (stateless HTTP — no local models)",
        "active_models": _config["models"],
        "judge_enabled": _gateway._judge.enabled,
    }


@app.get("/logs")
def get_logs(limit: int = 50):
    events = event_logger.get_recent_events(limit=limit)
    return {"events": events, "count": len(events)}


@app.get("/stats")
def get_stats():
    return event_logger.get_stats()


@app.get("/config")
def get_config():
    """Return current in-memory thresholds and model paths."""
    return {
        "models":     _config["models"],
        "thresholds": _config["thresholds"],
        "defaults": {
            "models":     _DEFAULT_MODELS,
            "thresholds": _DEFAULT_THRESHOLDS,
        },
    }


@app.patch("/config")
def update_config(body: ConfigUpdateRequest):
    """
    Update thresholds and/or model paths at runtime.

    Threshold-only changes: instant (models stay cached, gateway re-created).
    Model path changes: clears model cache + re-creates gateway (~30-60s first hit).
    """
    global _gateway, _config

    model_changed = False

    # Apply model changes
    if body.models:
        for key, val in body.models.items():
            if key in _config["models"] and val and val != _config["models"][key]:
                _config["models"][key] = val
                model_changed = True

    # Apply threshold changes
    if body.thresholds:
        for key, val in body.thresholds.items():
            if key in _config["thresholds"] and val is not None:
                _config["thresholds"][key] = float(val)

    # Re-create gateway — validators are stateless HTTP clients, model path change is instant
    _gateway = _build_gateway()

    return {
        "status": "ok",
        "config": _config,
        "model_changed": model_changed,
    }


@app.post("/config/reset")
def reset_config():
    """Reset all config back to server startup defaults."""
    global _gateway, _config
    _config["models"]     = dict(_DEFAULT_MODELS)
    _config["thresholds"] = dict(_DEFAULT_THRESHOLDS)
    _gateway = _build_gateway()
    return {"status": "ok", "config": _config}
