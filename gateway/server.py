"""
FastAPI gateway server.

Start with: uvicorn gateway.server:app --reload --port 8080
  OR from inside gateway/: uvicorn server:app --reload --port 8080
"""

import os
from typing import List, Optional

from fastapi import FastAPI
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
    result = _gateway.process(request.text)
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
    )


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
            "prompt_injection": bool(CustomPIValidator._PIPELINE_CACHE),
        },
    }


@app.get("/logs")
def get_logs(limit: int = 20):
    events = event_logger.get_recent_events(limit=limit)
    return {"events": events, "count": len(events)}


@app.get("/stats")
def get_stats():
    return event_logger.get_stats()
