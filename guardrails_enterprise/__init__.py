"""
guardrails_enterprise — Enterprise AI Guardrails SDK
====================================================

Installation
------------
    pip install guardrails-enterprise                    # core (client mode only)
    pip install guardrails-enterprise[gateway]           # + input validation
    pip install guardrails-enterprise[eval]              # + DeepEval CSE scoring
    pip install guardrails-enterprise[all]               # everything

Quick-start: library mode (in-process, no servers except Ollama)
-----------------------------------------------------------------
    from guardrails_enterprise import run_pipeline

    result = run_pipeline("What does HIPAA require for PHI encryption?")
    print(result.routing)           # DELIVER
    print(result.confidence)        # 0.87
    print(result.cse)               # CSEResult(score=0.87, routing=DELIVER, ...)
    print(result.llm_answer)        # "HIPAA requires covered entities to..."

    # Or with more control:
    from guardrails_enterprise import GuardrailsPipeline

    pipe = GuardrailsPipeline(
        ollama_url    = "http://localhost:11434",
        ollama_model  = "qwen2.5:7b",
        skip_gateway  = False,
        skip_deepeval = False,
    )
    result = pipe.run("What does GDPR Article 17 require?")

Quick-start: client mode (HTTP, talk to deployed services)
-----------------------------------------------------------
    from guardrails_enterprise import GuardrailsClient

    client = GuardrailsClient(
        app_url     = "https://guardrails.my-company.com",
        mad_url     = "https://mad.my-company.com",
        gateway_url = "https://gateway.my-company.com",
    )
    session = client.query("What does CCPA require?")
    print(session.cse)              # CSEResult(...)

    # Retrieve CSE for a past session:
    cse = client.get_cse(session.session_id)

Environment variables
---------------------
    OLLAMA_BASE_URL          — Ollama server (default: http://localhost:11434)
    VERIFIER_MODEL           — Ollama model  (default: qwen2.5:7b)
    GUARDRAILS_APP_URL       — App Backend   (default: http://localhost:8000)
    GUARDRAILS_MAD_URL       — MAD API       (default: http://localhost:8001)
    GUARDRAILS_GATEWAY_URL   — Gateway       (default: http://localhost:8080)

CLI
---
    guardrails-run "What does HIPAA require for PHI encryption?"
    guardrails-run "query" --skip-deepeval --json
    guardrails-run --health
"""

from guardrails_enterprise.pipeline import GuardrailsPipeline, run_pipeline
from guardrails_enterprise.client import GuardrailsClient
from guardrails_enterprise.types import (
    PipelineResult,
    GatewayResult,
    MADSummary,
    CSEResult,
    ComponentScores,
    SessionResult,
)

__version__ = "0.2.0"

__all__ = [
    # Library mode
    "run_pipeline",
    "GuardrailsPipeline",
    # Client mode
    "GuardrailsClient",
    # Types
    "PipelineResult",
    "GatewayResult",
    "MADSummary",
    "CSEResult",
    "ComponentScores",
    "SessionResult",
]
