"""
guardrails_enterprise — Python SDK for the Enterprise Guardrails Platform.

Quick-start
-----------
    from guardrails_enterprise import GuardrailsClient

    client = GuardrailsClient()

    # Run a query through the full pipeline
    session = client.query("What are HIPAA encryption requirements?")
    print(session.routing_decision)
    print(session.cse.final_score)

    # Fetch CSE breakdown for a past session
    cse = client.get_cse(session.session_id)
    print(cse.components.f_llm, cse.components.judge_eval)

Services
--------
    Gateway  : http://localhost:8080
    MAD API  : http://localhost:8001
    App API  : http://localhost:8000

Environment variables
---------------------
    GUARDRAILS_APP_URL     — override App Backend URL
    GUARDRAILS_MAD_URL     — override MAD API URL
    GUARDRAILS_GATEWAY_URL — override Gateway URL
"""
from guardrails_enterprise.client import GuardrailsClient
from guardrails_enterprise.types import CSEResult, ComponentScores, SessionResult

__version__ = "0.2.0"

__all__ = [
    "GuardrailsClient",
    "CSEResult",
    "ComponentScores",
    "SessionResult",
]
