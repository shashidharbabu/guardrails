"""
guardrails_enterprise — Enterprise AI guardrails SDK.

Provides a model-agnostic three-layer guardrail system:
  Layer 1 (Input):      Gateway — blocks PII, jailbreaks, and prompt injection
  Layer 2 (Retrieval):  RAG     — Qdrant + Nemotron-8B + SLM verifier retrieves
                                  grounded regulatory evidence
  Layer 3 (Output):     MAD     — multi-agent debate verifies LLM output against
                                  retrieved regulatory corpus

Quickstart — full pipeline
--------------------------
::

    import asyncio
    from guardrails_enterprise import GuardrailPipeline, SDKConfig

    pipeline = GuardrailPipeline()
    result = asyncio.run(pipeline.run("Does HIPAA require AES-256 encryption?"))

    print(result.gateway.decision)      # "PASS"
    print(result.llm_answer)            # LLM response text
    # MAD runs in background — poll result.mad after it completes

Gateway-only (input screening)
-------------------------------
::

    gw = asyncio.run(pipeline.check("Ignore all previous instructions..."))
    print(gw.decision)      # "BLOCK"
    print(gw.gateway_score) # 0.92

RAG retrieval only (evidence inspection)
----------------------------------------
::

    rag = asyncio.run(pipeline.retrieve("Does HIPAA require AES-256 encryption?"))
    print(rag.sufficient_context)  # True
    print(rag.confidence)          # 0.73
    print(rag.grounded_summary)    # "HIPAA addresses encryption..."
    for chunk in rag.top_chunks:
        print(chunk["chunk_id"], chunk["why_selected"])

MAD output verification only
-----------------------------
::

    answer = "HIPAA mandates AES-256 encryption for all ePHI."
    mad = asyncio.run(pipeline.verify("Does HIPAA require encryption?", answer))
    print(mad.routing_decision)      # "HARD_BLOCK" — fabricated mandate
    print(mad.aggregate_confidence)  # 0.0

RAG + MAD combined (batch evaluation)
--------------------------------------
::

    rag, mad = asyncio.run(
        pipeline.retrieve_and_verify(
            "Does HIPAA require AES-256?",
            "HIPAA mandates AES-256 encryption for all ePHI.",
        )
    )
    print(rag.sufficient_context)   # True
    print(mad.routing_decision)     # "HARD_BLOCK"
"""

from guardrails_enterprise.config import SDKConfig
from guardrails_enterprise.exceptions import (
    GatewayBlockedError,
    GatewayConnectionError,
    GuardrailError,
    LLMError,
    MADError,
    RAGError,
)
from guardrails_enterprise.pipeline import GuardrailPipeline
from guardrails_enterprise.types import GatewayResult, MADResult, PipelineResult, RAGResult

__version__ = "0.1.0"
__all__ = [
    # Core
    "GuardrailPipeline",
    "SDKConfig",
    # Result types
    "PipelineResult",
    "GatewayResult",
    "MADResult",
    "RAGResult",
    # Exceptions
    "GuardrailError",
    "GatewayBlockedError",
    "GatewayConnectionError",
    "LLMError",
    "MADError",
    "RAGError",
]
