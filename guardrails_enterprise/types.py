"""
guardrails_enterprise.types — Public result types returned by the SDK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GatewayResult:
    """
    Result of the input gateway check.

    Attributes:
        decision:       "PASS", "ESCALATE", or "BLOCK"
        allowed:        True if request may proceed to LLM
        gateway_score:  Composite risk score 0.0–1.0
        pii_score:      PII detector score
        jailbreak_score: Jailbreak classifier score
        pi_score:       Prompt-injection classifier score
        pii_entities:   List of detected PII entity dicts
        threat_types:   List of detected threat type strings
        blocked_reason: Human-readable reason when decision == BLOCK
        duration_ms:    Gateway latency in milliseconds
        raw:            Full raw JSON response from the gateway service
    """
    decision: str
    allowed: bool
    gateway_score: float
    pii_score: float
    jailbreak_score: float
    pi_score: float
    pii_entities: list[dict] = field(default_factory=list)
    threat_types: list[str] = field(default_factory=list)
    blocked_reason: str | None = None
    duration_ms: int | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict) -> "GatewayResult":
        scores = data.get("scores", {})
        return cls(
            decision=data["decision"],
            allowed=data["allowed"],
            gateway_score=data["gateway_score"],
            pii_score=scores.get("pii", 0.0),
            jailbreak_score=scores.get("jailbreak", 0.0),
            pi_score=scores.get("prompt_injection", 0.0),
            pii_entities=data.get("pii_entities", []),
            threat_types=data.get("threat_types", []),
            blocked_reason=data.get("blocked_reason"),
            duration_ms=data.get("duration_ms"),
            raw=data,
        )


@dataclass
class MADResult:
    """
    Result of the Multi-Agent Debate output verification.

    Attributes:
        routing_decision:    "DELIVER", "RETRY", "HARD_BLOCK", or "HUMAN_REVIEW"
        aggregate_confidence: Weighted aggregate judge confidence 0.0–1.0
        query_id:            UUID linking all 4 GRPO SQLite tables
        rollout_id:          UUID for GRPO multi-rollout comparison
        correction_signal:   Human-readable correction hint (RETRY path only)
        claims_count:        Total claims extracted from LLM answer
        material_claims:     Number of claims marked is_material
        raw:                 Full MADOutput dict
    """
    routing_decision: str
    aggregate_confidence: float
    query_id: str = ""
    rollout_id: str = ""
    correction_signal: str | None = None
    claims_count: int = 0
    material_claims: int = 0
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_mad_output(cls, out: Any) -> "MADResult":
        """Build from a MADOutput Pydantic model instance."""
        claims = getattr(out, "claims", [])
        return cls(
            routing_decision=out.routing_decision,
            aggregate_confidence=out.aggregate_confidence,
            query_id=out.query_id,
            rollout_id=out.rollout_id,
            correction_signal=out.correction_signal,
            claims_count=len(claims),
            material_claims=sum(1 for c in claims if c.is_material),
            raw=out.model_dump(mode="json"),
        )


@dataclass
class RAGResult:
    """
    Result of the RAG retrieval + SLM verification step.

    Attributes:
        query:            Original query or claim text sent to the RAG pipeline
        sufficient_context: True if the SLM verifier found enough evidence
        confidence:       Verifier confidence score 0.0–1.0
        grounded_summary: Human-readable summary grounded in retrieved evidence
        top_chunks:       Verified top chunks selected by the SLM verifier
        rejected_chunks:  Chunks the verifier considered and rejected
        retrieved_count:  Total candidates fetched from Qdrant before filtering
        raw:              Full pipeline result dict (for debugging)
    """
    query: str
    sufficient_context: bool
    confidence: float
    grounded_summary: str
    top_chunks: list[dict] = field(default_factory=list)
    rejected_chunks: list[dict] = field(default_factory=list)
    retrieved_count: int = 0
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_pipeline(cls, result: dict) -> "RAGResult":
        """Build from the dict returned by rag.pipeline.run_rag_pipeline()."""
        verified = result.get("verified_output", {})
        return cls(
            query=result.get("query", ""),
            sufficient_context=verified.get("sufficient_context", False),
            confidence=float(verified.get("confidence") or 0.0),
            grounded_summary=verified.get("grounded_summary", ""),
            top_chunks=verified.get("top_chunks", []),
            rejected_chunks=verified.get("rejected_chunks", []),
            retrieved_count=result.get("retrieved_count", 0),
            raw=result,
        )


@dataclass
class PipelineResult:
    """
    Full result of one end-to-end pipeline run.

    Attributes:
        session_id:     UUID for the session (stored in SQLite)
        query:          Original user query
        gateway:        GatewayResult from the input check
        llm_answer:     LLM response text (None if request was blocked)
        mad:            MADResult (None if MAD is disabled or still pending)
        rag:            RAGResult if retrieve() was called standalone; None otherwise
        duration_ms:    Total wall-clock time for gateway + LLM (MAD is async)
    """
    session_id: str
    query: str
    gateway: GatewayResult
    llm_answer: str | None
    mad: MADResult | None
    duration_ms: int
    rag: Optional["RAGResult"] = None
