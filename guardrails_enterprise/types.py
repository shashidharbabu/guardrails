"""
guardrails_enterprise/types.py — All public types returned by the SDK.

Hierarchy
---------
PipelineResult          ← top-level, returned by GuardrailsPipeline.run()
  .gateway: GatewayResult    ← PII / jailbreak / prompt-injection scores
  .llm_answer: str | None    ← LLM response (None if gateway BLOCK)
  .mad: MADSummary | None    ← Multi-Agent Debate result
  .cse: CSEResult | None     ← Confidence Scoring Engine breakdown

CSEResult               ← confidence score + component breakdown
  .components: ComponentScores

SessionResult           ← returned by GuardrailsClient (HTTP client mode)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Gateway types ─────────────────────────────────────────────────────────────

@dataclass
class GatewayResult:
    """Result from the input guardrail (PII, jailbreak, prompt-injection)."""
    decision:       str          # PASS / ESCALATE / BLOCK
    gateway_score:  float
    pii_score:      float
    jb_score:       float
    pi_score:       float
    pii_entities:   List[Dict] = field(default_factory=list)
    threat_types:   List[str]  = field(default_factory=list)
    blocked_reason: Optional[str] = None

    @property
    def is_allowed(self) -> bool:
        return self.decision != "BLOCK"

    @classmethod
    def from_dict(cls, d: dict) -> "GatewayResult":
        scores = d.get("scores", {})
        return cls(
            decision       = d.get("decision", "UNKNOWN"),
            gateway_score  = float(d.get("gateway_score", 0.0)),
            pii_score      = float(scores.get("pii", d.get("pii_score", 0.0))),
            jb_score       = float(scores.get("jailbreak", d.get("jb_score", 0.0))),
            pi_score       = float(scores.get("prompt_injection", d.get("pi_score", 0.0))),
            pii_entities   = d.get("pii_entities", []),
            threat_types   = d.get("threat_types", []),
            blocked_reason = d.get("blocked_reason"),
        )

    @classmethod
    def from_gateway_result(cls, r: Any) -> "GatewayResult":
        """Convert from gateway.decision_engine.GatewayResult (library mode)."""
        d = r.to_dict() if hasattr(r, "to_dict") else {}
        return cls.from_dict(d)


# ── CSE types ─────────────────────────────────────────────────────────────────

@dataclass
class ComponentScores:
    """Per-component scores from the Confidence Scoring Engine."""
    f_llm:      float   # DeepEval FaithfulnessMetric  (0–1)
    h_llm:      float   # DeepEval HallucinationMetric (0–1, lower = better)
    relevancy:  float   # DeepEval ContextualRelevancy (0–1)
    judge_eval: float   # MAD judge aggregate           (0–1)

    @classmethod
    def from_dict(cls, d: dict) -> "ComponentScores":
        return cls(
            f_llm      = float(d.get("f_llm", 0.0)),
            h_llm      = float(d.get("h_llm", 0.0)),
            relevancy  = float(d.get("relevancy", 0.0)),
            judge_eval = float(d.get("judge_eval", 0.0)),
        )


@dataclass
class CSEResult:
    """
    Full Confidence Scoring Engine result.

    Formula (v2.0):
        0.30 * f_llm + 0.25 * (1 - h_llm) + 0.10 * relevancy + 0.35 * judge_eval

    Routing:
        HARD_BLOCK   — any is_material claim with judge score = 0.0
        DELIVER      — final_score >= 0.8
        RETRY        — 0.4 <= final_score < 0.8
        HUMAN_REVIEW — final_score < 0.4
    """
    final_score:      float
    routing_decision: str
    components:       ComponentScores
    version:          str  = "v2.0"   # "v2.0" = full formula, "v0.1" = judge-only fallback
    hard_blocked:     bool = False
    error:            str  = ""       # non-empty if DeepEval fell back to v0.1

    @classmethod
    def from_dict(cls, d: dict) -> "CSEResult":
        if d is None:
            return None
        return cls(
            final_score      = float(d.get("final_score", 0.0)),
            routing_decision = d.get("routing_decision", "UNKNOWN"),
            components       = ComponentScores.from_dict(d.get("components", {})),
            version          = d.get("version", "v0.1"),
            hard_blocked     = bool(d.get("hard_blocked", False)),
            error            = d.get("error", ""),
        )

    def __repr__(self) -> str:
        c = self.components
        return (
            f"CSEResult(score={self.final_score:.4f}, routing={self.routing_decision}, "
            f"version={self.version}, "
            f"F={c.f_llm:.3f}, H={c.h_llm:.3f}, R={c.relevancy:.3f}, J={c.judge_eval:.3f})"
        )


# ── MAD summary ───────────────────────────────────────────────────────────────

@dataclass
class MADSummary:
    """Condensed view of the Multi-Agent Debate result."""
    routing_decision:     str
    aggregate_confidence: float
    correction_signal:    Optional[str]
    claim_count:          int
    material_claim_count: int
    query_id:             str
    rollout_id:           str
    debate_transcript:    str = ""

    @classmethod
    def from_mad_output(cls, out: Any) -> "MADSummary":
        """Convert from multi_agent.models.MADOutput (library mode)."""
        claims = out.claims or []
        return cls(
            routing_decision     = out.routing_decision,
            aggregate_confidence = out.aggregate_confidence,
            correction_signal    = out.correction_signal,
            claim_count          = len(claims),
            material_claim_count = sum(1 for c in claims if c.is_material),
            query_id             = out.query_id,
            rollout_id           = out.rollout_id,
            debate_transcript    = out.debate_transcript or "",
        )

    @classmethod
    def from_dict(cls, d: dict) -> "MADSummary":
        claims = d.get("claims", [])
        return cls(
            routing_decision     = d.get("routing_decision", "UNKNOWN"),
            aggregate_confidence = float(d.get("aggregate_confidence", 0.0)),
            correction_signal    = d.get("correction_signal"),
            claim_count          = len(claims),
            material_claim_count = sum(1 for c in claims if c.get("is_material")),
            query_id             = d.get("query_id", ""),
            rollout_id           = d.get("rollout_id", ""),
            debate_transcript    = d.get("debate_transcript", ""),
        )


# ── Top-level pipeline result ─────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Full result of one end-to-end pipeline run.

    Returned by GuardrailsPipeline.run() and GuardrailsClient.run().

    Attributes
    ----------
    query       : The original user query.
    gateway     : Input guardrail result (PII, jailbreak, prompt injection).
    llm_answer  : The LLM's response. None if gateway blocked the request.
    mad         : Multi-Agent Debate summary. None if gateway blocked or LLM failed.
    cse         : Confidence Scoring Engine result. None if MAD did not run.
    routing     : Final routing decision (gateway decision if blocked, else MAD/CSE routing).
    blocked     : True if the request was blocked at the gateway.
    """
    query:      str
    gateway:    GatewayResult
    llm_answer: Optional[str]        = None
    mad:        Optional[MADSummary] = None
    cse:        Optional[CSEResult]  = None

    @property
    def routing(self) -> str:
        """Final routing decision across all pipeline stages."""
        if not self.gateway.is_allowed:
            return f"GATEWAY_{self.gateway.decision}"
        if self.cse:
            return self.cse.routing_decision
        if self.mad:
            return self.mad.routing_decision
        return "NO_MAD"

    @property
    def blocked(self) -> bool:
        return not self.gateway.is_allowed

    @property
    def confidence(self) -> Optional[float]:
        """Best available confidence score (CSE > MAD aggregate)."""
        if self.cse:
            return self.cse.final_score
        if self.mad:
            return self.mad.aggregate_confidence
        return None

    def __repr__(self) -> str:
        return (
            f"PipelineResult(routing={self.routing!r}, "
            f"confidence={self.confidence}, "
            f"blocked={self.blocked})"
        )


# ── Client-mode session ───────────────────────────────────────────────────────

@dataclass
class SessionResult:
    """Summary of one pipeline session from the App Backend (HTTP client mode)."""
    session_id:           str
    query:                str
    gateway_decision:     str
    gateway_score:        float
    llm_answer:           Optional[str]
    mad_routing:          Optional[str]
    mad_confidence:       Optional[float]
    cse:                  Optional[CSEResult]
    created_at:           str
    pipeline_duration_ms: int
    raw:                  Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionResult":
        cse = None
        mad_output_json = d.get("mad_output_json")
        if mad_output_json:
            try:
                mad_output = (
                    json.loads(mad_output_json)
                    if isinstance(mad_output_json, str)
                    else mad_output_json
                )
                cse_dict = mad_output.get("cse_result")
                if cse_dict:
                    cse = CSEResult.from_dict(cse_dict)
            except Exception:
                pass

        return cls(
            session_id           = d.get("id", ""),
            query                = d.get("query", ""),
            gateway_decision     = d.get("gateway_decision", ""),
            gateway_score        = float(d.get("gateway_score", 0.0)),
            llm_answer           = d.get("llm_answer"),
            mad_routing          = d.get("mad_routing"),
            mad_confidence       = d.get("mad_confidence"),
            cse                  = cse,
            created_at           = d.get("created_at", ""),
            pipeline_duration_ms = int(d.get("pipeline_duration_ms", 0)),
            raw                  = d,
        )
