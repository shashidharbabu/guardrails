"""
guardrails_enterprise/types.py — Public types returned by GuardrailsClient.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComponentScores:
    """Per-component scores from the Confidence Scoring Engine."""
    f_llm:      float   # DeepEval FaithfulnessMetric  (0–1, higher = more faithful)
    h_llm:      float   # DeepEval HallucinationMetric (0–1, lower = less hallucination)
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

    formula: 0.30*f_llm + 0.25*(1-h_llm) + 0.10*relevancy + 0.35*judge_eval
    """
    final_score:      float
    routing_decision: str          # DELIVER / RETRY / HARD_BLOCK / HUMAN_REVIEW
    components:       ComponentScores
    version:          str  = "v2.0"
    hard_blocked:     bool = False
    error:            str  = ""

    @classmethod
    def from_dict(cls, d: dict) -> "CSEResult":
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


@dataclass
class SessionResult:
    """Summary of one pipeline session (gateway + LLM + MAD + CSE)."""
    session_id:       str
    query:            str
    gateway_decision: str
    gateway_score:    float
    llm_answer:       Optional[str]
    mad_routing:      Optional[str]
    mad_confidence:   Optional[float]
    cse:              Optional[CSEResult]
    created_at:       str
    pipeline_duration_ms: int
    raw:              Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionResult":
        # Parse CSE from mad_output_json if present
        cse = None
        mad_output_json = d.get("mad_output_json")
        if mad_output_json:
            import json
            try:
                mad_output = json.loads(mad_output_json) if isinstance(mad_output_json, str) else mad_output_json
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
