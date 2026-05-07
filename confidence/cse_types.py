"""
confidence/cse_types.py — Enterprise CSE output contract.

All fields are backward-compatible: callers that only read
``final_score`` and ``routing_decision`` continue to work unchanged.
New fields (score_breakdown, triggered_flags, explanation, etc.) are
available for auditing, dashboards, and retry orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


# ── Enums ─────────────────────────────────────────────────────────────────────

class RoutingDecision(str, Enum):
    DELIVER      = "DELIVER"
    RETRY        = "RETRY"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    HARD_BLOCK   = "HARD_BLOCK"


class ScoringMode(str, Enum):
    FULL                  = "FULL"                   # DeepEval + MAD judge available
    JUDGE_ONLY_FALLBACK   = "JUDGE_ONLY_FALLBACK"    # DeepEval unavailable
    CONTEXT_ONLY_FALLBACK = "CONTEXT_ONLY_FALLBACK"  # MAD judge unavailable
    ERROR_FALLBACK        = "ERROR_FALLBACK"         # Both unavailable


# ── Sub-objects ───────────────────────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    """
    Per-signal scores and the weights actually applied for this run.

    Scores marked Optional are absent when the signal was unavailable —
    they are never silently substituted with 0.
    """
    faithfulness_score:           Optional[float]  # F_llm from DeepEval
    hallucination_score:          Optional[float]  # H_llm from DeepEval (raw, high = more hallucination)
    hallucination_risk_inverse:   Optional[float]  # 1 - H_llm (what enters the formula)
    contextual_relevancy_score:   Optional[float]  # relevancy from DeepEval
    judge_eval_score:             float            # MAD judge aggregate (always present or 0.5 neutral)
    claim_quality_score:          Optional[float]  # severity-weighted claim verdict aggregate
    context_quality_score:        Optional[float]  # lightweight context quality signal
    penalties:                    float = 0.0      # total penalty deducted
    applied_weights:              Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "faithfulness_score":         self.faithfulness_score,
            "hallucination_score":        self.hallucination_score,
            "hallucination_risk_inverse": self.hallucination_risk_inverse,
            "contextual_relevancy_score": self.contextual_relevancy_score,
            "judge_eval_score":           round(self.judge_eval_score, 4),
            "claim_quality_score":        self.claim_quality_score,
            "context_quality_score":      self.context_quality_score,
            "penalties":                  round(self.penalties, 4),
            "applied_weights":            {k: round(v, 4) for k, v in self.applied_weights.items()},
        }


@dataclass
class TriggeredFlags:
    """Boolean flags that inform routing and explanation generation."""
    material_claim_failed:     bool = False
    critical_claim_failed:     bool = False
    missing_context:           bool = False
    deepeval_unavailable:      bool = False
    low_context_relevancy:     bool = False
    low_faithfulness:          bool = False
    high_hallucination_risk:   bool = False
    judge_disagreement:        bool = False
    unsupported_claims_present: bool = False

    def any_hard_block_flag(self) -> bool:
        return self.material_claim_failed or self.critical_claim_failed

    def as_dict(self) -> dict:
        return {
            "material_claim_failed":      self.material_claim_failed,
            "critical_claim_failed":      self.critical_claim_failed,
            "missing_context":            self.missing_context,
            "deepeval_unavailable":       self.deepeval_unavailable,
            "low_context_relevancy":      self.low_context_relevancy,
            "low_faithfulness":           self.low_faithfulness,
            "high_hallucination_risk":    self.high_hallucination_risk,
            "judge_disagreement":         self.judge_disagreement,
            "unsupported_claims_present": self.unsupported_claims_present,
        }


@dataclass
class FailedClaim:
    """A claim that contributed to a negative routing decision."""
    claim_id:   int
    claim_text: str
    verdict:    str   # e.g. "contradicted", "unsupported", "partially_supported"
    severity:   str   # "critical", "material", "minor"
    score:      float

    def as_dict(self) -> dict:
        return {
            "claim_id":   self.claim_id,
            "claim_text": self.claim_text,
            "verdict":    self.verdict,
            "severity":   self.severity,
            "score":      round(self.score, 4),
        }


@dataclass
class CSEMetadata:
    """Audit metadata attached to every CSE result."""
    timestamp:          str
    cse_version:        str
    config_version:     str
    formula_version:    str
    scoring_mode:       str
    request_id:         Optional[str] = None
    model_name:         Optional[str] = None
    policy_domain:      Optional[str] = None
    source_chunk_ids:   List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "timestamp":       self.timestamp,
            "cse_version":     self.cse_version,
            "config_version":  self.config_version,
            "formula_version": self.formula_version,
            "scoring_mode":    self.scoring_mode,
            "request_id":      self.request_id,
            "model_name":      self.model_name,
            "policy_domain":   self.policy_domain,
            "source_chunk_ids": self.source_chunk_ids,
        }


# ── Legacy compat shim ────────────────────────────────────────────────────────

@dataclass
class ComponentScores:
    """
    Preserved for backward compatibility.

    New code should use ScoreBreakdown instead.  Existing callers that
    access result.components.f_llm / .h_llm / .relevancy / .judge_eval
    continue to work without change.
    """
    f_llm:      float
    h_llm:      float
    relevancy:  float
    judge_eval: float

    def as_dict(self) -> dict:
        return {
            "f_llm":      round(self.f_llm, 4),
            "h_llm":      round(self.h_llm, 4),
            "relevancy":  round(self.relevancy, 4),
            "judge_eval": round(self.judge_eval, 4),
        }


# ── Primary output type ───────────────────────────────────────────────────────

@dataclass
class CSEResult:
    """
    Full output of one CSE scoring run.

    Routing priority:
        HARD_BLOCK   → any material/critical claim is clearly false
        HUMAN_REVIEW → score < human_review_threshold, or error fallback
        RETRY        → human_review_threshold ≤ score < deliver_threshold
        DELIVER      → score ≥ deliver_threshold, no hard flags

    Scoring modes:
        FULL                  — DeepEval + MAD judge available
        JUDGE_ONLY_FALLBACK   — DeepEval failed; weights re-normalised to judge + claims
        CONTEXT_ONLY_FALLBACK — MAD judge unavailable; DeepEval signals only
        ERROR_FALLBACK        — Both unavailable; routes to HUMAN_REVIEW

    Versioning:
        cse_v1.1_weighted_claim_aware (see CSEScoringConfig for sub-versions)
    """

    # ── Core output (matches legacy contract) ─────────────────────────────────
    final_score:      float           # 0.0 – 1.0, always clamped
    routing_decision: str             # RoutingDecision value
    scoring_mode:     str             # ScoringMode value

    # ── Detailed breakdown ────────────────────────────────────────────────────
    score_breakdown:  ScoreBreakdown
    triggered_flags:  TriggeredFlags
    top_failed_claims: List[FailedClaim] = field(default_factory=list)

    # ── Human-readable explanations ───────────────────────────────────────────
    explanation:    str = ""
    retry_reasons:  List[str] = field(default_factory=list)
    review_reasons: List[str] = field(default_factory=list)
    block_reasons:  List[str] = field(default_factory=list)

    # ── Audit metadata ────────────────────────────────────────────────────────
    metadata: Optional[CSEMetadata] = None

    # ── Legacy compatibility fields ───────────────────────────────────────────
    components:   Optional[ComponentScores] = None   # kept for existing callers
    version:      str  = "cse_v1.1_weighted_claim_aware"
    hard_blocked: bool = False   # True when HARD_BLOCK was triggered
    error:        str  = ""      # non-empty if a fallback was used

    def as_dict(self) -> dict:
        return {
            "final_score":       round(self.final_score, 4),
            "routing_decision":  self.routing_decision,
            "scoring_mode":      self.scoring_mode,
            "score_breakdown":   self.score_breakdown.as_dict(),
            "triggered_flags":   self.triggered_flags.as_dict(),
            "top_failed_claims": [c.as_dict() for c in self.top_failed_claims],
            "explanation":       self.explanation,
            "retry_reasons":     self.retry_reasons,
            "review_reasons":    self.review_reasons,
            "block_reasons":     self.block_reasons,
            "metadata":          self.metadata.as_dict() if self.metadata else None,
            # legacy
            "version":           self.version,
            "hard_blocked":      self.hard_blocked,
            "error":             self.error,
            "components":        self.components.as_dict() if self.components else None,
        }
