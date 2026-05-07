"""
confidence/cse_config.py — CSE scoring policy configuration.

All weights, thresholds, penalties, and version strings live here.
Override at runtime via environment variables or by passing a custom
CSEScoringConfig instance to ConfidenceScorer.

WEIGHT RATIONALE
----------------
Defaults follow the spec's balanced baseline (FULL mode).  In
JUDGE_ONLY_FALLBACK the scorer drops DeepEval signals automatically
and re-normalises the remaining weights — no manual override needed.

Ablation study (2026-05-06) found judge_heavy (w_judge=0.60) gives
75% routing accuracy with TF-IDF / noisy DeepEval.  Once real Qdrant
retrieval is available and DeepEval scores are meaningful, the
balanced defaults below should be re-validated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class CSEScoringConfig:
    """
    Scoring policy for the Confidence Scoring Engine.

    Weights
    -------
    Must sum to 1.0 when all five signals are available.
    If a signal is absent, the scorer drops that weight and
    re-normalises the remainder — missing signals are never
    silently treated as 0.

    Routing thresholds
    ------------------
    deliver_threshold      — DELIVER if final_score ≥ this value
    human_review_threshold — HUMAN_REVIEW if final_score < this value
    Between them           — RETRY (subject to retry_count limit)

    Penalties
    ---------
    Subtracted from the weighted score before clamping to [0, 1].
    Multiple penalties accumulate but the total is never allowed to
    drive the score below 0.
    """

    # ── Signal weights (FULL mode) ────────────────────────────────────────────
    w_faithfulness:         float = 0.25
    w_hallucination_inverse: float = 0.20
    w_contextual_relevancy: float = 0.15
    w_judge_eval:           float = 0.35
    w_context_quality:      float = 0.05

    # ── Routing thresholds ────────────────────────────────────────────────────
    deliver_threshold:       float = 0.78
    human_review_threshold:  float = 0.55
    max_retry_count:         int   = 2

    # ── Penalties ─────────────────────────────────────────────────────────────
    penalty_missing_context:       float = 0.10
    penalty_low_context_relevancy: float = 0.05
    penalty_insufficient_chunks:   float = 0.05
    penalty_weak_citation_coverage: float = 0.05
    penalty_judge_disagreement:    float = 0.05

    # ── Context quality thresholds ────────────────────────────────────────────
    min_chunks_required:            int   = 2
    low_relevancy_threshold:        float = 0.40
    judge_disagreement_std_threshold: float = 0.30

    # ── Claim severity multipliers ────────────────────────────────────────────
    # "critical" is reserved — current data model has only is_material (bool).
    # When the MAD pipeline adds a severity field, critical=1.0 activates.
    severity_critical: float = 1.0
    severity_material: float = 0.8   # is_material=True
    severity_minor:    float = 0.2   # is_material=False

    # ── Version identifiers ───────────────────────────────────────────────────
    cse_version:    str = "cse_v1.1_weighted_claim_aware"
    config_version: str = "v1.1"
    formula_version: str = "weighted_claim_aware_v1"

    # ── Environment override constructor ─────────────────────────────────────
    @classmethod
    def from_env(cls) -> "CSEScoringConfig":
        """
        Build a config from defaults, then apply any env-var overrides.
        Supported variables:
            CSE_DELIVER_THRESHOLD        float
            CSE_HUMAN_REVIEW_THRESHOLD   float
            CSE_MAX_RETRY_COUNT          int
        """
        cfg = cls()
        if v := os.getenv("CSE_DELIVER_THRESHOLD"):
            cfg.deliver_threshold = float(v)
        if v := os.getenv("CSE_HUMAN_REVIEW_THRESHOLD"):
            cfg.human_review_threshold = float(v)
        if v := os.getenv("CSE_MAX_RETRY_COUNT"):
            cfg.max_retry_count = int(v)
        return cfg


# Module-level singleton — used when no explicit config is passed.
DEFAULT_CONFIG: CSEScoringConfig = CSEScoringConfig()
