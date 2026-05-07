"""
confidence/tests/test_cse.py — Pytest unit tests for the Confidence Scoring Engine.

Run from repo root:
    pytest confidence/tests/test_cse.py -v

All tests are pure-Python and require no external services (Ollama, DeepEval,
Anthropic API).  DeepEval is never imported here — ConfidenceScorer._run_deepeval
is monkeypatched where needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import patch

import pytest

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_repo = Path(__file__).resolve().parent.parent.parent
for _p in [str(_repo), str(_repo / "multi_agent_debate"), str(_repo / "rag_folder")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multi_agent.models import Claim, JudgeVerdict, Verdict

from confidence.cse_config import CSEScoringConfig
from confidence.cse_types import RoutingDecision, ScoringMode
from confidence.scorer import (
    ConfidenceScorer,
    _aggregate_judge,
    _check_hard_block,
    _claim_quality,
    _context_quality,
    _normalise_weights,
    _resolve_scoring_mode,
    _route,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture()
def cfg() -> CSEScoringConfig:
    return CSEScoringConfig()


@pytest.fixture()
def scorer(cfg) -> ConfidenceScorer:
    return ConfidenceScorer(config=cfg)


def _deepeval_patch(f_llm, h_llm, relevancy, err=""):
    """Return a monkeypatch function that simulates _run_deepeval output."""
    def _mock(self, *args, **kwargs):
        return f_llm, h_llm, relevancy, err
    return _mock


def _make_claim(cid: int, text: str, material: bool, verdict=Verdict.SUPPORTED) -> Claim:
    return Claim(claim_id=cid, claim_text=text, is_material=material, verdict=verdict)


def _make_verdict(cid: int, text: str, material: bool, score: float) -> JudgeVerdict:
    return JudgeVerdict(
        claim_id=cid, claim_text=text, is_material=material,
        score=score, reasoning="test"
    )


# ── Helper: score with a fixed deepeval result ────────────────────────────────

def _score_with_deepeval(
    scorer: ConfidenceScorer,
    claims: List[Claim],
    verdicts: List[JudgeVerdict],
    chunks: Optional[List[str]] = None,
    f_llm: float = 0.9,
    h_llm: float = 0.1,
    relevancy: float = 0.85,
    deepeval_err: str = "",
    retry_count: int = 0,
):
    with patch.object(type(scorer), "_run_deepeval", _deepeval_patch(f_llm, h_llm, relevancy, deepeval_err)):
        return scorer.score(
            query="test query",
            llm_answer="test answer",
            rag_chunks=chunks if chunks is not None else ["context chunk one", "context chunk two"],
            final_claims=claims,
            judge_verdicts=verdicts,
            retry_count=retry_count,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 1. ROUTING: DELIVER
# ══════════════════════════════════════════════════════════════════════════════

class TestDeliver:
    def test_high_confidence_delivers(self, scorer):
        claims   = [_make_claim(1, "HIPAA requires PHI safeguards", True)]
        verdicts = [_make_verdict(1, "HIPAA requires PHI safeguards", True, 1.0)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.95, h_llm=0.05, relevancy=0.90)

        assert result.routing_decision == RoutingDecision.DELIVER.value
        assert result.final_score >= scorer.cfg.deliver_threshold

    def test_deliver_score_clamped_to_1(self, scorer):
        # Perfect signals should not exceed 1.0
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 1.0)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=1.0, h_llm=0.0, relevancy=1.0)

        assert result.final_score <= 1.0

    def test_explanation_mentions_deliver(self, scorer):
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 1.0)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.95, h_llm=0.05, relevancy=0.90)

        assert "DELIVER" in result.explanation


# ══════════════════════════════════════════════════════════════════════════════
# 2. ROUTING: RETRY
# ══════════════════════════════════════════════════════════════════════════════

class TestRetry:
    def test_medium_confidence_retries(self, scorer):
        claims   = [_make_claim(1, "partial claim", False)]
        verdicts = [_make_verdict(1, "partial claim", False, 0.5)]

        # Moderate signals → score in [0.55, 0.78)
        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.6, h_llm=0.4, relevancy=0.5)

        assert result.routing_decision == RoutingDecision.RETRY.value
        assert scorer.cfg.human_review_threshold <= result.final_score < scorer.cfg.deliver_threshold

    def test_retry_reasons_populated(self, scorer):
        claims   = [_make_claim(1, "claim", False)]
        verdicts = [_make_verdict(1, "claim", False, 0.5)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.6, h_llm=0.4, relevancy=0.5)

        assert result.routing_decision == RoutingDecision.RETRY.value
        assert len(result.retry_reasons) > 0

    def test_explanation_mentions_retry(self, scorer):
        claims   = [_make_claim(1, "claim", False)]
        verdicts = [_make_verdict(1, "claim", False, 0.5)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.6, h_llm=0.4, relevancy=0.5)

        if result.routing_decision == RoutingDecision.RETRY.value:
            assert "RETRY" in result.explanation


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROUTING: HUMAN_REVIEW
# ══════════════════════════════════════════════════════════════════════════════

class TestHumanReview:
    def test_low_confidence_goes_to_review(self, scorer):
        claims   = [_make_claim(1, "vague claim", False)]
        verdicts = [_make_verdict(1, "vague claim", False, 0.3)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.2, h_llm=0.7, relevancy=0.2)

        assert result.routing_decision == RoutingDecision.HUMAN_REVIEW.value
        assert result.final_score < scorer.cfg.human_review_threshold

    def test_review_reasons_populated(self, scorer):
        claims   = [_make_claim(1, "vague claim", False)]
        verdicts = [_make_verdict(1, "vague claim", False, 0.3)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.2, h_llm=0.7, relevancy=0.2)

        assert result.routing_decision == RoutingDecision.HUMAN_REVIEW.value
        assert len(result.review_reasons) > 0

    def test_retry_limit_escalates_to_review(self, scorer):
        """When retry_count >= max_retry_count, a RETRY candidate becomes HUMAN_REVIEW."""
        claims   = [_make_claim(1, "claim", False)]
        verdicts = [_make_verdict(1, "claim", False, 0.6)]

        # Scores that would normally route to RETRY
        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.65, h_llm=0.35, relevancy=0.6,
                                      retry_count=scorer.cfg.max_retry_count)

        assert result.routing_decision == RoutingDecision.HUMAN_REVIEW.value
        assert any("Retry limit" in r for r in result.review_reasons)

    def test_explanation_mentions_human_review(self, scorer):
        claims   = [_make_claim(1, "vague claim", False)]
        verdicts = [_make_verdict(1, "vague claim", False, 0.3)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.2, h_llm=0.7, relevancy=0.2)

        if result.routing_decision == RoutingDecision.HUMAN_REVIEW.value:
            assert "HUMAN_REVIEW" in result.explanation


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROUTING: HARD_BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class TestHardBlock:
    def test_false_material_claim_hard_blocks(self, scorer):
        claims   = [_make_claim(1, "GDPR right to erasure is absolute", True)]
        verdicts = [_make_verdict(1, "GDPR right to erasure is absolute", True, 0.0)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.9, h_llm=0.1, relevancy=0.9)

        assert result.routing_decision == RoutingDecision.HARD_BLOCK.value
        assert result.hard_blocked is True

    def test_hard_block_overrides_high_aggregate_score(self, scorer):
        """HARD_BLOCK must fire even when the aggregate score would otherwise DELIVER."""
        claims = [
            _make_claim(1, "correct claim", True),
            _make_claim(2, "false material claim", True),
        ]
        verdicts = [
            _make_verdict(1, "correct claim", True, 1.0),
            _make_verdict(2, "false material claim", True, 0.0),
        ]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=1.0, h_llm=0.0, relevancy=1.0)

        assert result.routing_decision == RoutingDecision.HARD_BLOCK.value

    def test_block_reasons_populated(self, scorer):
        claims   = [_make_claim(1, "false claim", True)]
        verdicts = [_make_verdict(1, "false claim", True, 0.0)]

        result = _score_with_deepeval(scorer, claims, verdicts)

        assert result.routing_decision == RoutingDecision.HARD_BLOCK.value
        assert len(result.block_reasons) > 0

    def test_non_material_zero_score_does_not_hard_block(self, scorer):
        """A non-material claim with score=0.0 should not trigger HARD_BLOCK."""
        claims   = [_make_claim(1, "minor claim", False)]
        verdicts = [_make_verdict(1, "minor claim", False, 0.0)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.9, h_llm=0.1, relevancy=0.9)

        assert result.routing_decision != RoutingDecision.HARD_BLOCK.value

    def test_top_failed_claims_includes_blocked_claim(self, scorer):
        claims   = [_make_claim(1, "false material claim text", True)]
        verdicts = [_make_verdict(1, "false material claim text", True, 0.0)]

        result = _score_with_deepeval(scorer, claims, verdicts)

        assert result.routing_decision == RoutingDecision.HARD_BLOCK.value
        assert len(result.top_failed_claims) > 0
        assert result.top_failed_claims[0].claim_id == 1


# ══════════════════════════════════════════════════════════════════════════════
# 5. FALLBACK MODES
# ══════════════════════════════════════════════════════════════════════════════

class TestFallbackModes:
    def test_deepeval_unavailable_triggers_judge_only_fallback(self, scorer):
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 0.9)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      deepeval_err="ConnectionError: Ollama offline",
                                      f_llm=None, h_llm=None, relevancy=None)

        assert result.scoring_mode == ScoringMode.JUDGE_ONLY_FALLBACK.value
        assert result.triggered_flags.deepeval_unavailable is True

    def test_missing_context_skips_deepeval_flags_correctly(self, scorer):
        """No context → DeepEval must not run and missing_context flag is set."""
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 1.0)]

        with patch.object(type(scorer), "_run_deepeval",
                          _deepeval_patch(None, None, None, "no_context")):
            result = scorer.score(
                query="q", llm_answer="a",
                rag_chunks=[],  # no context
                final_claims=claims,
                judge_verdicts=verdicts,
            )

        assert result.triggered_flags.missing_context is True
        assert result.score_breakdown.faithfulness_score is None
        assert result.score_breakdown.hallucination_score is None

    def test_missing_context_is_not_silently_zero(self, scorer):
        """Missing DeepEval signals must appear as None in breakdown, not 0."""
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 1.0)]

        with patch.object(type(scorer), "_run_deepeval",
                          _deepeval_patch(None, None, None, "no_context")):
            result = scorer.score(
                query="q", llm_answer="a",
                rag_chunks=[],
                final_claims=claims,
                judge_verdicts=verdicts,
            )

        assert result.score_breakdown.faithfulness_score is None
        assert result.score_breakdown.contextual_relevancy_score is None

    def test_error_fallback_forces_human_review(self, scorer):
        """When both DeepEval and judge are unavailable, route to HUMAN_REVIEW."""
        result = _score_with_deepeval(
            scorer, [], [],   # no claims, no verdicts
            deepeval_err="both_unavailable",
            f_llm=None, h_llm=None, relevancy=None,
        )

        assert result.scoring_mode == ScoringMode.ERROR_FALLBACK.value
        assert result.routing_decision == RoutingDecision.HUMAN_REVIEW.value

    def test_judge_only_fallback_does_not_deliver_on_weak_judge(self, scorer):
        """In JUDGE_ONLY_FALLBACK with a weak judge score, should not DELIVER."""
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 0.5)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=None, h_llm=None, relevancy=None,
                                      deepeval_err="Ollama timeout")

        assert result.routing_decision != RoutingDecision.DELIVER.value


# ══════════════════════════════════════════════════════════════════════════════
# 6. WEIGHT NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

class TestWeightNormalisation:
    def test_normalised_weights_sum_to_one(self, cfg):
        weights = {"a": 0.25, "b": 0.20, "c": 0.15, "d": 0.35, "e": 0.05}
        normed  = _normalise_weights(weights)
        assert abs(sum(normed.values()) - 1.0) < 1e-9

    def test_missing_signal_weights_renormed(self, cfg):
        """Drop a signal and re-normalise — remaining weights still sum to 1."""
        full = {"faithfulness": 0.25, "halluc_inverse": 0.20,
                "contextual_rel": 0.15, "judge_eval": 0.35, "ctx_quality": 0.05}
        # Drop DeepEval signals (faithfulness, halluc, relevancy, ctx_quality)
        judge_only = {"judge_eval": full["judge_eval"]}
        normed = _normalise_weights(judge_only)
        assert abs(sum(normed.values()) - 1.0) < 1e-9
        assert normed["judge_eval"] == pytest.approx(1.0)

    def test_empty_weights_returns_empty_dict(self):
        assert _normalise_weights({}) == {}

    def test_unavailable_signals_not_zero(self, scorer):
        """Score breakdown must show None for missing signals, not 0.0."""
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 0.9)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=None, h_llm=None, relevancy=None,
                                      deepeval_err="timeout")

        bd = result.score_breakdown
        assert bd.faithfulness_score is None
        assert bd.hallucination_score is None
        assert bd.hallucination_risk_inverse is None
        assert bd.contextual_relevancy_score is None


# ══════════════════════════════════════════════════════════════════════════════
# 7. SCORE CLAMPING
# ══════════════════════════════════════════════════════════════════════════════

class TestScoreClamping:
    def test_score_always_between_0_and_1(self, scorer):
        for score in [0.0, 0.3, 0.5, 0.78, 0.9, 1.0]:
            claims   = [_make_claim(1, "c", False)]
            verdicts = [_make_verdict(1, "c", False, score)]
            result = _score_with_deepeval(scorer, claims, verdicts,
                                          f_llm=score, h_llm=1.0 - score, relevancy=score)
            assert 0.0 <= result.final_score <= 1.0, f"score out of range for input={score}"

    def test_zero_score_clamped_not_negative(self, scorer):
        claims   = [_make_claim(1, "bad claim", False)]
        verdicts = [_make_verdict(1, "bad claim", False, 0.0)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.0, h_llm=1.0, relevancy=0.0,
                                      chunks=["one chunk"])

        assert result.final_score >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLAIM SEVERITY AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

class TestClaimSeverity:
    def test_critical_unsupported_claim_blocks_despite_high_score(self, scorer):
        """A critical/material false claim must block even when other scores are high."""
        claims = [
            _make_claim(1, "correct minor claim", False),
            _make_claim(2, "false material claim", True),
        ]
        verdicts = [
            _make_verdict(1, "correct minor claim", False, 1.0),
            _make_verdict(2, "false material claim", True, 0.0),
        ]
        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.95, h_llm=0.05, relevancy=0.95)

        assert result.routing_decision == RoutingDecision.HARD_BLOCK.value

    def test_claim_quality_lower_for_unsupported_claims(self, cfg):
        claims = [
            _make_claim(1, "material claim", True),
            _make_claim(2, "minor claim", False),
        ]
        verdicts_good = [
            _make_verdict(1, "material claim", True, 1.0),
            _make_verdict(2, "minor claim", False, 1.0),
        ]
        verdicts_bad = [
            _make_verdict(1, "material claim", True, 0.0),
            _make_verdict(2, "minor claim", False, 0.0),
        ]

        score_good, _ = _claim_quality(claims, verdicts_good, cfg)
        score_bad,  _ = _claim_quality(claims, verdicts_bad,  cfg)

        assert score_good > score_bad

    def test_failed_claims_list_populated(self, scorer):
        claims   = [_make_claim(1, "wrong claim", True)]
        verdicts = [_make_verdict(1, "wrong claim", True, 0.0)]

        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.5, h_llm=0.5, relevancy=0.5)

        assert len(result.top_failed_claims) > 0
        fc = result.top_failed_claims[0]
        assert fc.claim_id == 1
        assert fc.severity in ("material", "critical")


# ══════════════════════════════════════════════════════════════════════════════
# 9. CONTEXT QUALITY
# ══════════════════════════════════════════════════════════════════════════════

class TestContextQuality:
    def test_no_context_sets_missing_context_flag(self, scorer):
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 1.0)]

        with patch.object(type(scorer), "_run_deepeval",
                          _deepeval_patch(None, None, None, "no_context")):
            result = scorer.score("q", "a", [], claims, verdicts)

        assert result.triggered_flags.missing_context is True

    def test_insufficient_chunks_incurs_penalty(self, cfg):
        # Only 1 chunk when min_chunks_required=2
        _, flags, penalty = _context_quality(["single chunk"], None, cfg)
        assert flags.get("insufficient_chunks") is True
        assert penalty > 0.0

    def test_low_relevancy_sets_flag(self, cfg):
        _, flags, penalty = _context_quality(
            ["chunk a", "chunk b"],
            0.1,   # well below low_relevancy_threshold=0.40
            cfg,
        )
        assert flags.get("low_relevancy") is True
        assert penalty > 0.0

    def test_good_context_no_penalty(self, cfg):
        _, flags, penalty = _context_quality(
            ["chunk a", "chunk b", "chunk c"],
            0.8,  # high relevancy
            cfg,
        )
        assert penalty == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 10. SCORING MODES
# ══════════════════════════════════════════════════════════════════════════════

class TestScoringModes:
    def test_full_mode_when_both_available(self):
        mode = _resolve_scoring_mode(deepeval_available=True, judge_available=True)
        assert mode == ScoringMode.FULL

    def test_judge_only_fallback_when_deepeval_missing(self):
        mode = _resolve_scoring_mode(deepeval_available=False, judge_available=True)
        assert mode == ScoringMode.JUDGE_ONLY_FALLBACK

    def test_context_only_fallback_when_judge_missing(self):
        mode = _resolve_scoring_mode(deepeval_available=True, judge_available=False)
        assert mode == ScoringMode.CONTEXT_ONLY_FALLBACK

    def test_error_fallback_when_both_missing(self):
        mode = _resolve_scoring_mode(deepeval_available=False, judge_available=False)
        assert mode == ScoringMode.ERROR_FALLBACK


# ══════════════════════════════════════════════════════════════════════════════
# 11. ROUTING LOGIC (pure function)
# ══════════════════════════════════════════════════════════════════════════════

class TestRouteFunction:
    def test_hard_block_returns_hard_block(self, cfg):
        from confidence.cse_types import TriggeredFlags
        flags = TriggeredFlags(material_claim_failed=True)
        routing, _, _, block = _route(0.9, True, flags, ScoringMode.FULL, 0, cfg)
        assert routing == RoutingDecision.HARD_BLOCK.value
        assert block

    def test_error_fallback_forces_review(self, cfg):
        from confidence.cse_types import TriggeredFlags
        flags = TriggeredFlags()
        routing, _, review, _ = _route(0.5, False, flags, ScoringMode.ERROR_FALLBACK, 0, cfg)
        assert routing == RoutingDecision.HUMAN_REVIEW.value
        assert review

    def test_retry_limit_forces_review(self, cfg):
        from confidence.cse_types import TriggeredFlags
        flags = TriggeredFlags()
        routing, _, review, _ = _route(0.65, False, flags, ScoringMode.FULL,
                                        cfg.max_retry_count, cfg)
        assert routing == RoutingDecision.HUMAN_REVIEW.value

    def test_below_human_threshold_routes_review(self, cfg):
        from confidence.cse_types import TriggeredFlags
        flags = TriggeredFlags()
        routing, _, _, _ = _route(cfg.human_review_threshold - 0.01, False,
                                   flags, ScoringMode.FULL, 0, cfg)
        assert routing == RoutingDecision.HUMAN_REVIEW.value

    def test_between_thresholds_routes_retry(self, cfg):
        from confidence.cse_types import TriggeredFlags
        flags = TriggeredFlags()
        mid   = (cfg.human_review_threshold + cfg.deliver_threshold) / 2.0
        routing, _, _, _ = _route(mid, False, flags, ScoringMode.FULL, 0, cfg)
        assert routing == RoutingDecision.RETRY.value

    def test_above_deliver_threshold_delivers(self, cfg):
        from confidence.cse_types import TriggeredFlags
        flags = TriggeredFlags()
        routing, _, _, _ = _route(cfg.deliver_threshold + 0.01, False,
                                   flags, ScoringMode.FULL, 0, cfg)
        assert routing == RoutingDecision.DELIVER.value


# ══════════════════════════════════════════════════════════════════════════════
# 12. OUTPUT CONTRACT
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputContract:
    def test_as_dict_contains_required_keys(self, scorer):
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 1.0)]

        result = _score_with_deepeval(scorer, claims, verdicts)
        d = result.as_dict()

        for key in (
            "final_score", "routing_decision", "scoring_mode",
            "score_breakdown", "triggered_flags", "top_failed_claims",
            "explanation", "retry_reasons", "review_reasons", "block_reasons",
            "metadata", "version", "hard_blocked", "error",
        ):
            assert key in d, f"Missing key: {key}"

    def test_score_breakdown_keys(self, scorer):
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 1.0)]

        result = _score_with_deepeval(scorer, claims, verdicts)
        bd = result.score_breakdown.as_dict()

        for key in (
            "faithfulness_score", "hallucination_score", "hallucination_risk_inverse",
            "contextual_relevancy_score", "judge_eval_score",
            "claim_quality_score", "context_quality_score",
            "penalties", "applied_weights",
        ):
            assert key in bd, f"Missing breakdown key: {key}"

    def test_triggered_flags_keys(self, scorer):
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 1.0)]

        result = _score_with_deepeval(scorer, claims, verdicts)
        tf = result.triggered_flags.as_dict()

        for key in (
            "material_claim_failed", "critical_claim_failed", "missing_context",
            "deepeval_unavailable", "low_context_relevancy", "low_faithfulness",
            "high_hallucination_risk", "judge_disagreement", "unsupported_claims_present",
        ):
            assert key in tf, f"Missing flag key: {key}"

    def test_legacy_components_still_present(self, scorer):
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 1.0)]

        result = _score_with_deepeval(scorer, claims, verdicts)

        assert result.components is not None
        assert hasattr(result.components, "f_llm")
        assert hasattr(result.components, "h_llm")
        assert hasattr(result.components, "relevancy")
        assert hasattr(result.components, "judge_eval")

    def test_metadata_timestamp_present(self, scorer):
        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 1.0)]

        result = _score_with_deepeval(scorer, claims, verdicts)

        assert result.metadata is not None
        assert result.metadata.timestamp != ""
        assert result.metadata.cse_version != ""

    def test_explanation_non_empty_for_all_routes(self, scorer):
        scenarios = [
            # (f_llm, h_llm, relevancy, material_score)
            (0.95, 0.05, 0.90, 1.0),   # DELIVER
            (0.60, 0.40, 0.50, 0.5),   # RETRY or HUMAN_REVIEW
            (0.10, 0.90, 0.10, 0.3),   # HUMAN_REVIEW
            (0.95, 0.05, 0.90, 0.0),   # HARD_BLOCK
        ]
        for f, h, r, js in scenarios:
            claims   = [_make_claim(1, "c", True)]
            verdicts = [_make_verdict(1, "c", True, js)]
            result = _score_with_deepeval(scorer, claims, verdicts, f_llm=f, h_llm=h, relevancy=r)
            assert result.explanation != "", f"Empty explanation for f={f},j={js}"


# ══════════════════════════════════════════════════════════════════════════════
# 13. CONFIGURABLE THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigurableThresholds:
    def test_custom_thresholds_respected(self):
        custom_cfg = CSEScoringConfig(
            deliver_threshold=0.90,
            human_review_threshold=0.60,
        )
        scorer = ConfidenceScorer(config=custom_cfg)

        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 0.8)]

        # With high deliver_threshold=0.90, a score ~0.80 should RETRY not DELIVER
        result = _score_with_deepeval(scorer, claims, verdicts,
                                      f_llm=0.80, h_llm=0.20, relevancy=0.80)

        assert result.routing_decision in (
            RoutingDecision.RETRY.value, RoutingDecision.HUMAN_REVIEW.value
        )

    def test_config_version_in_metadata(self):
        cfg = CSEScoringConfig(config_version="v2.0-test")
        scorer = ConfidenceScorer(config=cfg)

        claims   = [_make_claim(1, "claim", True)]
        verdicts = [_make_verdict(1, "claim", True, 1.0)]

        result = _score_with_deepeval(scorer, claims, verdicts)
        assert result.metadata.config_version == "v2.0-test"


# ══════════════════════════════════════════════════════════════════════════════
# 14. CLAIM SEVERITY FIELD
# ══════════════════════════════════════════════════════════════════════════════

class TestClaimSeverityModel:
    def test_claim_severity_field_optional(self):
        """Claim without severity defaults to None — no breaking change."""
        c = Claim(claim_id=1, claim_text="test", is_material=True)
        assert c.severity is None

    def test_claim_severity_field_accepts_string(self):
        """Claim accepts severity string values."""
        c = Claim(claim_id=1, claim_text="test", is_material=True, severity="critical")
        assert c.severity == "critical"

    def test_claim_severity_material_string(self):
        c = Claim(claim_id=1, claim_text="test", is_material=False, severity="material")
        assert c.severity == "material"


# ══════════════════════════════════════════════════════════════════════════════
# 16. SEVERITY FIELD IN SCORER
# ══════════════════════════════════════════════════════════════════════════════

class TestClaimSeverityField:
    """Scorer must honour Claim.severity when computing hard-block and weights."""

    def _make_scorer(self):
        scorer = ConfidenceScorer()
        # Patch DeepEval to always return None (JUDGE_ONLY_FALLBACK mode)
        scorer._run_deepeval = lambda q, a, c: (None, None, None, "test_no_deepeval")
        return scorer

    def _make_claim(self, claim_id: int, is_material: bool, severity: Optional[str]) -> Claim:
        return Claim(
            claim_id=claim_id,
            claim_text=f"claim {claim_id}",
            is_material=is_material,
            severity=severity,
        )

    def _make_verdict(self, claim_id: int, score: float, is_material: bool) -> JudgeVerdict:
        return JudgeVerdict(
            claim_id=claim_id,
            claim_text=f"claim {claim_id}",
            is_material=is_material,
            score=score,
            reasoning="test",
        )

    def test_critical_severity_triggers_hard_block(self):
        """severity='critical', is_material=False, score=0.0 → HARD_BLOCK + critical_claim_failed."""
        scorer = self._make_scorer()
        claims = [self._make_claim(1, is_material=False, severity="critical")]
        verdicts = [self._make_verdict(1, score=0.0, is_material=False)]

        result = scorer.score("q", "a", ["ctx"], claims, verdicts)

        assert result.routing_decision == "HARD_BLOCK"
        assert result.triggered_flags.critical_claim_failed is True
        assert result.triggered_flags.material_claim_failed is True

    def test_severity_none_falls_back_to_is_material(self):
        """severity=None, is_material=True, score=1.0 → DELIVER (no regression)."""
        scorer = self._make_scorer()
        claims = [self._make_claim(1, is_material=True, severity=None)]
        verdicts = [self._make_verdict(1, score=1.0, is_material=True)]

        result = scorer.score("q", "a", ["ctx"], claims, verdicts)

        assert result.routing_decision == "DELIVER"
        assert result.triggered_flags.material_claim_failed is False
        assert result.triggered_flags.critical_claim_failed is False

    def test_minor_severity_uses_minor_weight(self):
        """severity='minor', is_material=True → uses severity_minor weight (not severity_material)."""
        from confidence.cse_config import CSEScoringConfig
        from confidence.scorer import _claim_quality

        cfg = CSEScoringConfig()
        # severity="minor" should use cfg.severity_minor (0.2), not cfg.severity_material (0.8)
        claims = [Claim(claim_id=1, claim_text="c1", is_material=True, severity="minor")]
        verdicts = [JudgeVerdict(claim_id=1, claim_text="c1", is_material=True, score=1.0, reasoning="")]

        score, failed = _claim_quality(claims, verdicts, cfg)

        # With severity_minor weight, the single claim contributes severity_minor / severity_minor = 1.0
        assert score == 1.0
        # If it were severity_material, weight would still produce 1.0 for a fully supported claim
        # The real distinction: add a second claim with is_material=True, severity=None to compare weights
        claims2 = [
            Claim(claim_id=1, claim_text="c1", is_material=True, severity="minor"),
            Claim(claim_id=2, claim_text="c2", is_material=True, severity=None),
        ]
        verdicts2 = [
            JudgeVerdict(claim_id=1, claim_text="c1", is_material=True, score=0.0, reasoning=""),
            JudgeVerdict(claim_id=2, claim_text="c2", is_material=True, score=1.0, reasoning=""),
        ]
        score2, _ = _claim_quality(claims2, verdicts2, cfg)
        # claim1 (minor, score=0) contributes 0, weight=0.2
        # claim2 (material, score=1) contributes 0.8, weight=0.8
        # result = 0.8 / (0.2 + 0.8) = 0.8
        assert abs(score2 - 0.8) < 1e-4


# ══════════════════════════════════════════════════════════════════════════════
# 15. _print_cse_result — crash safety
# ══════════════════════════════════════════════════════════════════════════════

class TestPrintCseResult:
    """_print_cse_result must not crash when components is None."""

    def _make_result_no_components(self, routing="DELIVER", score=0.85):
        """Build a minimal CSEResult-like object with components=None."""
        from confidence.cse_types import CSEResult, TriggeredFlags, CSEMetadata
        import datetime

        flags = TriggeredFlags()
        meta  = CSEMetadata(
            timestamp=datetime.datetime.utcnow().isoformat(),
            cse_version="v2.0",
            config_version="test",
            formula_version="v2.0",
            scoring_mode=ScoringMode.JUDGE_ONLY_FALLBACK.value,
        )
        return CSEResult(
            final_score=score,
            routing_decision=routing,
            scoring_mode=ScoringMode.JUDGE_ONLY_FALLBACK.value,
            score_breakdown=None,
            triggered_flags=flags,
            top_failed_claims=[],
            explanation=f"Routed to {routing}.",
            metadata=meta,
            components=None,  # the case we're testing
            version="v2.0",
            error="",
        )

    def _get_print_fn(self):
        """Load _print_cse_result from the canonical pipeline file directly."""
        import importlib.util
        pipeline_path = (
            Path(__file__).resolve().parent.parent.parent
            / "multi_agent_debate" / "multi_agent" / "mad_pipeline.py"
        )
        spec   = importlib.util.spec_from_file_location("_canon_pipeline", pipeline_path)
        module = importlib.util.module_from_spec(spec)
        # Stub out heavy imports so the module body loads without Ollama/storage
        import sys, types
        fake_storage = types.ModuleType("multi_agent.storage")
        fake_storage.init_db = lambda: None
        fake_storage.write_query = lambda **kw: None
        fake_storage.write_judge_verdicts = lambda **kw: None
        fake_storage.update_query_cse = lambda **kw: None
        sys.modules.setdefault("multi_agent.storage", fake_storage)
        spec.loader.exec_module(module)
        return module._print_cse_result

    def test_print_cse_result_no_crash_with_components_none(self, capsys):
        """_print_cse_result must not raise when result.components is None."""
        _print_cse_result = self._get_print_fn()

        result = self._make_result_no_components(routing="DELIVER")
        _print_cse_result(result)

        captured = capsys.readouterr()
        assert "DELIVER" in captured.out
        assert "unavailable" in captured.out.lower() or "Final score" in captured.out

    def test_print_cse_result_uses_score_breakdown_when_present(self, capsys):
        """_print_cse_result uses score_breakdown when available."""
        from confidence.cse_types import ScoreBreakdown
        _print_cse_result = self._get_print_fn()

        result = self._make_result_no_components(routing="DELIVER")
        result.score_breakdown = ScoreBreakdown(
            faithfulness_score=0.9,
            hallucination_score=0.1,
            hallucination_risk_inverse=0.9,
            contextual_relevancy_score=0.75,
            judge_eval_score=0.88,
            claim_quality_score=None,
            context_quality_score=None,
        )
        _print_cse_result(result)

        captured = capsys.readouterr()
        # score_breakdown path should print something about the result
        assert "DELIVER" in captured.out or "Final score" in captured.out
