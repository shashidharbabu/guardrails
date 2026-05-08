"""Unit tests for Confidence Scoring Engine."""
from __future__ import annotations

import pytest

from confidence.cse_config import CSEScoringConfig
from confidence.cse_types import RoutingDecision, ScoringMode


class TestCSEConfig:
    def test_default_weights_sum_to_one(self):
        from confidence.cse_config import DEFAULT_CONFIG
        w = DEFAULT_CONFIG.weights
        total = w.faithfulness + w.hallucination_inverse + w.contextual_relevancy + w.judge_eval + w.context_quality
        assert pytest.approx(total, abs=1e-6) == 1.0

    def test_deliver_threshold_above_human_review(self):
        from confidence.cse_config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.deliver_threshold > DEFAULT_CONFIG.human_review_threshold

    def test_custom_config_overrides(self):
        cfg = CSEScoringConfig(deliver_threshold=0.90, human_review_threshold=0.60)
        assert cfg.deliver_threshold == 0.90
        assert cfg.human_review_threshold == 0.60


class TestRoutingDecisionEnum:
    def test_all_values_present(self):
        decisions = {d.value for d in RoutingDecision}
        assert decisions == {"DELIVER", "RETRY", "HUMAN_REVIEW", "HARD_BLOCK"}


class TestScoringModeEnum:
    def test_all_modes_present(self):
        modes = {m.value for m in ScoringMode}
        assert "FULL" in modes
        assert "ERROR_FALLBACK" in modes


class TestCSEScorer:
    """Integration-light tests using stub claims — no LLM calls."""

    def _make_claims(self, verdicts: list[float]):
        """Build minimal Claim + JudgeVerdict objects."""
        from confidence.cse_types import RoutingDecision
        try:
            from multi_agent.models import Claim, JudgeVerdict
        except ImportError:
            pytest.skip("multi_agent.models not importable in this environment")

        claims = []
        judge_verdicts = []
        for i, v in enumerate(verdicts):
            c = Claim(
                claim_id=str(i),
                claim_text=f"Claim {i}",
                is_material=True,
                confidence_p=0.8 if v >= 0.5 else 0.3,
            )
            claims.append(c)
            jv = JudgeVerdict(
                claim_id=str(i),
                v_label=v,
                reasoning="test",
            )
            judge_verdicts.append(jv)
        return claims, judge_verdicts

    def test_hard_block_on_false_material_claim(self):
        from confidence.scorer import ConfidenceScorer
        scorer = ConfidenceScorer()
        claims, jvs = self._make_claims([0.0])  # one false material claim
        result = scorer.score(
            query="test",
            llm_answer="test answer",
            rag_chunks=["chunk"],
            claims=claims,
            judge_verdicts=jvs,
        )
        assert result.routing_decision == RoutingDecision.HARD_BLOCK

    def test_deliver_on_all_supported_claims(self):
        from confidence.scorer import ConfidenceScorer
        scorer = ConfidenceScorer()
        claims, jvs = self._make_claims([1.0, 1.0, 1.0])
        result = scorer.score(
            query="What does HIPAA require?",
            llm_answer="HIPAA requires privacy safeguards.",
            rag_chunks=["HIPAA 45 CFR §164 requires safeguards."],
            claims=claims,
            judge_verdicts=jvs,
        )
        assert result.routing_decision in (RoutingDecision.DELIVER, RoutingDecision.RETRY)

    def test_result_score_in_range(self):
        from confidence.scorer import ConfidenceScorer
        scorer = ConfidenceScorer()
        claims, jvs = self._make_claims([0.5])
        result = scorer.score(
            query="q", llm_answer="a", rag_chunks=[], claims=claims, judge_verdicts=jvs
        )
        assert 0.0 <= result.final_score <= 1.0
