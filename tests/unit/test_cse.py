"""Unit tests for Confidence Scoring Engine."""
from __future__ import annotations

import pytest

from confidence.cse_config import CSEScoringConfig, DEFAULT_CONFIG
from confidence.cse_types import RoutingDecision, ScoringMode


class TestCSEConfig:
    def test_default_weights_sum_to_one(self):
        cfg = DEFAULT_CONFIG
        total = (
            cfg.w_faithfulness
            + cfg.w_hallucination_inverse
            + cfg.w_contextual_relevancy
            + cfg.w_judge_eval
            + cfg.w_context_quality
        )
        assert pytest.approx(total, abs=1e-6) == 1.0

    def test_deliver_threshold_above_human_review(self):
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
    """Integration-light tests using stub claims — no LLM calls.

    The legacy `multi_agent.models.Claim` and `JudgeVerdict` are used here
    because `confidence/scorer.py` imports from that module.
    """

    def _make_claims_and_verdicts(self, v_labels: list):
        """Build minimal Claim + JudgeVerdict lists."""
        try:
            from multi_agent_debate.multi_agent.models import Claim, JudgeVerdict
        except ImportError:
            try:
                from multi_agent.models import Claim, JudgeVerdict
            except ImportError:
                pytest.skip("multi_agent.models not importable — skip CSE scorer tests")

        claims, judge_verdicts = [], []
        for i, v in enumerate(v_labels):
            confidence = 0.85 if v >= 0.5 else 0.25
            claims.append(
                Claim(
                    claim_id=i,
                    claim_text=f"Claim {i}: a verifiable statement.",
                    is_material=True,
                    confidence=confidence,
                    verdict=None,
                    reasoning="",
                )
            )
            judge_verdicts.append(
                JudgeVerdict(
                    claim_id=i,
                    claim_text=f"Claim {i}: a verifiable statement.",
                    is_material=True,
                    score=v,
                    reasoning="test judge",
                )
            )
        return claims, judge_verdicts

    def test_hard_block_on_false_material_claim(self):
        from confidence.scorer import ConfidenceScorer
        scorer = ConfidenceScorer()
        claims, jvs = self._make_claims_and_verdicts([0.0])
        result = scorer.score(
            query="test query",
            llm_answer="test answer",
            rag_chunks=["context chunk"],
            final_claims=claims,
            judge_verdicts=jvs,
        )
        assert result.routing_decision == RoutingDecision.HARD_BLOCK

    def test_deliver_on_all_supported_claims(self):
        from confidence.scorer import ConfidenceScorer
        scorer = ConfidenceScorer()
        claims, jvs = self._make_claims_and_verdicts([1.0, 1.0, 1.0])
        result = scorer.score(
            query="What are the technical safeguards required?",
            llm_answer="Technical safeguards are required under the Security Rule.",
            rag_chunks=["The Security Rule requires technical safeguards."],
            final_claims=claims,
            judge_verdicts=jvs,
        )
        assert result.routing_decision in (RoutingDecision.DELIVER, RoutingDecision.RETRY)

    def test_result_score_in_range(self):
        from confidence.scorer import ConfidenceScorer
        scorer = ConfidenceScorer()
        claims, jvs = self._make_claims_and_verdicts([0.5])
        result = scorer.score(
            query="q", llm_answer="a", rag_chunks=[], final_claims=claims, judge_verdicts=jvs
        )
        assert 0.0 <= result.final_score <= 1.0

    def test_result_has_routing_decision(self):
        from confidence.scorer import ConfidenceScorer
        scorer = ConfidenceScorer()
        claims, jvs = self._make_claims_and_verdicts([0.5])
        result = scorer.score(
            query="q", llm_answer="a", rag_chunks=[], final_claims=claims, judge_verdicts=jvs
        )
        assert result.routing_decision in list(RoutingDecision)
