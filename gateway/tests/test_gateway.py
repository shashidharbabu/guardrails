"""
Basic unit tests for gateway components.
Run: pytest gateway/tests/test_gateway.py -v
"""

from gateway.decision_engine import Decision, DecisionEngine


class TestDecisionEngine:
    """Test the decision engine logic in isolation (no models needed)."""

    def setup_method(self):
        self.engine = DecisionEngine(
            pass_threshold=0.3,
            block_threshold=0.7,
        )

    def test_clean_input_passes(self):
        result = self.engine.decide("", 0.0, 0.0, 0.0)
        assert result.decision == Decision.PASS
        assert result.gateway_score == 0.0
        assert result.is_allowed is True

    def test_high_jailbreak_hard_blocks(self):
        # jb_score=0.95 >= jb_override_threshold=0.7 → immediate BLOCK
        result = self.engine.decide("", pii_score=0.0, jb_score=0.95, pi_score=0.0)
        assert result.decision == Decision.BLOCK
        assert result.is_allowed is False
        assert result.blocked_reason is not None

    def test_high_pii_hard_blocks(self):
        # pii_score=0.9 >= pii_override_threshold=0.9 → immediate BLOCK
        result = self.engine.decide("", pii_score=0.9, jb_score=0.1, pi_score=0.1)
        assert result.decision == Decision.BLOCK
        assert result.is_allowed is False

    def test_combined_high_scores_block(self):
        result = self.engine.decide("", pii_score=0.8, jb_score=0.9, pi_score=0.8)
        assert result.decision == Decision.BLOCK
        assert result.gateway_score > 0.7

    def test_borderline_escalates(self):
        result = self.engine.decide("", pii_score=0.3, jb_score=0.5, pi_score=0.3)
        assert result.decision == Decision.ESCALATE

    def test_score_clamped_to_1(self):
        result = self.engine.decide("", pii_score=1.0, jb_score=1.0, pi_score=1.0)
        assert result.gateway_score == 1.0

    def test_non_pass_reason_populated(self):
        result = self.engine.decide("test", pii_score=0.0, jb_score=0.95, pi_score=0.0)
        assert result.blocked_reason is not None
        assert (
            "FLAGGED FOR REVIEW" in result.blocked_reason.upper()
            or "BLOCKED" in result.blocked_reason.upper()
        )
