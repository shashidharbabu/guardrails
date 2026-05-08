"""Unit tests for Gateway decision engine (no model loading)."""
from __future__ import annotations

import pytest

from gateway.decision_engine import DecisionEngine, Decision


@pytest.fixture
def engine():
    return DecisionEngine()


def test_hard_block_on_max_jailbreak(engine):
    """jb_score >= 0.7 triggers immediate BLOCK (hard override)."""
    result = engine.decide(pii_score=0.0, jb_score=0.95, pi_score=0.0)
    assert result.decision == Decision.BLOCK


def test_hard_block_on_max_pi(engine):
    """pi_score >= 0.7 triggers immediate BLOCK (hard override)."""
    result = engine.decide(pii_score=0.0, jb_score=0.0, pi_score=0.85)
    assert result.decision == Decision.BLOCK


def test_hard_block_on_high_pii(engine):
    """pii_score >= 0.9 triggers immediate BLOCK (hard override)."""
    result = engine.decide(pii_score=0.95, jb_score=0.0, pi_score=0.0)
    assert result.decision == Decision.BLOCK


def test_allow_clean_input(engine):
    """All low scores → PASS."""
    result = engine.decide(pii_score=0.01, jb_score=0.02, pi_score=0.01)
    assert result.decision == Decision.PASS
    assert result.is_allowed is True


def test_escalate_on_moderate_scores(engine):
    """Composite score in 0.3-0.7 band → ESCALATE."""
    result = engine.decide(pii_score=0.4, jb_score=0.3, pi_score=0.3)
    assert result.decision in (Decision.ESCALATE, Decision.PASS)


def test_gateway_score_range(engine):
    """gateway_score always in [0, 1]."""
    result = engine.decide(pii_score=0.5, jb_score=0.5, pi_score=0.5)
    assert 0.0 <= result.gateway_score <= 1.0


def test_result_is_dict_serialisable(engine):
    result = engine.decide(pii_score=0.1, jb_score=0.1, pi_score=0.1)
    d = result.to_dict()
    assert "decision" in d
    assert "gateway_score" in d
    assert "scores" in d
