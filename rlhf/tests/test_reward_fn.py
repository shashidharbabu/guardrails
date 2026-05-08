from __future__ import annotations

import math

import pytest

from rlhf.feedback_loop.heuristics import composite_auto_reward, compute_heuristics
from rlhf.feedback_loop.scorer import compute_attack_b_reward


def test_brier_components():
    p, v = 0.9, 1.0
    brier = 2 * p * v - p**2
    assert math.isclose(brier, 0.99, rel_tol=1e-6)


def test_attack_b_reward_gaslight():
    assert compute_attack_b_reward(1.0, 0.9, 0.7) == -1.0


def test_attack_b_reward_helpful():
    assert compute_attack_b_reward(0.5, 0.8, 0.5) == 1.0


def test_attack_b_reward_no_effect():
    assert compute_attack_b_reward(1.0, 0.9, 0.85) == 0.0


def test_heuristics_citation_bonus():
    text = "See 45 CFR §164.312 for technical safeguards."
    h = compute_heuristics(
        p_final=0.5,
        v_label=1.0,
        completion_text=text,
        use_presidio=False,
    )
    assert h.citation_bonus == 0.15
    auto = composite_auto_reward(0.25, h)
    assert auto > 0.25


@pytest.mark.parametrize(
    "use_presidio",
    [False],
)
def test_heuristics_no_presidio(use_presidio: bool):
    h = compute_heuristics(
        p_final=0.9,
        v_label=0.8,
        completion_text="No regulatory cite here.",
        use_presidio=use_presidio,
    )
    assert h.phi_penalty == 0.0
