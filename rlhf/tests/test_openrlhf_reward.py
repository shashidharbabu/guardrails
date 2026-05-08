import pytest

healthcare_reward = pytest.importorskip(
    "openrlhf.reward_fn.healthcare_reward",
    reason="openrlhf package not installed — skipping",
)

extract_confidence = healthcare_reward.extract_confidence
scalar_reward = healthcare_reward.scalar_reward


def test_extract_confidence():
    text = "Answer here.\nCONFIDENCE: 0.82"
    assert extract_confidence(text) == pytest.approx(0.82)


def test_scalar_reward_no_confidence_line():
    r = scalar_reward("no confidence here", 1.0, use_presidio=False)
    assert r == -0.20


def test_scalar_reward_with_citation():
    text = "HIPAA 45 CFR §164.312 requires safeguards.\nCONFIDENCE: 0.9"
    r = scalar_reward(text, 1.0, use_presidio=False)
    assert r > 0.5
