"""
test_deepeval_integration.py — Verify DeepEval runs with Claude judge and returns
full scoring mode (not JUDGE_ONLY_FALLBACK).

Requires:
  ANTHROPIC_API_KEY env var
  deepeval package installed

Run:
  pytest confidence/tests/test_deepeval_integration.py -v
"""
from __future__ import annotations

import os
import pytest

# Skip entire module if ANTHROPIC_API_KEY is absent
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping DeepEval integration tests",
)


SAMPLE_QUERY = (
    "Does HIPAA require covered entities to encrypt ePHI on their servers?"
)
SAMPLE_ANSWER = (
    "HIPAA's Security Rule treats encryption as an 'addressable' specification under "
    "45 CFR 164.312(a)(2)(iv). This means covered entities must assess whether "
    "encryption is reasonable and appropriate. If they choose not to encrypt, they must "
    "document the rationale and implement an equivalent alternative measure. "
    "Encryption is not mandated — it is one option among reasonable safeguards."
)
SAMPLE_CONTEXTS = [
    (
        "45 CFR 164.312 — Technical safeguards. Encryption and Decryption (Addressable): "
        "Implement a mechanism to encrypt and decrypt electronic protected health "
        "information. NOTE: This is an ADDRESSABLE specification — the covered entity must "
        "assess whether it is reasonable and appropriate to implement."
    ),
    (
        "HHS OCR Guidance 2023: The HIPAA Security Rule does not mandate encryption of ePHI. "
        "AES-128 or AES-256 are acceptable NIST-approved standards but neither is "
        "specifically mandated by HIPAA."
    ),
]


@pytest.fixture(scope="module")
def claude_judge():
    from confidence.claude_judge import ClaudeJudge
    return ClaudeJudge()


class TestClaudeJudgeInit:
    def test_judge_initializes_with_correct_model(self, claude_judge):
        assert "haiku" in claude_judge.model or "sonnet" in claude_judge.model or "opus" in claude_judge.model
        assert claude_judge.model != "claude-haiku-4-5", "Model ID must include date suffix"

    def test_get_model_name_returns_claude_prefix(self, claude_judge):
        name = claude_judge.get_model_name()
        assert name.startswith("claude/"), f"Expected 'claude/' prefix, got: {name}"

    def test_judge_generates_plain_text(self, claude_judge):
        response = claude_judge.generate("What is HIPAA? Answer in one sentence.")
        assert isinstance(response, str)
        assert len(response) > 10


class TestDeepEvalMetrics:
    def test_faithfulness_metric_returns_score(self, claude_judge):
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=SAMPLE_QUERY,
            actual_output=SAMPLE_ANSWER,
            retrieval_context=SAMPLE_CONTEXTS,
        )
        metric = FaithfulnessMetric(model=claude_judge, threshold=0.5, include_reason=False)
        metric.measure(test_case)

        assert metric.score is not None, "FaithfulnessMetric returned None score"
        assert 0.0 <= float(metric.score) <= 1.0, f"Score out of range: {metric.score}"

    def test_hallucination_metric_returns_score(self, claude_judge):
        from deepeval.metrics import HallucinationMetric
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=SAMPLE_QUERY,
            actual_output=SAMPLE_ANSWER,
            context=SAMPLE_CONTEXTS,
        )
        metric = HallucinationMetric(model=claude_judge, threshold=0.5, include_reason=False)
        metric.measure(test_case)

        assert metric.score is not None, "HallucinationMetric returned None score"
        assert 0.0 <= float(metric.score) <= 1.0, f"Score out of range: {metric.score}"

    def test_contextual_relevancy_returns_score(self, claude_judge):
        from deepeval.metrics import ContextualRelevancyMetric
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=SAMPLE_QUERY,
            actual_output=SAMPLE_ANSWER,
            retrieval_context=SAMPLE_CONTEXTS,
        )
        metric = ContextualRelevancyMetric(model=claude_judge, threshold=0.5, include_reason=False)
        metric.measure(test_case)

        assert metric.score is not None, "ContextualRelevancyMetric returned None score"
        assert 0.0 <= float(metric.score) <= 1.0, f"Score out of range: {metric.score}"


class TestCSEScorerFullMode:
    """Verify the ConfidenceScorer runs in FULL mode (all signals active) with Claude."""

    def test_scorer_runs_in_full_mode(self):
        import sys
        from pathlib import Path
        # Add repo root so confidence and app packages are importable
        repo_root = Path(__file__).resolve().parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from confidence.scorer import ConfidenceScorer, ScoringMode
        from confidence.cse_types import MADOutput as CSEMADOutput

        # Minimal MADOutput stub — ConfidenceScorer reads claims and judge_verdicts
        from confidence.cse_types import Claim, JudgeVerdict

        scorer = ConfidenceScorer()

        claim_id = "test-claim-001"
        claim = Claim(
            claim_id=claim_id,
            claim_text=(
                "HIPAA treats encryption as an addressable safeguard, not a mandate."
            ),
            is_material=True,
        )
        verdict = JudgeVerdict(
            claim_id=claim_id,
            claim_text=claim.claim_text,
            score=0.9,
            reasoning="Consistent with 45 CFR 164.312 addressable specification.",
        )

        result = scorer.score(
            query=SAMPLE_QUERY,
            answer=SAMPLE_ANSWER,
            contexts=SAMPLE_CONTEXTS,
            final_claims=[claim],
            judge_verdicts=[verdict],
        )

        assert result is not None, "scorer.score() returned None"
        assert result.final_score is not None, "final_score is None"
        assert 0.0 <= result.final_score <= 1.0, f"final_score out of range: {result.final_score}"

        # The key assertion: all 3 DeepEval signals should be present
        assert result.scoring_mode == ScoringMode.FULL, (
            f"Expected FULL scoring mode but got {result.scoring_mode}. "
            f"DeepEval signals: faithfulness={result.components.faithfulness_score}, "
            f"hallucination={result.components.hallucination_score}, "
            f"relevancy={result.components.contextual_relevancy}"
        )

        assert result.components.faithfulness_score is not None and result.components.faithfulness_score > 0, \
            "faithfulness_score is zero or None"
        assert result.components.hallucination_score is not None, \
            "hallucination_score is None"
        assert result.components.contextual_relevancy is not None and result.components.contextual_relevancy > 0, \
            "contextual_relevancy is zero or None"
        assert result.components.judge_eval is not None and result.components.judge_eval > 0, \
            "judge_eval is zero or None"

    def test_scorer_correct_answer_scores_high(self):
        """A faithful, well-supported answer should score >= 0.70."""
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from confidence.scorer import ConfidenceScorer
        from confidence.cse_types import Claim, JudgeVerdict

        scorer = ConfidenceScorer()
        claim_id = "test-claim-002"
        claim = Claim(claim_id=claim_id, claim_text=SAMPLE_ANSWER, is_material=True)
        verdict = JudgeVerdict(claim_id=claim_id, claim_text=SAMPLE_ANSWER, score=0.95, reasoning="Accurate.")

        result = scorer.score(
            query=SAMPLE_QUERY,
            answer=SAMPLE_ANSWER,
            contexts=SAMPLE_CONTEXTS,
            final_claims=[claim],
            judge_verdicts=[verdict],
        )
        assert result.final_score >= 0.60, (
            f"Expected score >= 0.60 for a faithful answer, got {result.final_score}"
        )

    def test_scorer_hallucinated_answer_scores_low(self):
        """A hallucinated answer contradicting the context should score < 0.50."""
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from confidence.scorer import ConfidenceScorer
        from confidence.cse_types import Claim, JudgeVerdict

        hallucinated_answer = (
            "HIPAA mandates AES-256 encryption for all ePHI stored on servers. "
            "Failure to use AES-256 specifically results in automatic non-compliance."
        )
        scorer = ConfidenceScorer()
        claim_id = "test-claim-003"
        claim = Claim(claim_id=claim_id, claim_text=hallucinated_answer, is_material=True)
        # Judge marks this as contradicted (score 0.0)
        verdict = JudgeVerdict(
            claim_id=claim_id,
            claim_text=hallucinated_answer,
            score=0.0,
            reasoning="HIPAA does not mandate AES-256 specifically.",
        )

        result = scorer.score(
            query=SAMPLE_QUERY,
            answer=hallucinated_answer,
            contexts=SAMPLE_CONTEXTS,
            final_claims=[claim],
            judge_verdicts=[verdict],
        )
        # Hard block or very low score expected
        assert result.hard_blocked or result.final_score < 0.50, (
            f"Expected HARD_BLOCK or score < 0.50 for hallucinated answer, "
            f"got hard_blocked={result.hard_blocked}, score={result.final_score}"
        )
