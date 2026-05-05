"""
confidence/scorer.py — ConfidenceScorer: full 4-component CSE formula.

FORMULA (v2.0)
--------------
    final = 0.30 * F_llm
          + 0.25 * (1 - H_llm)
          + 0.10 * relevancy
          + 0.35 * judge_eval_score

COMPONENTS
----------
    F_llm         — FaithfulnessMetric (DeepEval).
                    Measures: "Does every claim in the answer follow from the
                    retrieved context?" Score 0.0–1.0.

    H_llm         — HallucinationMetric (DeepEval).
                    Measures: "What fraction of claims are NOT supported by
                    context?" Score 0.0–1.0 (higher = more hallucination).
                    We use (1 - H_llm) so a high hallucination hurts the score.

    relevancy     — ContextualRelevancyMetric (DeepEval).
                    Measures: "How relevant are the retrieved RAG chunks to the
                    query?" Score 0.0–1.0.

    judge_eval    — MAD judge aggregate (existing _aggregate_score logic).
                    Weighted min-mean of judge verdicts:
                      70% * min(material claims) + 30% * mean(non-material)

FALLBACK (v0.1)
---------------
    If DeepEval metrics fail (Ollama offline, timeout, etc.), scorer falls back
    to judge-only aggregate (judge_eval_score only) and returns version="v0.1".
    The f_llm / h_llm / relevancy components are set to their neutral defaults:
      f_llm=0.75, h_llm=0.25, relevancy=0.75
    so the formula still runs but only judge_eval_score carries real information.

ROUTING (same thresholds as mad_pipeline)
------------------------------------------
    HARD_BLOCK   — any is_material judge score = 0.0 (checked first)
    DELIVER      — final_score >= 0.8
    RETRY        — 0.4 <= final_score < 0.8
    HUMAN_REVIEW — final_score < 0.4
"""
from __future__ import annotations

import logging
from typing import List, Optional

from multi_agent.config import (
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_LOW,
)
from multi_agent.models import Claim, JudgeVerdict

from confidence.cse_types import CSEResult, ComponentScores

logger = logging.getLogger(__name__)

# ── Formula weights ────────────────────────────────────────────────────────────
W_FAITHFULNESS  = 0.30
W_HALLUCINATION = 0.25   # multiplied against (1 - h_llm)
W_RELEVANCY     = 0.10
W_JUDGE         = 0.35

# ── Neutral defaults used in v0.1 fallback ─────────────────────────────────────
_DEFAULT_F_LLM    = 0.75
_DEFAULT_H_LLM    = 0.25
_DEFAULT_RELEVANCY = 0.75


class ConfidenceScorer:
    """
    Main CSE scoring class.

    Typical usage:
        scorer = ConfidenceScorer()
        result = scorer.score(query, llm_answer, rag_chunks, claims, judge_verdicts)

    rag_chunks can be a list of Chunk objects (with .text attribute) or a
    list of plain strings — both are handled transparently.
    """

    def __init__(self, ollama_model: str = "qwen2.5:7b") -> None:
        self.ollama_model = ollama_model
        self._judge: Optional[object] = None   # lazy init — avoids import cost at startup

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        query:          str,
        llm_answer:     str,
        rag_chunks:     list,
        final_claims:   List[Claim],
        judge_verdicts: List[JudgeVerdict],
    ) -> CSEResult:
        """
        Compute the full CSE score for one MAD pipeline run.

        Parameters
        ----------
        query          : Original user query.
        llm_answer     : The LLM answer that passed through MAD.
        rag_chunks     : Retrieved context chunks (Chunk objects or strings).
        final_claims   : Final Claim objects after debate.
        judge_verdicts : JudgeVerdict objects from MAD judge.

        Returns
        -------
        CSEResult with final_score, routing_decision, per-component scores,
        version string, and any error message.
        """
        # ── Step 1: Hard-block check (always runs, no DeepEval needed) ────────
        hard_blocked = self._check_hard_block(final_claims, judge_verdicts)

        # ── Step 2: Judge aggregate (always available) ─────────────────────────
        judge_eval = self._aggregate_judge(judge_verdicts)

        # ── Step 3: DeepEval metrics (may fall back to neutral defaults) ───────
        context_strings = _extract_context(rag_chunks)
        f_llm, h_llm, relevancy, error, version = self._run_deepeval(
            query, llm_answer, context_strings
        )

        # ── Step 4: Apply formula ──────────────────────────────────────────────
        final_score = _apply_formula(f_llm, h_llm, relevancy, judge_eval)

        # ── Step 5: Routing decision ───────────────────────────────────────────
        routing = self._routing(final_score, hard_blocked)

        components = ComponentScores(
            f_llm=f_llm,
            h_llm=h_llm,
            relevancy=relevancy,
            judge_eval=judge_eval,
        )

        logger.info(
            "[CSE] %s  final=%.4f  F=%.3f  H=%.3f  R=%.3f  J=%.3f  → %s",
            version, final_score, f_llm, h_llm, relevancy, judge_eval, routing,
        )

        return CSEResult(
            final_score=round(final_score, 4),
            routing_decision=routing,
            components=components,
            version=version,
            hard_blocked=hard_blocked,
            error=error,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_deepeval(
        self,
        query:    str,
        answer:   str,
        contexts: List[str],
    ) -> tuple[float, float, float, str, str]:
        """
        Run FaithfulnessMetric, HallucinationMetric, ContextualRelevancyMetric.

        Returns (f_llm, h_llm, relevancy, error_msg, version).
        Falls back to neutral defaults on any error.
        """
        if not contexts:
            logger.warning("[CSE] No RAG context — using neutral defaults for DeepEval metrics")
            return _DEFAULT_F_LLM, _DEFAULT_H_LLM, _DEFAULT_RELEVANCY, "no_context", "v0.1"

        try:
            from deepeval import evaluate
            from deepeval.metrics import (
                FaithfulnessMetric,
                HallucinationMetric,
                ContextualRelevancyMetric,
            )
            from deepeval.test_case import LLMTestCase

            judge = self._get_judge()

            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=contexts,
                context=contexts,
            )

            faith_metric   = FaithfulnessMetric(model=judge,   threshold=0.5, include_reason=False)
            halluc_metric  = HallucinationMetric(model=judge,  threshold=0.5, include_reason=False)
            relev_metric   = ContextualRelevancyMetric(model=judge, threshold=0.5, include_reason=False)

            # Run all three metrics synchronously (DeepEval's measure() is sync)
            faith_metric.measure(test_case)
            halluc_metric.measure(test_case)
            relev_metric.measure(test_case)

            f_llm    = float(faith_metric.score   or 0.0)
            h_llm    = float(halluc_metric.score  or 0.0)
            relevancy = float(relev_metric.score  or 0.0)

            return f_llm, h_llm, relevancy, "", "v2.0"

        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.warning("[CSE] DeepEval failed — falling back to v0.1 judge-only. Reason: %s", err)
            return _DEFAULT_F_LLM, _DEFAULT_H_LLM, _DEFAULT_RELEVANCY, err, "v0.1"

    def _get_judge(self) -> object:
        """Lazy-init OllamaJudge to avoid import cost at module load."""
        if self._judge is None:
            from confidence.ollama_judge import OllamaJudge
            self._judge = OllamaJudge(model=self.ollama_model)
        return self._judge

    @staticmethod
    def _check_hard_block(
        final_claims:   List[Claim],
        judge_verdicts: List[JudgeVerdict],
    ) -> bool:
        """Return True if any is_material claim has judge score = 0.0."""
        claim_map = {c.claim_id: c for c in final_claims}
        for jv in judge_verdicts:
            claim = claim_map.get(jv.claim_id)
            if claim and claim.is_material and jv.score == 0.0:
                logger.warning(
                    "[CSE] HARD_BLOCK — is_material claim %s scored 0.0: %s",
                    jv.claim_id, jv.claim_text[:80],
                )
                return True
        return False

    @staticmethod
    def _aggregate_judge(judge_verdicts: List[JudgeVerdict]) -> float:
        """
        Weighted min-mean aggregate of judge verdicts.
        Same logic as mad_pipeline._aggregate_score().
        """
        if not judge_verdicts:
            return 0.5

        mat  = [jv for jv in judge_verdicts if jv.is_material]
        nmat = [jv for jv in judge_verdicts if not jv.is_material]

        if mat and nmat:
            return round(
                0.70 * min(jv.score for jv in mat)
                + 0.30 * (sum(jv.score for jv in nmat) / len(nmat)),
                4,
            )
        if mat:
            return round(min(jv.score for jv in mat), 4)
        return round(sum(jv.score for jv in nmat) / len(nmat), 4)

    @staticmethod
    def _routing(final_score: float, hard_blocked: bool) -> str:
        if hard_blocked:
            return "HARD_BLOCK"
        if final_score >= CONFIDENCE_THRESHOLD_HIGH:
            return "DELIVER"
        if final_score >= CONFIDENCE_THRESHOLD_LOW:
            return "RETRY"
        return "HUMAN_REVIEW"


# ── Formula ───────────────────────────────────────────────────────────────────

def _apply_formula(
    f_llm:    float,
    h_llm:    float,
    relevancy: float,
    judge_eval: float,
) -> float:
    """
    final = 0.30*F_llm + 0.25*(1-H_llm) + 0.10*relevancy + 0.35*judge_eval
    """
    return (
        W_FAITHFULNESS  * f_llm
        + W_HALLUCINATION * (1.0 - h_llm)
        + W_RELEVANCY     * relevancy
        + W_JUDGE         * judge_eval
    )


def _extract_context(rag_chunks: list) -> List[str]:
    """
    Accept either Chunk objects (with .text) or plain strings.
    Returns a list of non-empty strings for DeepEval.
    """
    texts = []
    for chunk in rag_chunks:
        if isinstance(chunk, str):
            texts.append(chunk)
        elif hasattr(chunk, "text"):
            texts.append(chunk.text)
        else:
            texts.append(str(chunk))
    return [t for t in texts if t.strip()]
