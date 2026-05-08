"""
confidence/scorer.py — Enterprise Confidence Scoring Engine v1.1

SCORING FORMULA
---------------
    weighted_raw = Σ (weight_i × signal_i)  for all available signals
    final_score  = clamp(weighted_raw − penalties, 0.0, 1.0)

    Signals and default weights (FULL mode):
        faithfulness_score      0.25  — F_llm (DeepEval)
        hallucination_inverse   0.20  — 1 − H_llm (DeepEval)
        contextual_relevancy    0.15  — relevancy (DeepEval)
        judge_eval              0.35  — MAD judge aggregate
        context_quality         0.05  — lightweight context checks

    When a signal is unavailable its weight is dropped and the remaining
    weights are re-normalised to sum to 1.0.  Missing signals are NEVER
    substituted with a neutral 0/0.5 — their absence is reflected in
    scoring_mode and triggered_flags.

ROUTING PRIORITY
----------------
    1. HARD_BLOCK   — any material claim judged false (score=0.0); overrides all
    2. HUMAN_REVIEW — score < human_review_threshold (default 0.55),
                      or ERROR_FALLBACK, or retry limit reached
    3. RETRY        — human_review_threshold ≤ score < deliver_threshold (0.78)
    4. DELIVER      — score ≥ deliver_threshold, no hard flags

FALLBACK MODES
--------------
    FULL                  — DeepEval + MAD judge both available
    JUDGE_ONLY_FALLBACK   — DeepEval failed; re-weight across judge + claims
    CONTEXT_ONLY_FALLBACK — MAD judge unavailable; DeepEval signals only
    ERROR_FALLBACK        — both unavailable; forced HUMAN_REVIEW

VERSIONING
----------
    cse_version:    cse_v1.1_weighted_claim_aware
    config_version: v1.1
    formula_version: weighted_claim_aware_v1
"""
from __future__ import annotations

import logging
import os
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    from multi_agent.models import Claim, JudgeVerdict
except ImportError:
    from multi_agent_debate.multi_agent.models import Claim, JudgeVerdict

from confidence.cse_config import CSEScoringConfig, DEFAULT_CONFIG
from confidence.cse_types import (
    CSEMetadata,
    CSEResult,
    ComponentScores,
    FailedClaim,
    RoutingDecision,
    ScoreBreakdown,
    ScoringMode,
    TriggeredFlags,
)

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Enterprise Confidence Scoring Engine.

    Usage::

        scorer = ConfidenceScorer()
        result = scorer.score(query, llm_answer, rag_chunks, claims, judge_verdicts)
        print(result.routing_decision, result.explanation)

    All scoring behaviour is driven by ``CSEScoringConfig``.  Pass a custom
    config instance to override weights, thresholds, or penalties without
    touching source code.

    RAG chunks may be Chunk objects (with a ``.text`` attribute) or plain
    strings — both are handled transparently.
    """

    def __init__(
        self,
        config:        Optional[CSEScoringConfig] = None,
        ollama_model:  str = "qwen2.5:7b",
    ) -> None:
        self.cfg  = config or DEFAULT_CONFIG
        self.ollama_model = ollama_model
        self._judge: Optional[object] = None   # lazy-init — avoids import cost at startup

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        query:          str,
        llm_answer:     str,
        rag_chunks:     list,
        final_claims:   List[Claim],
        judge_verdicts: List[JudgeVerdict],
        *,
        retry_count:    int  = 0,
        request_id:     Optional[str] = None,
        model_name:     Optional[str] = None,
        policy_domain:  Optional[str] = None,
    ) -> CSEResult:
        """
        Compute the full CSE score for one MAD pipeline run.

        Parameters
        ----------
        query, llm_answer  : Inputs to the LLM.
        rag_chunks         : Retrieved context (Chunk objects or strings).
        final_claims       : Claim objects after MAD debate.
        judge_verdicts     : JudgeVerdict objects from MAD judge.
        retry_count        : How many times this query has already been retried.
        request_id         : Optional correlation ID for audit trail.
        model_name         : LLM model that produced the answer.
        policy_domain      : Domain label (e.g. "HIPAA", "GDPR").
        """
        cfg = self.cfg
        context_strings = _extract_context(rag_chunks)
        chunk_ids       = _extract_chunk_ids(rag_chunks)

        # ── Step 1: Hard-block check (no DeepEval needed) ─────────────────────
        hard_blocked, block_claims = _check_hard_block(final_claims, judge_verdicts, cfg)

        # ── Step 2: MAD judge aggregate ────────────────────────────────────────
        judge_eval, judge_disagree = _aggregate_judge(judge_verdicts, cfg)
        judge_available = bool(judge_verdicts)

        # ── Step 3: DeepEval metrics ───────────────────────────────────────────
        f_llm, h_llm, relevancy, deepeval_err = self._run_deepeval(
            query, llm_answer, context_strings
        )
        deepeval_available = deepeval_err == "" and f_llm is not None

        # ── Step 4: Context quality signal ────────────────────────────────────
        ctx_quality, ctx_flags, ctx_penalty = _context_quality(
            context_strings, relevancy, cfg
        )

        # ── Step 5: Claim severity-weighted score ──────────────────────────────
        claim_quality, failed_claims = _claim_quality(
            final_claims, judge_verdicts, cfg
        )

        # ── Step 6: Determine scoring mode ────────────────────────────────────
        scoring_mode = _resolve_scoring_mode(deepeval_available, judge_available)

        # ── Step 7: Build signal dict and normalise weights ───────────────────
        raw_signals: Dict[str, Optional[float]] = {}
        raw_weights: Dict[str, float] = {}

        if deepeval_available:
            raw_signals["faithfulness"]    = f_llm
            raw_signals["halluc_inverse"]  = (1.0 - h_llm) if h_llm is not None else None
            raw_signals["contextual_rel"]  = relevancy
            raw_weights["faithfulness"]    = cfg.w_faithfulness
            raw_weights["halluc_inverse"]  = cfg.w_hallucination_inverse
            raw_weights["contextual_rel"]  = cfg.w_contextual_relevancy

        if judge_available:
            raw_signals["judge_eval"]  = judge_eval
            raw_weights["judge_eval"]  = cfg.w_judge_eval

        if ctx_quality is not None:
            raw_signals["ctx_quality"] = ctx_quality
            raw_weights["ctx_quality"] = cfg.w_context_quality

        available = {k: v for k, v in raw_signals.items() if v is not None}
        normed_weights = _normalise_weights(
            {k: raw_weights[k] for k in available}
        )

        # ── Step 8: Weighted score ─────────────────────────────────────────────
        if available:
            weighted_raw = sum(normed_weights[k] * available[k] for k in available)
        else:
            weighted_raw = 0.3   # error-fallback neutral-low

        # ── Step 9: Penalties ──────────────────────────────────────────────────
        penalties = ctx_penalty
        if judge_disagree:
            penalties += cfg.penalty_judge_disagreement

        final_score = max(0.0, min(1.0, weighted_raw - penalties))
        final_score = round(final_score, 4)

        # ── Step 10: Triggered flags ───────────────────────────────────────────
        critical_claim_failed = any(
            f.severity == "critical" for f in block_claims
        )
        flags = TriggeredFlags(
            material_claim_failed     = hard_blocked,
            critical_claim_failed     = critical_claim_failed,
            missing_context           = not context_strings,
            deepeval_unavailable      = not deepeval_available,
            low_context_relevancy     = ctx_flags.get("low_relevancy", False),
            low_faithfulness          = (f_llm is not None and f_llm < 0.4),
            high_hallucination_risk   = (h_llm is not None and h_llm > 0.6),
            judge_disagreement        = judge_disagree,
            unsupported_claims_present = any(
                c.score <= 0.0 for c in (failed_claims or [])
            ),
        )

        # ── Step 11: Routing ───────────────────────────────────────────────────
        routing, retry_reasons, review_reasons, block_reasons = _route(
            final_score, hard_blocked, flags, scoring_mode,
            retry_count, cfg
        )

        # ── Step 12: Score breakdown ───────────────────────────────────────────
        breakdown = ScoreBreakdown(
            faithfulness_score         = f_llm,
            hallucination_score        = h_llm,
            hallucination_risk_inverse = (1.0 - h_llm) if h_llm is not None else None,
            contextual_relevancy_score = relevancy,
            judge_eval_score           = judge_eval,
            claim_quality_score        = claim_quality,
            context_quality_score      = ctx_quality,
            penalties                  = round(penalties, 4),
            applied_weights            = {k: round(v, 4) for k, v in normed_weights.items()},
        )

        # ── Step 13: Top failed claims (at most 5 most severe) ─────────────────
        top_failed = sorted(failed_claims, key=lambda c: c.score)[:5]

        # ── Step 14: Explanation ───────────────────────────────────────────────
        explanation = _explain(
            routing, final_score, scoring_mode, flags,
            retry_reasons, review_reasons, block_reasons,
            deepeval_err, top_failed,
        )

        # ── Step 15: Metadata ──────────────────────────────────────────────────
        metadata = CSEMetadata(
            timestamp      = datetime.now(timezone.utc).isoformat(),
            cse_version    = cfg.cse_version,
            config_version = cfg.config_version,
            formula_version= cfg.formula_version,
            scoring_mode   = scoring_mode.value,
            request_id     = request_id,
            model_name     = model_name,
            policy_domain  = policy_domain,
            source_chunk_ids = chunk_ids,
        )

        # ── Legacy ComponentScores ─────────────────────────────────────────────
        components = ComponentScores(
            f_llm      = f_llm      if f_llm      is not None else 0.75,
            h_llm      = h_llm      if h_llm      is not None else 0.25,
            relevancy  = relevancy  if relevancy  is not None else 0.75,
            judge_eval = judge_eval,
        )

        logger.info(
            "[CSE %s] mode=%s final=%.4f route=%s F=%s H=%s R=%s J=%.3f penalties=%.3f",
            cfg.cse_version, scoring_mode.value, final_score, routing,
            f"{f_llm:.3f}" if f_llm is not None else "N/A",
            f"{h_llm:.3f}" if h_llm is not None else "N/A",
            f"{relevancy:.3f}" if relevancy is not None else "N/A",
            judge_eval, penalties,
        )

        return CSEResult(
            final_score       = final_score,
            routing_decision  = routing,
            scoring_mode      = scoring_mode.value,
            score_breakdown   = breakdown,
            triggered_flags   = flags,
            top_failed_claims = top_failed,
            explanation       = explanation,
            retry_reasons     = retry_reasons,
            review_reasons    = review_reasons,
            block_reasons     = block_reasons,
            metadata          = metadata,
            components        = components,
            version           = cfg.cse_version,
            hard_blocked      = hard_blocked,
            error             = deepeval_err,
        )

    # ── DeepEval integration ──────────────────────────────────────────────────

    def _run_deepeval(
        self,
        query:    str,
        answer:   str,
        contexts: List[str],
    ) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
        """
        Run FaithfulnessMetric, HallucinationMetric, ContextualRelevancyMetric.

        Returns (f_llm, h_llm, relevancy, error_message).
        Returns (None, None, None, reason) when DeepEval is unavailable —
        callers must not substitute 0 for None.
        """
        if not contexts:
            logger.warning("[CSE] No RAG context — DeepEval skipped (missing_context)")
            return None, None, None, "no_context"

        try:
            from deepeval.metrics import (
                ContextualRelevancyMetric,
                FaithfulnessMetric,
                HallucinationMetric,
            )
            from deepeval.test_case import LLMTestCase

            judge = self._get_judge()
            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=contexts,
                context=contexts,
            )

            faith_metric  = FaithfulnessMetric(model=judge,  threshold=0.5, include_reason=False)
            halluc_metric = HallucinationMetric(model=judge, threshold=0.5, include_reason=False)
            relev_metric  = ContextualRelevancyMetric(model=judge, threshold=0.5, include_reason=False)

            faith_metric.measure(test_case)
            halluc_metric.measure(test_case)
            relev_metric.measure(test_case)

            f_llm    = float(faith_metric.score  or 0.0)
            h_llm    = float(halluc_metric.score or 0.0)
            relevancy = float(relev_metric.score  or 0.0)

            return f_llm, h_llm, relevancy, ""

        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.warning("[CSE] DeepEval failed — JUDGE_ONLY_FALLBACK. Reason: %s", err)
            return None, None, None, err

    def _get_judge(self) -> object:
        """Lazy-init the DeepEval judge backend (Ollama or Claude)."""
        if self._judge is None:
            backend = os.getenv("JUDGE_BACKEND", "ollama").lower()
            if backend == "claude":
                from confidence.claude_judge import ClaudeJudge
                self._judge = ClaudeJudge()
            else:
                from confidence.ollama_judge import OllamaJudge
                self._judge = OllamaJudge(model=self.ollama_model)
        return self._judge


# ── Pure functions ────────────────────────────────────────────────────────────
# All stateless — easy to unit-test without instantiating ConfidenceScorer.

def _check_hard_block(
    final_claims:   List[Claim],
    judge_verdicts: List[JudgeVerdict],
    cfg:            CSEScoringConfig,
) -> Tuple[bool, List[FailedClaim]]:
    """
    HARD_BLOCK if any material or critical-severity claim has judge score == 0.0.

    severity="critical" claims trigger HARD_BLOCK independently of is_material.
    Returns (hard_blocked, list_of_failed_claims).
    """
    claim_map   = {c.claim_id: c for c in final_claims}
    hard_block  = False
    failed: List[FailedClaim] = []

    for jv in judge_verdicts:
        claim = claim_map.get(jv.claim_id)
        if not claim:
            continue

        sev = getattr(claim, "severity", None)
        is_blocking = claim.is_material or sev == "critical"

        if is_blocking and jv.score == 0.0:
            hard_block = True
            sev_label = sev if sev else "material"
            logger.warning(
                "[CSE] HARD_BLOCK triggered by claim %s (severity=%s, score=0.0): %s",
                jv.claim_id, sev_label, jv.claim_text[:80],
            )
            failed.append(FailedClaim(
                claim_id   = jv.claim_id,
                claim_text = jv.claim_text,
                verdict    = "contradicted",
                severity   = sev_label,
                score      = 0.0,
            ))
        elif jv.score < 0.5:
            severity_label = sev if sev else ("material" if claim.is_material else "minor")
            failed.append(FailedClaim(
                claim_id   = jv.claim_id,
                claim_text = jv.claim_text,
                verdict    = "unsupported" if jv.score == 0.0 else "partially_supported",
                severity   = severity_label,
                score      = jv.score,
            ))

    return hard_block, failed


def _aggregate_judge(
    judge_verdicts: List[JudgeVerdict],
    cfg:            CSEScoringConfig,
) -> Tuple[float, bool]:
    """
    Weighted min-mean aggregate plus disagreement detection.

    Returns (aggregate_score, judge_disagreement_flag).
    """
    if not judge_verdicts:
        return 0.5, False

    mat  = [jv for jv in judge_verdicts if jv.is_material]
    nmat = [jv for jv in judge_verdicts if not jv.is_material]

    if mat and nmat:
        agg = round(
            0.70 * min(jv.score for jv in mat)
            + 0.30 * (sum(jv.score for jv in nmat) / len(nmat)),
            4,
        )
    elif mat:
        agg = round(min(jv.score for jv in mat), 4)
    else:
        agg = round(sum(jv.score for jv in nmat) / len(nmat), 4)

    # Disagreement: high std-dev across all judge scores indicates conflicting verdicts
    scores = [jv.score for jv in judge_verdicts]
    disagree = False
    if len(scores) >= 2:
        try:
            std = statistics.stdev(scores)
            disagree = std >= cfg.judge_disagreement_std_threshold
        except statistics.StatisticsError:
            pass

    return agg, disagree


def _context_quality(
    contexts: List[str],
    relevancy: Optional[float],
    cfg:       CSEScoringConfig,
) -> Tuple[Optional[float], dict, float]:
    """
    Lightweight context quality signal.

    Returns (quality_score, flags_dict, penalty_total).
    quality_score is None when there is no context (no signal to report).
    """
    if not contexts:
        return None, {"no_context": True}, cfg.penalty_missing_context

    flags:    dict  = {}
    penalty:  float = 0.0
    score:    float = 1.0

    # Not enough chunks
    if len(contexts) < cfg.min_chunks_required:
        flags["insufficient_chunks"] = True
        penalty += cfg.penalty_insufficient_chunks
        score   -= 0.15

    # Low relevancy from DeepEval (only available in FULL mode)
    if relevancy is not None and relevancy < cfg.low_relevancy_threshold:
        flags["low_relevancy"] = True
        penalty += cfg.penalty_low_context_relevancy
        score   -= 0.15

    score = max(0.0, min(1.0, round(score, 4)))
    return score, flags, round(penalty, 4)


def _claim_quality(
    final_claims:   List[Claim],
    judge_verdicts: List[JudgeVerdict],
    cfg:            CSEScoringConfig,
) -> Tuple[Optional[float], List[FailedClaim]]:
    """
    Severity-weighted aggregate of claim verdicts.

    Contribution per claim:
        supported      →  +severity_weight
        partial        →  +severity_weight * 0.5  (small penalty via low contribution)
        unsupported    →  −severity_weight * 0.4  (medium penalty)
        contradicted   →  −severity_weight * 1.0  (high penalty)

    Returns (claim_quality_score 0–1 or None, failed_claims list).
    """
    if not judge_verdicts:
        return None, []

    claim_map = {c.claim_id: c for c in final_claims}
    total_weight  = 0.0
    weighted_sum  = 0.0
    failed: List[FailedClaim] = []

    for jv in judge_verdicts:
        claim = claim_map.get(jv.claim_id)
        sev = getattr(claim, "severity", None) if claim else None
        if sev == "critical":
            sev_weight = cfg.severity_critical
            sev_label  = "critical"
        elif sev == "minor":
            sev_weight = cfg.severity_minor
            sev_label  = "minor"
        elif sev == "material" or (claim and claim.is_material):
            sev_weight = cfg.severity_material
            sev_label  = "material"
        else:
            sev_weight = cfg.severity_minor
            sev_label  = "minor"

        total_weight += sev_weight

        if jv.score >= 0.8:
            contribution = sev_weight
        elif jv.score >= 0.4:
            contribution = sev_weight * 0.5
        elif jv.score > 0.0:
            contribution = 0.0   # unsupported — no positive contribution
            failed.append(FailedClaim(
                claim_id   = jv.claim_id,
                claim_text = jv.claim_text,
                verdict    = "partially_supported",
                severity   = sev_label,
                score      = jv.score,
            ))
        else:
            contribution = 0.0   # contradicted — no positive contribution
            if not any(f.claim_id == jv.claim_id for f in failed):
                failed.append(FailedClaim(
                    claim_id   = jv.claim_id,
                    claim_text = jv.claim_text,
                    verdict    = "contradicted",
                    severity   = sev_label,
                    score      = 0.0,
                ))

        weighted_sum += contribution

    if total_weight == 0.0:
        return None, failed

    score = max(0.0, min(1.0, round(weighted_sum / total_weight, 4)))
    return score, failed


def _resolve_scoring_mode(
    deepeval_available: bool,
    judge_available:    bool,
) -> ScoringMode:
    if deepeval_available and judge_available:
        return ScoringMode.FULL
    if judge_available:
        return ScoringMode.JUDGE_ONLY_FALLBACK
    if deepeval_available:
        return ScoringMode.CONTEXT_ONLY_FALLBACK
    return ScoringMode.ERROR_FALLBACK


def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Re-normalise a weight dict so values sum to 1.0.
    Returns an empty dict if total is 0 (avoids division by zero).
    """
    total = sum(weights.values())
    if total == 0.0:
        return {}
    return {k: v / total for k, v in weights.items()}


def _route(
    final_score:  float,
    hard_blocked: bool,
    flags:        TriggeredFlags,
    scoring_mode: ScoringMode,
    retry_count:  int,
    cfg:          CSEScoringConfig,
) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Deterministic routing in priority order.

    Returns (routing_decision, retry_reasons, review_reasons, block_reasons).
    """
    retry_reasons:  List[str] = []
    review_reasons: List[str] = []
    block_reasons:  List[str] = []

    # ── A. HARD_BLOCK ──────────────────────────────────────────────────────────
    if hard_blocked:
        block_reasons.append("A material claim was judged clearly false (score=0.0).")
        return RoutingDecision.HARD_BLOCK.value, [], [], block_reasons

    # ── B. HUMAN_REVIEW conditions ─────────────────────────────────────────────
    def _force_review(reason: str) -> Tuple[str, List[str], List[str], List[str]]:
        review_reasons.append(reason)
        return RoutingDecision.HUMAN_REVIEW.value, [], review_reasons, []

    if scoring_mode == ScoringMode.ERROR_FALLBACK:
        return _force_review(
            "Both DeepEval and MAD judge signals are unavailable — cannot deliver automatically."
        )

    if retry_count >= cfg.max_retry_count:
        return _force_review(
            f"Retry limit reached ({retry_count}/{cfg.max_retry_count}). "
            "Escalating to human review."
        )

    if final_score < cfg.human_review_threshold:
        review_reasons.append(
            f"Final score {final_score:.3f} is below the human-review threshold "
            f"({cfg.human_review_threshold})."
        )
        if flags.deepeval_unavailable and final_score < cfg.human_review_threshold:
            review_reasons.append(
                "DeepEval is unavailable and judge signal is also weak — "
                "insufficient evidence to retry."
            )
        if flags.unsupported_claims_present:
            review_reasons.append("Multiple claims are unsupported or contradicted.")
        if flags.missing_context:
            review_reasons.append(
                "No regulatory context was retrieved; the answer cannot be "
                "citation-grounded."
            )
        return RoutingDecision.HUMAN_REVIEW.value, [], review_reasons, []

    # ── C. RETRY ───────────────────────────────────────────────────────────────
    if final_score < cfg.deliver_threshold:
        if flags.low_faithfulness:
            retry_reasons.append(
                "Faithfulness score is low — answer may contain claims not grounded "
                "in the retrieved context."
            )
        if flags.high_hallucination_risk:
            retry_reasons.append(
                "High hallucination risk detected — regenerating with stricter "
                "citation grounding may help."
            )
        if flags.low_context_relevancy:
            retry_reasons.append(
                "Retrieved context has low relevancy — better retrieval may improve "
                "the answer quality."
            )
        if flags.unsupported_claims_present:
            retry_reasons.append(
                "Some claims are only partially supported — a more precise answer "
                "with explicit citations is needed."
            )
        if flags.judge_disagreement:
            retry_reasons.append(
                "Judge agents disagreed on claim verdicts — the answer is ambiguous "
                "and should be clarified."
            )
        if not retry_reasons:
            retry_reasons.append(
                f"Score {final_score:.3f} is below the delivery threshold "
                f"({cfg.deliver_threshold}) — answer quality needs improvement."
            )
        return RoutingDecision.RETRY.value, retry_reasons, [], []

    # ── D. DELIVER ─────────────────────────────────────────────────────────────
    return RoutingDecision.DELIVER.value, [], [], []


def _explain(
    routing:       str,
    final_score:   float,
    scoring_mode:  ScoringMode,
    flags:         TriggeredFlags,
    retry_reasons:  List[str],
    review_reasons: List[str],
    block_reasons:  List[str],
    deepeval_err:  str,
    top_failed:    List[FailedClaim],
) -> str:
    """Produce a single, concise human-readable explanation for the routing decision."""
    mode_note = (
        "" if scoring_mode == ScoringMode.FULL
        else f" [scoring mode: {scoring_mode.value}]"
    )

    if routing == RoutingDecision.HARD_BLOCK.value:
        claim_summary = ""
        if top_failed:
            claim_summary = (
                f" Blocked claim: \"{top_failed[0].claim_text[:80]}\""
                + ("..." if len(top_failed[0].claim_text) > 80 else "")
            )
        return (
            f"HARD_BLOCK: A material regulatory claim was judged clearly false.{claim_summary}"
            f"{mode_note}"
        )

    if routing == RoutingDecision.HUMAN_REVIEW.value:
        reason = review_reasons[0] if review_reasons else "Low confidence score."
        return f"HUMAN_REVIEW (score={final_score:.3f}): {reason}{mode_note}"

    if routing == RoutingDecision.RETRY.value:
        reason = retry_reasons[0] if retry_reasons else "Score below delivery threshold."
        return (
            f"RETRY (score={final_score:.3f}): {reason}{mode_note}"
        )

    # DELIVER
    fallback_note = ""
    if scoring_mode != ScoringMode.FULL:
        fallback_note = f" DeepEval was unavailable ({deepeval_err or 'no context'}); judge signal drove the decision."
    return (
        f"DELIVER (score={final_score:.3f}): All signals meet the delivery threshold.{fallback_note}"
    )


# ── Utility ───────────────────────────────────────────────────────────────────

def _extract_context(rag_chunks: list) -> List[str]:
    """Accept Chunk objects (with .text) or plain strings. Returns non-empty strings."""
    texts = []
    for chunk in rag_chunks:
        if isinstance(chunk, str):
            texts.append(chunk)
        elif hasattr(chunk, "text"):
            texts.append(chunk.text)
        else:
            texts.append(str(chunk))
    return [t for t in texts if t.strip()]


def _extract_chunk_ids(rag_chunks: list) -> List[str]:
    """Extract chunk_id strings for metadata — falls back to positional index."""
    ids = []
    for i, chunk in enumerate(rag_chunks):
        if hasattr(chunk, "chunk_id"):
            ids.append(str(chunk.chunk_id))
        else:
            ids.append(str(i))
    return ids
