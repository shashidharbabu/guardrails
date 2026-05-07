"""
confidence/ — Confidence Scoring Engine (CSE) v1.1

Enterprise-grade scoring component that combines DeepEval signals and
MAD judge verdicts into a final routing decision.

Public API::

    from confidence.scorer import ConfidenceScorer
    from confidence.cse_types import CSEResult, RoutingDecision, ScoringMode
    from confidence.cse_config import CSEScoringConfig

    result = ConfidenceScorer().score(
        query, llm_answer, rag_chunks, final_claims, judge_verdicts
    )
    print(result.routing_decision)   # DELIVER / RETRY / HUMAN_REVIEW / HARD_BLOCK
    print(result.explanation)
    print(result.score_breakdown.applied_weights)

Backward-compatible fields (existing callers unchanged):
    result.final_score        — float 0.0–1.0
    result.routing_decision   — str
    result.components         — ComponentScores (f_llm, h_llm, relevancy, judge_eval)
    result.version            — str
    result.hard_blocked       — bool
    result.error              — str
"""
from confidence.cse_types import CSEResult, RoutingDecision, ScoringMode
from confidence.scorer import ConfidenceScorer

__all__ = [
    "ConfidenceScorer",
    "CSEResult",
    "RoutingDecision",
    "ScoringMode",
]
