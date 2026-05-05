"""
confidence/ — Confidence Scoring Engine (CSE)

Full 4-component formula:
    final = 0.30 * F_llm
          + 0.25 * (1 - H_llm)
          + 0.10 * relevancy
          + 0.35 * judge_eval_score

Public API:
    from confidence.scorer import ConfidenceScorer
    from confidence.types import CSEResult

    result = ConfidenceScorer().score(
        query, llm_answer, rag_chunks, final_claims, judge_verdicts
    )
    print(result.final_score, result.routing_decision)
"""
from confidence.cse_types import CSEResult
from confidence.scorer import ConfidenceScorer

__all__ = ["ConfidenceScorer", "CSEResult"]
