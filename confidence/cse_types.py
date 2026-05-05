"""
confidence/types.py — CSEResult dataclass.

Returned by ConfidenceScorer.score(). Contains the full breakdown
of the 4-component formula so every caller (API, SDK, GRPO loop)
can inspect individual component scores.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ComponentScores:
    """Individual scores for each of the 4 CSE formula components."""
    f_llm:           float   # 0–1  faithfulness (DeepEval)
    h_llm:           float   # 0–1  hallucination rate — use (1-h_llm) in formula
    relevancy:       float   # 0–1  context relevancy (DeepEval)
    judge_eval:      float   # 0–1  MAD judge aggregate (min material + mean non-material)

    def as_dict(self) -> dict:
        return {
            "f_llm":      round(self.f_llm, 4),
            "h_llm":      round(self.h_llm, 4),
            "relevancy":  round(self.relevancy, 4),
            "judge_eval": round(self.judge_eval, 4),
        }


@dataclass
class CSEResult:
    """
    Full output of one CSE scoring run.

    final_score is the weighted combination:
        0.30 * f_llm + 0.25 * (1 - h_llm) + 0.10 * relevancy + 0.35 * judge_eval

    routing_decision follows the MAD routing thresholds:
        HARD_BLOCK   — any is_material judge score = 0.0 (hard rule, checked first)
        DELIVER      — final_score >= 0.8
        RETRY        — 0.4 <= final_score < 0.8
        HUMAN_REVIEW — final_score < 0.4

    version:
        "v0.1"  — judge-only aggregate (DeepEval unavailable / fallback)
        "v2.0"  — full 4-component formula with DeepEval
    """
    final_score:      float
    routing_decision: str
    components:       ComponentScores
    version:          str   = "v2.0"
    hard_blocked:     bool  = False   # True when HARD_BLOCK triggered by is_material v=0.0
    error:            str   = ""      # non-empty if DeepEval failed and v0.1 was used

    def as_dict(self) -> dict:
        return {
            "final_score":      round(self.final_score, 4),
            "routing_decision": self.routing_decision,
            "components":       self.components.as_dict(),
            "version":          self.version,
            "hard_blocked":     self.hard_blocked,
            "error":            self.error,
        }
