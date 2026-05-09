from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mad.models import MADOutput, Claim, JudgeVerdict


class MADPipeline:
    def run(self, query: str, llm_answer: str) -> "MADOutput":
        from mad.mad_pipeline import run_mad
        return run_mad(query, llm_answer)


def run_mad(query: str, llm_answer: str) -> "MADOutput":
    from mad.mad_pipeline import run_mad as _run_mad
    return _run_mad(query, llm_answer)


def __getattr__(name: str):
    if name in ("MADOutput", "Claim", "JudgeVerdict"):
        from mad import models
        return getattr(models, name)
    raise AttributeError(f"module 'guardrails_enterprise.mad' has no attribute {name!r}")


__all__ = ["MADPipeline", "run_mad", "MADOutput", "Claim", "JudgeVerdict"]
