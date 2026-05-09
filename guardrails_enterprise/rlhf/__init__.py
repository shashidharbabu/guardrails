from __future__ import annotations


def __getattr__(name: str):
    if name == "feedback_api":
        from rlhf.feedback_loop import api as feedback_api
        return feedback_api
    raise AttributeError(f"module 'guardrails_enterprise.rlhf' has no attribute {name!r}")


__all__ = ["feedback_api"]
