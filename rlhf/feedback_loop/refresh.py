"""Recompute grpo_advantage only (after human edits, without rescoring rewards)."""
from __future__ import annotations

from rlhf.feedback_loop.advantage import compute_grpo_advantages
from rlhf.feedback_loop.config import get_db_path
from rlhf.feedback_loop.db import connect


def refresh_advantages(db_path: str | None = None) -> int:
    with connect(db_path or get_db_path()) as con:
        return compute_grpo_advantages(con)
