"""Environment-driven settings for the feedback loop (no MAD package import)."""
from __future__ import annotations

import os
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
# Default matches multi_agent_debate/multi_agent/config.py (DB under multi_agent_debate/)
_default_mad_db = _repo_root / "multi_agent_debate" / "mad_store.db"


def get_db_path() -> str:
    return os.getenv("MAD_DB_PATH", str(_default_mad_db))


def human_feedback_log_path() -> str:
    return os.getenv(
        "HUMAN_FEEDBACK_LOG_PATH",
        str(_repo_root / "human_feedback_log.jsonl"),
    )


def triage_ambiguous_low() -> float:
    return float(os.getenv("FEEDBACK_TRIAGE_LOW", "-0.10"))


def triage_ambiguous_high() -> float:
    return float(os.getenv("FEEDBACK_TRIAGE_HIGH", "0.30"))


def use_presidio() -> bool:
    return os.getenv("FEEDBACK_USE_PRESIDIO", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def feedback_api_host() -> str:
    return os.getenv("FEEDBACK_API_HOST", "0.0.0.0")


def feedback_api_port() -> int:
    return int(os.getenv("FEEDBACK_API_PORT", "8002"))
