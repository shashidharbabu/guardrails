"""SQLite access for feedback loop: same file as MAD (`MAD_DB_PATH`)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from rlhf.feedback_loop.config import get_db_path


@contextmanager
def connect(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or get_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_feedback_schema(db_path: str | None = None) -> None:
    """
    Create the `rewards` table if missing (idempotent).
    MAD owns queries/claims/attacks/judge_verdicts; feedback loop owns `rewards`.
    """
    with connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS rewards (
                query_id                 TEXT NOT NULL,
                rollout_id               TEXT NOT NULL,
                claim_id                   TEXT NOT NULL,
                p_final                    REAL NOT NULL,
                v_label                    REAL NOT NULL,
                brier_reward               REAL NOT NULL,
                verdict_bonus              REAL NOT NULL DEFAULT 0,
                citation_bonus             REAL NOT NULL DEFAULT 0,
                phi_penalty                REAL NOT NULL DEFAULT 0,
                overconfidence_penalty     REAL NOT NULL DEFAULT 0,
                format_penalty             REAL NOT NULL DEFAULT 0,
                auto_reward                REAL NOT NULL,
                human_reward               REAL,
                human_reviewed             INTEGER NOT NULL DEFAULT 0,
                final_reward               REAL,
                is_clean                   INTEGER NOT NULL,
                is_material                INTEGER NOT NULL DEFAULT 1,
                grpo_advantage             REAL,
                scored_at                  TEXT NOT NULL,
                PRIMARY KEY (query_id, rollout_id, claim_id)
            )
            """
        )
