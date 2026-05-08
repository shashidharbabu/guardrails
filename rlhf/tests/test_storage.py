from __future__ import annotations

import sqlite3

import pytest

from rlhf.feedback_loop.db import init_feedback_schema
from rlhf.feedback_loop.scorer import run_batch_scoring
from rlhf.tests.conftest import create_minimal_mad_schema, seed_two_rollouts, tmp_db_path


def test_full_scoring_two_rollouts(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_USE_PRESIDIO", "0")
    db = tmp_db_path(tmp_path)
    con = sqlite3.connect(db)
    create_minimal_mad_schema(con)
    seed_two_rollouts(con)
    con.commit()
    con.close()

    init_feedback_schema(db)
    stats = run_batch_scoring(db, apply_advantage=True)
    assert stats["attacks_updated"] == 2
    assert stats["rewards_upserted"] == 2

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM rewards ORDER BY rollout_id").fetchall()
    assert len(rows) == 2
    r1 = next(r for r in rows if r["rollout_id"] == "r1")
    r2 = next(r for r in rows if r["rollout_id"] == "r2")
    assert r1["is_clean"] == 0  # gaslighting on r1
    assert r2["is_clean"] == 1
    assert r1["grpo_advantage"] is not None and r2["grpo_advantage"] is not None
    assert r1["grpo_advantage"] != r2["grpo_advantage"]
    con.close()


def test_single_clean_rollout(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_USE_PRESIDIO", "0")
    db = tmp_db_path(tmp_path)
    con = sqlite3.connect(db)
    create_minimal_mad_schema(con)
    qid, rid = "q1", "r1"
    con.execute(
        """INSERT INTO queries VALUES (?,?,?,?,?,?,?,?)""",
        (qid, rid, "q", "a", "[]", "ts", None, None),
    )
    con.execute(
        """INSERT INTO claims VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            qid,
            rid,
            "1",
            "HIPAA Security Rule 45 CFR addresses access.",
            1,
            0.9,
            "SUPPORTED",
            "post_cycle2",
            "full prompt",
            "[]",
            "r",
            "ts",
        ),
    )
    con.execute(
        """INSERT INTO judge_verdicts VALUES (?,?,?,?,?,?,?)""",
        (qid, rid, "1", 1.0, "ok", "[]", "ts"),
    )
    con.execute(
        """INSERT INTO attacks (query_id, rollout_id, claim_id, cycle,
            b_critique_text, b_challenge_type, p_before_attack, p_after_attack,
            agent_b_prompt, timestamp, b_reward)
            VALUES (?,?,?,?,?,?,?,?,?,?,NULL)""",
        (qid, rid, "1", 1, "c", "GAP_FINDING", 0.9, 0.88, None, "ts"),
    )
    con.commit()
    con.close()

    run_batch_scoring(db, apply_advantage=True)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM rewards WHERE rollout_id=?", (rid,)).fetchone()
    assert row["is_clean"] == 1
    brier = 2 * 0.9 * 1.0 - 0.9**2
    assert pytest.approx(row["brier_reward"], rel=1e-5) == brier
    assert pytest.approx(row["grpo_advantage"], abs=1e-5) == 0.0
    con.close()
