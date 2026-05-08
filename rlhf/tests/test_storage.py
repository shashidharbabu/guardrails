from __future__ import annotations

import sqlite3
import uuid

import pytest

from rlhf.feedback_loop.db import init_feedback_schema
from rlhf.feedback_loop.scorer import run_batch_scoring
from rlhf.tests.conftest import (
    create_minimal_mad_schema,
    seed_two_rollouts,
    tmp_db_path,
    _uid,
)


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
    # Two agent_b delta rows scanned
    assert stats["attacks_updated"] == 2
    # Two reward rows written (one per claim/rollout)
    assert stats["rewards_upserted"] == 2

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM rewards ORDER BY rollout_id").fetchall()
    assert len(rows) == 2
    r1 = next(r for r in rows if r["rollout_id"] == "r1")
    r2 = next(r for r in rows if r["rollout_id"] == "r2")
    # r1: agent_b dropped confidence by 0.2 on v_label=1.0 -> gaslighting -> is_clean=0
    assert r1["is_clean"] == 0
    # r2: drop on partial claim (v_label=0.5) -> helpful, not gaslighting -> is_clean=1
    assert r2["is_clean"] == 1
    assert r1["grpo_advantage"] is not None and r2["grpo_advantage"] is not None
    # Two different rollouts under same logical query -> different advantages
    assert r1["grpo_advantage"] != r2["grpo_advantage"]
    con.close()


def test_single_clean_rollout(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_USE_PRESIDIO", "0")
    db = tmp_db_path(tmp_path)
    con = sqlite3.connect(db)
    create_minimal_mad_schema(con)

    query_id = "q1"
    run_id = "r1"
    claim_id = "c1"

    con.execute(
        "INSERT INTO queries (query_id, run_id, user_query, baseline_answer) VALUES (?,?,?,?)",
        (query_id, run_id, "What does HIPAA require?", "answer"),
    )
    con.execute(
        """INSERT INTO claims
           (claim_id, query_id, claim_text, claim_index, is_material, is_critical,
            confidence_prior, coverage_check, coverage_ratio)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (claim_id, query_id, "HIPAA Security Rule 45 CFR addresses access.", 0, 1, 0, 0.9, 0, 0.0),
    )
    # Agent A round=1 — p_final=0.9
    con.execute(
        """INSERT INTO agent_outputs
           (output_id, claim_id, agent_role, round_num, verdict,
            reasoning, confidence_internal)
           VALUES (?,?,?,?,?,?,?)""",
        (_uid(), claim_id, "agent_a", 1, "SUPPORTED", "full prompt reasoning", 0.9),
    )
    # Agent B delta — small drop (0.9 -> 0.88, delta=-0.02): NOT gaslighting
    con.execute(
        """INSERT INTO agent_deltas
           (delta_id, claim_id, agent_role,
            confidence_r0, confidence_r1, delta,
            verdict_r0, verdict_r1, verdict_changed)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (_uid(), claim_id, "agent_b", 0.9, 0.88, -0.02, "SUPPORTED", "SUPPORTED", 0),
    )
    # Judge: fully supported
    con.execute(
        """INSERT INTO judge_verdicts
           (verdict_id, claim_id, v_label, judge_confidence, judge_reasoning)
           VALUES (?,?,?,?,?)""",
        (_uid(), claim_id, 1.0, 0.95, "ok"),
    )
    con.commit()
    con.close()

    run_batch_scoring(db, apply_advantage=True)

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM rewards WHERE rollout_id=?", (run_id,)).fetchone()
    assert row["is_clean"] == 1
    brier = 2 * 0.9 * 1.0 - 0.9**2
    assert pytest.approx(row["brier_reward"], rel=1e-5) == brier
    # Single rollout -> advantage should be 0
    assert pytest.approx(row["grpo_advantage"], abs=1e-5) == 0.0
    con.close()
