from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from rlhf.feedback_loop.db import init_feedback_schema
from rlhf.feedback_loop.scorer import run_batch_scoring
from rlhf.tests.conftest import create_minimal_mad_schema, tmp_db_path, _uid


def _seed_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_USE_PRESIDIO", "0")
    monkeypatch.setenv("MAD_DB_PATH", tmp_db_path(tmp_path))
    db = tmp_db_path(tmp_path)
    con = sqlite3.connect(db)
    create_minimal_mad_schema(con)

    query_id = "q-amb"
    run_id = "r-amb"
    claim_id = "c-amb"

    con.execute(
        "INSERT INTO queries (query_id, run_id, user_query, baseline_answer) VALUES (?,?,?,?)",
        (query_id, run_id, "question", "answer"),
    )
    # p=0.5, v=0.5 -> brier=0.25, small bonuses -> auto_reward lands in ambiguous band [-0.10, 0.30]
    con.execute(
        """INSERT INTO claims
           (claim_id, query_id, claim_text, claim_index, is_material, is_critical,
            confidence_prior, coverage_check, coverage_ratio)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (claim_id, query_id, "Some regulatory point without CFR cite.", 0, 1, 0, 0.5, 0, 0.0),
    )
    con.execute(
        """INSERT INTO agent_outputs
           (output_id, claim_id, agent_role, round_num, verdict,
            reasoning, confidence_internal)
           VALUES (?,?,?,?,?,?,?)""",
        (_uid(), claim_id, "agent_a", 1, "PARTIAL", "reasoning text", 0.5),
    )
    # Agent B small drop (0.5->0.48, delta=-0.02): helpful but not strong enough -> not gaslighting
    con.execute(
        """INSERT INTO agent_deltas
           (delta_id, claim_id, agent_role,
            confidence_r0, confidence_r1, delta,
            verdict_r0, verdict_r1, verdict_changed)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (_uid(), claim_id, "agent_b", 0.5, 0.48, -0.02, "PARTIAL", "PARTIAL", 0),
    )
    con.execute(
        """INSERT INTO judge_verdicts
           (verdict_id, claim_id, v_label, judge_confidence, judge_reasoning)
           VALUES (?,?,?,?,?)""",
        (_uid(), claim_id, 0.5, 0.7, "partial"),
    )
    con.commit()
    con.close()

    init_feedback_schema(db)
    run_batch_scoring(db, apply_advantage=True)
    return db, query_id, run_id


def test_review_next_and_submit(tmp_path, monkeypatch):
    _seed_ambiguous(tmp_path, monkeypatch)
    monkeypatch.setenv("HUMAN_FEEDBACK_LOG_PATH", str(tmp_path / "audit.jsonl"))

    from rlhf.feedback_loop import api as api_module

    client = TestClient(api_module.app)
    nxt = client.get("/review/next").json()
    assert nxt.get("pending") is True
    qid, rid, cid = nxt["query_id"], nxt["rollout_id"], nxt["claim_id"]
    r = client.post(
        f"/review/{qid}/{rid}/{cid}",
        json={"decision": "good"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["final_reward"] == 0.80
    again = client.get("/review/next").json()
    assert again.get("pending") is False
