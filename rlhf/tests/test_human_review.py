from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from rlhf.feedback_loop.db import init_feedback_schema
from rlhf.feedback_loop.scorer import run_batch_scoring
from rlhf.tests.conftest import create_minimal_mad_schema, tmp_db_path


def _seed_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_USE_PRESIDIO", "0")
    monkeypatch.setenv("MAD_DB_PATH", tmp_db_path(tmp_path))
    db = tmp_db_path(tmp_path)
    con = sqlite3.connect(db)
    create_minimal_mad_schema(con)
    qid, rid = "q-amb", "r-amb"
    con.execute(
        """INSERT INTO queries VALUES (?,?,?,?,?,?,?,?)""",
        (qid, rid, "question", "answer", "[]", "ts", None, None),
    )
    # p=0.5 v=0.5 -> brier 0.25 + small bonuses -> likely in ambiguous band
    con.execute(
        """INSERT INTO claims VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            qid,
            rid,
            "1",
            "Some regulatory point without CFR cite.",
            1,
            0.5,
            "PARTIAL",
            "post_cycle2",
            "prompt ctx",
            "[]",
            "reasoning text",
            "ts",
        ),
    )
    con.execute(
        """INSERT INTO judge_verdicts VALUES (?,?,?,?,?,?,?)""",
        (qid, rid, "1", 0.5, "partial", "[]", "ts"),
    )
    con.execute(
        """INSERT INTO attacks (query_id, rollout_id, claim_id, cycle,
            b_critique_text, b_challenge_type, p_before_attack, p_after_attack,
            agent_b_prompt, timestamp, b_reward)
            VALUES (?,?,?,?,?,?,?,?,?,?,NULL)""",
        (qid, rid, "1", 1, "c", "GAP_FINDING", 0.5, 0.48, None, "ts"),
    )
    con.commit()
    con.close()
    init_feedback_schema(db)
    run_batch_scoring(db, apply_advantage=True)
    return db, qid, rid


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
