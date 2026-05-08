"""Integration tests for the RLHF Feedback Loop FastAPI service.

Runs the feedback loop API directly (TestClient), seeds a synthetic DB,
and exercises the full review queue → submit → GRPO advantage cycle.
"""
from __future__ import annotations

import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

from rlhf.feedback_loop.db import init_feedback_schema
from rlhf.feedback_loop.scorer import run_batch_scoring
from rlhf.tests.conftest import create_minimal_mad_schema, tmp_db_path, _uid


def _seed_ambiguous_item(db: str, monkeypatch) -> tuple[str, str, str]:
    """Seed one item that will land in the ambiguous human review band."""
    monkeypatch.setenv("MAD_DB_PATH", db)
    monkeypatch.setenv("FEEDBACK_USE_PRESIDIO", "0")

    query_id = "q-int"
    run_id = "r-int"
    claim_id = "c-int"

    con = sqlite3.connect(db)
    create_minimal_mad_schema(con)
    con.execute(
        "INSERT INTO queries (query_id, run_id, user_query, baseline_answer) VALUES (?,?,?,?)",
        (query_id, run_id, "What does HIPAA require?", "answer"),
    )
    con.execute(
        """INSERT INTO claims
           (claim_id, query_id, claim_text, claim_index, is_material, is_critical,
            confidence_prior, coverage_check, coverage_ratio)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (claim_id, query_id, "HIPAA requires safeguards.", 0, 1, 0, 0.5, 0, 0.0),
    )
    # p_final=0.5, v_label=0.5 → brier=0.25 → auto_reward in ambiguous band
    con.execute(
        """INSERT INTO agent_outputs
           (output_id, claim_id, agent_role, round_num, verdict,
            reasoning, confidence_internal)
           VALUES (?,?,?,?,?,?,?)""",
        (_uid(), claim_id, "agent_a", 1, "PARTIAL", "some reasoning", 0.5),
    )
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
    return query_id, run_id, claim_id


@pytest.fixture
def feedback_client(tmp_path, monkeypatch):
    db = tmp_db_path(tmp_path)
    monkeypatch.setenv("HUMAN_FEEDBACK_LOG_PATH", str(tmp_path / "audit.jsonl"))
    qid, rid, cid = _seed_ambiguous_item(db, monkeypatch)

    from rlhf.feedback_loop import api as api_mod
    client = TestClient(api_mod.app)
    return client, qid, rid, cid


class TestFeedbackHealth:
    def test_health_ok(self, feedback_client):
        client, *_ = feedback_client
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestReviewQueue:
    def test_review_next_pending(self, feedback_client):
        client, qid, rid, cid = feedback_client
        r = client.get("/review/next")
        assert r.status_code == 200
        body = r.json()
        assert body["pending"] is True
        assert "auto_reward" in body
        assert "triage_band" in body

    def test_submit_good_sets_final_reward_080(self, feedback_client):
        client, qid, rid, cid = feedback_client
        nxt = client.get("/review/next").json()
        actual_qid = nxt["query_id"]
        actual_rid = nxt["rollout_id"]
        actual_cid = nxt["claim_id"]

        r = client.post(
            f"/review/{actual_qid}/{actual_rid}/{actual_cid}",
            json={"decision": "good"},
        )
        assert r.status_code == 200
        assert r.json()["final_reward"] == pytest.approx(0.80)

    def test_submit_bad_sets_negative_reward(self, feedback_client):
        client, qid, rid, cid = feedback_client
        nxt = client.get("/review/next").json()
        if not nxt.get("pending"):
            pytest.skip("no pending items (already consumed by earlier test)")
        actual_qid = nxt["query_id"]
        actual_rid = nxt["rollout_id"]
        actual_cid = nxt["claim_id"]
        r = client.post(
            f"/review/{actual_qid}/{actual_rid}/{actual_cid}",
            json={"decision": "bad"},
        )
        assert r.status_code == 200
        assert r.json()["final_reward"] == pytest.approx(-0.60)

    def test_queue_empty_after_review(self, feedback_client):
        client, qid, rid, cid = feedback_client
        nxt = client.get("/review/next").json()
        if nxt.get("pending"):
            aqid, arid, acid = nxt["query_id"], nxt["rollout_id"], nxt["claim_id"]
            client.post(f"/review/{aqid}/{arid}/{acid}", json={"decision": "skip"})
        again = client.get("/review/next").json()
        assert again["pending"] is False

    def test_double_submit_returns_409(self, feedback_client):
        client, qid, rid, cid = feedback_client
        nxt = client.get("/review/next").json()
        if not nxt.get("pending"):
            pytest.skip("no pending items")
        aqid, arid, acid = nxt["query_id"], nxt["rollout_id"], nxt["claim_id"]
        client.post(f"/review/{aqid}/{arid}/{acid}", json={"decision": "good"})
        r2 = client.post(f"/review/{aqid}/{arid}/{acid}", json={"decision": "good"})
        assert r2.status_code == 409


class TestReviewLog:
    def test_log_endpoint_returns_entries(self, feedback_client):
        client, qid, rid, cid = feedback_client
        # Submit first so there's something in the log
        nxt = client.get("/review/next").json()
        if nxt.get("pending"):
            aqid, arid, acid = nxt["query_id"], nxt["rollout_id"], nxt["claim_id"]
            client.post(f"/review/{aqid}/{arid}/{acid}", json={"decision": "good"})
        r = client.get("/review/log?tail=10")
        assert r.status_code == 200
        assert "entries" in r.json()


class TestBatchScoring:
    def test_compute_rewards_returns_stats(self, feedback_client):
        client, *_ = feedback_client
        r = client.post("/rewards/compute")
        assert r.status_code == 200
        body = r.json()
        assert "rewards_upserted" in body
        assert "rollouts_advantaged" in body
