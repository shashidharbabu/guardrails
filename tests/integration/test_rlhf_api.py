"""Integration tests for the RLHF Feedback Loop FastAPI service."""
from __future__ import annotations

import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

from rlhf.feedback_loop.db import init_feedback_schema
from rlhf.feedback_loop.scorer import run_batch_scoring
from rlhf.tests.conftest import create_minimal_mad_schema, tmp_db_path, _uid


def _seed_ambiguous_item(db: str, monkeypatch) -> tuple:
    """
    Seed one claim that lands in the ambiguous human review band [-0.10, 0.30].

    Use p=0.5, v=0.5 with a non-regulatory plain-text claim so the heuristic
    citation_bonus (+0.15) is NOT triggered.
    brier = 2*0.5*0.5 - 0.5^2 = 0.25 → auto_reward = 0.25 ∈ [-0.10, 0.30]
    """
    monkeypatch.setenv("MAD_DB_PATH", db)
    monkeypatch.setenv("FEEDBACK_USE_PRESIDIO", "0")
    monkeypatch.setenv("FEEDBACK_TRIAGE_LOW", "-0.10")
    monkeypatch.setenv("FEEDBACK_TRIAGE_HIGH", "0.30")

    query_id = "q-int-test"
    run_id = "r-int-test"
    claim_id = "c-int-test"

    con = sqlite3.connect(db)
    create_minimal_mad_schema(con)
    # Use a neutral query/claim — no "HIPAA"/"CFR" to avoid citation_bonus
    con.execute(
        "INSERT INTO queries (query_id, run_id, user_query, baseline_answer) VALUES (?,?,?,?)",
        (query_id, run_id, "What are the general requirements?", "answer text"),
    )
    con.execute(
        """INSERT INTO claims
           (claim_id, query_id, claim_text, claim_index, is_material, is_critical,
            confidence_prior, coverage_check, coverage_ratio)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (claim_id, query_id, "The system requires written policies.", 0, 1, 0, 0.5, 0, 0.0),
    )
    # p_final=0.5, v_label=0.5 → brier=0.25, no citation bonus → auto_reward=0.25 ∈ band
    con.execute(
        """INSERT INTO agent_outputs
           (output_id, claim_id, agent_role, round_num, verdict,
            reasoning, confidence_internal)
           VALUES (?,?,?,?,?,?,?)""",
        (_uid(), claim_id, "agent_a", 1, "PARTIAL", "partial evidence found", 0.5),
    )
    # Agent B small drop — not gaslighting (v=0.5, not supported)
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
        (_uid(), claim_id, 0.5, 0.7, "partial support found"),
    )
    con.commit()
    con.close()

    init_feedback_schema(db)
    run_batch_scoring(db, apply_advantage=True)

    # Verify auto_reward landed in band before returning
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT auto_reward FROM rewards LIMIT 1").fetchone()
    con.close()
    assert row is not None, "Scoring produced no reward rows"
    assert -0.10 <= row["auto_reward"] <= 0.30, (
        f"auto_reward={row['auto_reward']} not in ambiguous band — fix seed"
    )

    return query_id, run_id, claim_id


@pytest.fixture
def feedback_client(tmp_path, monkeypatch):
    db = tmp_db_path(tmp_path)
    monkeypatch.setenv("HUMAN_FEEDBACK_LOG_PATH", str(tmp_path / "audit.jsonl"))
    _seed_ambiguous_item(db, monkeypatch)

    from rlhf.feedback_loop import api as api_mod
    client = TestClient(api_mod.app)
    return client


class TestFeedbackHealth:
    def test_health_ok(self, feedback_client):
        r = feedback_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestReviewQueue:
    def test_review_next_pending(self, feedback_client):
        r = feedback_client.get("/review/next")
        assert r.status_code == 200
        body = r.json()
        assert body["pending"] is True
        assert "auto_reward" in body
        assert "triage_band" in body
        assert "prompt" in body

    def test_submit_good_sets_final_reward_080(self, feedback_client):
        nxt = feedback_client.get("/review/next").json()
        assert nxt["pending"] is True
        qid, rid, cid = nxt["query_id"], nxt["rollout_id"], nxt["claim_id"]
        r = feedback_client.post(f"/review/{qid}/{rid}/{cid}", json={"decision": "good"})
        assert r.status_code == 200
        assert r.json()["final_reward"] == pytest.approx(0.80)

    def test_submit_skip_keeps_auto_reward(self, feedback_client):
        # Re-score to get a fresh pending item (previous test may have consumed it)
        feedback_client.post("/rewards/compute")
        nxt = feedback_client.get("/review/next").json()
        if not nxt.get("pending"):
            pytest.skip("no pending items left after previous submit")
        qid, rid, cid = nxt["query_id"], nxt["rollout_id"], nxt["claim_id"]
        auto = nxt["auto_reward"]
        r = feedback_client.post(f"/review/{qid}/{rid}/{cid}", json={"decision": "skip"})
        assert r.status_code == 200
        assert r.json()["final_reward"] == pytest.approx(auto)

    def test_queue_empty_after_all_reviewed(self, feedback_client):
        # Drain queue
        while True:
            nxt = feedback_client.get("/review/next").json()
            if not nxt.get("pending"):
                break
            qid, rid, cid = nxt["query_id"], nxt["rollout_id"], nxt["claim_id"]
            feedback_client.post(f"/review/{qid}/{rid}/{cid}", json={"decision": "skip"})
        again = feedback_client.get("/review/next").json()
        assert again["pending"] is False

    def test_double_submit_returns_409(self, feedback_client):
        """Submitting a review twice returns 409 Conflict."""
        # Re-seed by triggering compute (idempotent on already-reviewed rows)
        feedback_client.post("/rewards/compute")
        nxt = feedback_client.get("/review/next").json()
        if not nxt.get("pending"):
            pytest.skip("no pending items to test double-submit")
        qid, rid, cid = nxt["query_id"], nxt["rollout_id"], nxt["claim_id"]
        feedback_client.post(f"/review/{qid}/{rid}/{cid}", json={"decision": "good"})
        r2 = feedback_client.post(f"/review/{qid}/{rid}/{cid}", json={"decision": "good"})
        assert r2.status_code == 409


class TestReviewLog:
    def test_log_endpoint_returns_entries_list(self, feedback_client):
        # Drain and submit so log has at least one entry
        nxt = feedback_client.get("/review/next").json()
        if nxt.get("pending"):
            qid, rid, cid = nxt["query_id"], nxt["rollout_id"], nxt["claim_id"]
            feedback_client.post(f"/review/{qid}/{rid}/{cid}", json={"decision": "good"})
        r = feedback_client.get("/review/log?tail=10")
        assert r.status_code == 200
        assert "entries" in r.json()
        assert isinstance(r.json()["entries"], list)


class TestBatchScoring:
    def test_compute_rewards_returns_stats(self, feedback_client):
        r = feedback_client.post("/rewards/compute")
        assert r.status_code == 200
        body = r.json()
        assert "rewards_upserted" in body
        assert "rollouts_advantaged" in body
        assert "attacks_updated" in body
