"""Synthetic SQLite DB for feedback loop tests — matches full_FinalMAD_with_judge schema."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path


def create_minimal_mad_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE queries (
            query_id        TEXT PRIMARY KEY,
            run_id          TEXT NOT NULL,
            user_query      TEXT NOT NULL,
            rag_chunk_ids   TEXT NOT NULL DEFAULT '[]',
            rag_chunks      TEXT NOT NULL DEFAULT '[]',
            baseline_answer TEXT NOT NULL DEFAULT '',
            baseline_model  TEXT NOT NULL DEFAULT 'test-model',
            timestamp       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE claims (
            claim_id         TEXT PRIMARY KEY,
            query_id         TEXT NOT NULL,
            claim_text       TEXT NOT NULL,
            claim_index      INTEGER NOT NULL DEFAULT 0,
            is_material      INTEGER NOT NULL DEFAULT 1,
            is_critical      INTEGER NOT NULL DEFAULT 0,
            confidence_prior REAL NOT NULL DEFAULT 0.5,
            coverage_check   INTEGER NOT NULL DEFAULT 0,
            coverage_ratio   REAL NOT NULL DEFAULT 0.0,
            FOREIGN KEY (query_id) REFERENCES queries(query_id)
        );

        CREATE TABLE agent_outputs (
            output_id           TEXT PRIMARY KEY,
            claim_id            TEXT NOT NULL,
            agent_role          TEXT NOT NULL,
            round_num           INTEGER NOT NULL,
            verdict             TEXT NOT NULL,
            reasoning           TEXT NOT NULL DEFAULT '',
            evidence_cited      TEXT NOT NULL DEFAULT '[]',
            confidence_internal REAL NOT NULL,
            raw_response        TEXT NOT NULL DEFAULT '',
            latency_ms          INTEGER NOT NULL DEFAULT 0,
            tokens_in           INTEGER NOT NULL DEFAULT 0,
            tokens_out          INTEGER NOT NULL DEFAULT 0,
            timestamp           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
        );

        CREATE TABLE agent_deltas (
            delta_id        TEXT PRIMARY KEY,
            claim_id        TEXT NOT NULL,
            agent_role      TEXT NOT NULL,
            confidence_r0   REAL NOT NULL,
            confidence_r1   REAL NOT NULL,
            delta           REAL NOT NULL,
            verdict_r0      TEXT NOT NULL,
            verdict_r1      TEXT NOT NULL,
            verdict_changed INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
        );

        CREATE TABLE judge_verdicts (
            verdict_id         TEXT PRIMARY KEY,
            claim_id           TEXT NOT NULL,
            v_label            REAL NOT NULL,
            judge_confidence   REAL NOT NULL DEFAULT 1.0,
            judge_reasoning    TEXT NOT NULL DEFAULT '',
            evidence_chunk_ids TEXT NOT NULL DEFAULT '[]',
            judge_model        TEXT NOT NULL DEFAULT 'test-judge',
            raw_response       TEXT NOT NULL DEFAULT '',
            latency_ms         INTEGER NOT NULL DEFAULT 0,
            timestamp          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
        );
        """
    )


def _uid() -> str:
    return str(uuid.uuid4())


def seed_two_rollouts(con: sqlite3.Connection) -> None:
    """
    Same query_id, two different run_ids (= rollout_ids for GRPO mean subtraction).

    Both queries share query_id="q-test" so GRPO groups them together.
    r1: p_final=0.8, v_label=1.0, agent_b drops A by 0.2 on supported claim -> gaslighting
    r2: p_final=0.5, v_label=0.5, agent_b drops A by 0.2 on partial claim  -> helpful
    """
    shared_query_id = "q-test"
    for run_id, p_final, v_label, p_b_r0, p_b_r1, b_verdict in (
        ("r1", 0.8, 1.0, 0.9, 0.7, "SUPPORTED"),  # B gaslights: drop on supported claim
        ("r2", 0.5, 0.5, 0.6, 0.4, "PARTIAL"),    # B helps: drop on partial claim
    ):
        # Each rollout gets its own query row (queries.query_id must be unique PK)
        # but uses the SAME shared_query_id as claims.query_id so GRPO groups by it.
        # We use run_id as a prefix to make query_id unique.
        query_id = f"{shared_query_id}-{run_id}"
        claim_id = f"c-{run_id}"

        con.execute(
            "INSERT INTO queries (query_id, run_id, user_query, baseline_answer) VALUES (?,?,?,?)",
            (query_id, run_id, "What does HIPAA require?", "answer text"),
        )
        con.execute(
            """INSERT INTO claims
               (claim_id, query_id, claim_text, claim_index, is_material, is_critical,
                confidence_prior, coverage_check, coverage_ratio)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (claim_id, query_id, "HIPAA 45 CFR requires safeguards.", 0, 1, 0, 0.5, 0, 0.0),
        )
        # Agent A round=0 output
        con.execute(
            """INSERT INTO agent_outputs
               (output_id, claim_id, agent_role, round_num, verdict,
                reasoning, confidence_internal)
               VALUES (?,?,?,?,?,?,?)""",
            (_uid(), claim_id, "agent_a", 0, "SUPPORTED",
             f"Round 0 reasoning for {run_id}", p_b_r0),
        )
        # Agent A round=1 output — this is p_final used by scorer
        con.execute(
            """INSERT INTO agent_outputs
               (output_id, claim_id, agent_role, round_num, verdict,
                reasoning, confidence_internal)
               VALUES (?,?,?,?,?,?,?)""",
            (_uid(), claim_id, "agent_a", 1, "SUPPORTED",
             f"Round 1 reasoning for {run_id}", p_final),
        )
        # Agent B delta — confidence shift
        delta = p_b_r1 - p_b_r0   # negative = A dropped confidence after B's challenge
        con.execute(
            """INSERT INTO agent_deltas
               (delta_id, claim_id, agent_role,
                confidence_r0, confidence_r1, delta,
                verdict_r0, verdict_r1, verdict_changed)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (_uid(), claim_id, "agent_b",
             p_b_r0, p_b_r1, delta,
             b_verdict, b_verdict, 0),
        )
        # Judge verdict
        con.execute(
            """INSERT INTO judge_verdicts
               (verdict_id, claim_id, v_label, judge_confidence, judge_reasoning)
               VALUES (?,?,?,?,?)""",
            (_uid(), claim_id, v_label, 0.9, "judge reasoning"),
        )


def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_mad.db")
