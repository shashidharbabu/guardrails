"""Synthetic SQLite DB for feedback loop tests."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def create_minimal_mad_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE queries (
            query_id TEXT NOT NULL,
            rollout_id TEXT NOT NULL,
            query_text TEXT NOT NULL,
            llm_answer TEXT NOT NULL,
            rag_chunk_ids TEXT,
            timestamp TEXT NOT NULL,
            final_cse_score REAL,
            routing_decision TEXT,
            PRIMARY KEY (query_id, rollout_id)
        );
        CREATE TABLE claims (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT NOT NULL,
            rollout_id TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            claim_text TEXT NOT NULL,
            is_material INTEGER NOT NULL,
            confidence_p REAL NOT NULL,
            verdict TEXT NOT NULL,
            checkpoint TEXT NOT NULL,
            agent_a_prompt TEXT NOT NULL,
            evidence_chunks TEXT,
            reasoning TEXT,
            timestamp TEXT NOT NULL
        );
        CREATE TABLE attacks (
            attack_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT NOT NULL,
            rollout_id TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            b_critique_text TEXT NOT NULL,
            b_challenge_type TEXT NOT NULL,
            p_before_attack REAL NOT NULL,
            p_after_attack REAL,
            agent_b_prompt TEXT,
            timestamp TEXT NOT NULL,
            b_reward REAL
        );
        CREATE TABLE judge_verdicts (
            query_id TEXT NOT NULL,
            rollout_id TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            v_label REAL NOT NULL,
            judge_reasoning TEXT,
            evidence_chunk_ids TEXT,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (query_id, rollout_id, claim_id)
        );
        """
    )


def seed_two_rollouts(con: sqlite3.Connection) -> None:
    """Same query_id, two rollouts — for GRPO advantage mean subtraction."""
    qid = "q-test"
    for rid, p_final, v, p_before, p_after in (
        ("r1", 0.8, 1.0, 0.9, 0.7),  # B helped A drop on correct claim -> gaslight path if drop>=0.2
        ("r2", 0.5, 0.5, 0.6, 0.4),
    ):
        con.execute(
            """INSERT INTO queries VALUES (?,?,?,?,?,?,?,?)""",
            (qid, rid, "query text", "answer", "[]", "ts", None, None),
        )
        con.execute(
            """INSERT INTO claims VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                qid,
                rid,
                "1",
                "HIPAA 45 CFR requires safeguards.",
                1,
                p_final,
                "SUPPORTED",
                "post_cycle2",
                f"PROMPT for {rid}",
                "[]",
                "reason",
                "ts",
            ),
        )
        con.execute(
            """INSERT INTO judge_verdicts VALUES (?,?,?,?,?,?,?)""",
            (qid, rid, "1", v, "judge", "[]", "ts"),
        )
        # One attack: drop p_before -> p_after
        con.execute(
            """INSERT INTO attacks (query_id, rollout_id, claim_id, cycle,
                b_critique_text, b_challenge_type, p_before_attack, p_after_attack,
                agent_b_prompt, timestamp, b_reward)
                VALUES (?,?,?,?,?,?,?,?,?,?,NULL)""",
            (
                qid,
                rid,
                "1",
                1,
                "critique",
                "GAP_FINDING",
                p_before,
                p_after,
                None,
                "ts",
            ),
        )


def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_mad.db")
