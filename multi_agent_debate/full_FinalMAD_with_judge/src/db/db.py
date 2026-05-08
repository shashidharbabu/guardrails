from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import uuid4

from src.schemas.schemas import AgentOutputFull, AgentRole, Claim, JudgeOutput


DEFAULT_DB_PATH = "mad.db"


@contextmanager
def connect(db_path: str | Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    schema_path = Path(__file__).parent / "schema.sql"
    schema = schema_path.read_text()
    with connect(db_path) as conn:
        conn.executescript(schema)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def execute(
    sql: str,
    params: Sequence[Any] = (),
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    with connect(db_path) as conn:
        conn.execute(sql, params)


def fetch_all(
    sql: str,
    params: Sequence[Any] = (),
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def fetch_one(
    sql: str,
    params: Sequence[Any] = (),
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def insert_query(
    *,
    query_id: str,
    run_id: str,
    user_query: str,
    rag_chunk_ids: list[str],
    rag_chunks: list[dict[str, Any]],
    baseline_answer: str,
    baseline_model: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    execute(
        """
        INSERT OR IGNORE INTO queries (
            query_id, run_id, user_query, rag_chunk_ids, rag_chunks,
            baseline_answer, baseline_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query_id,
            run_id,
            user_query,
            _json(rag_chunk_ids),
            _json(rag_chunks),
            baseline_answer,
            baseline_model,
        ),
        db_path,
    )


def insert_claim(
    conn: sqlite3.Connection,
    claim: Claim,
    query_id: str,
    coverage_check: bool,
    coverage_ratio: float,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO claims (
            claim_id, query_id, claim_text, claim_index, is_material,
            is_critical, confidence_prior, coverage_check, coverage_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            claim.claim_id,
            query_id,
            claim.claim_text,
            claim.claim_index,
            claim.is_material,
            claim.is_critical,
            claim.confidence_prior,
            coverage_check,
            coverage_ratio,
        ),
    )


def insert_agent_output(
    conn: sqlite3.Connection,
    output: AgentOutputFull,
    claim_id: str,
    raw_response: str,
    latency_ms: int,
    tokens_in: int,
    tokens_out: int,
) -> None:
    conn.execute(
        """
        INSERT INTO agent_outputs (
            output_id, claim_id, agent_role, round_num, verdict, reasoning,
            evidence_cited, confidence_internal, raw_response, latency_ms,
            tokens_in, tokens_out
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            claim_id,
            output.agent_role.value,
            output.round_num,
            output.verdict.value,
            output.reasoning,
            _json([e.model_dump() for e in output.evidence_cited]),
            output.confidence_internal,
            raw_response,
            latency_ms,
            tokens_in,
            tokens_out,
        ),
    )


def insert_agent_delta(
    conn: sqlite3.Connection,
    claim_id: str,
    agent_role: AgentRole,
    r0: AgentOutputFull,
    r1: AgentOutputFull,
) -> None:
    conn.execute(
        """
        INSERT INTO agent_deltas (
            delta_id, claim_id, agent_role, confidence_r0, confidence_r1,
            delta, verdict_r0, verdict_r1, verdict_changed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            claim_id,
            agent_role.value,
            r0.confidence_internal,
            r1.confidence_internal,
            r1.confidence_internal - r0.confidence_internal,
            r0.verdict.value,
            r1.verdict.value,
            r0.verdict != r1.verdict,
        ),
    )


def insert_judge_verdict(
    conn: sqlite3.Connection,
    verdict: JudgeOutput,
    claim_id: str,
    judge_model: str,
    raw_response: str,
    latency_ms: int,
) -> None:
    conn.execute(
        """
        INSERT INTO judge_verdicts (
            verdict_id, claim_id, v_label, judge_confidence, judge_reasoning,
            evidence_chunk_ids, judge_model, raw_response, latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            claim_id,
            float(verdict.v_label.value),
            verdict.judge_confidence,
            verdict.judge_reasoning,
            _json(verdict.evidence_chunk_ids),
            judge_model,
            raw_response,
            latency_ms,
        ),
    )


def get_claims_for_query(
    query_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    return fetch_all(
        "SELECT * FROM claims WHERE query_id = ? ORDER BY claim_index",
        (query_id,),
        db_path,
    )


def get_agent_outputs_for_claim(
    claim_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT * FROM agent_outputs
        WHERE claim_id = ?
        ORDER BY round_num, agent_role
        """,
        (claim_id,),
        db_path,
    )
