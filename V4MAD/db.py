import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from config import DB_PATH

_conn: Optional[sqlite3.Connection] = None


def get_db_conn(path: str = None) -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(path or str(DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.row_factory = sqlite3.Row
    return _conn


def init_db(path: str = None) -> None:
    conn = get_db_conn(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS queries (
            query_id          TEXT PRIMARY KEY,
            run_id            TEXT,
            user_query        TEXT,
            baseline_answer   TEXT,
            baseline_model    TEXT,
            tokens_in         INTEGER DEFAULT 0,
            tokens_out        INTEGER DEFAULT 0,
            latency_ms        INTEGER DEFAULT 0,
            timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS query_chunks (
            query_id  TEXT PRIMARY KEY,
            chunks    TEXT,
            FOREIGN KEY (query_id) REFERENCES queries(query_id)
        );

        CREATE TABLE IF NOT EXISTS claims (
            claim_id         TEXT PRIMARY KEY,
            query_id         TEXT,
            claim_text       TEXT,
            claim_index      INTEGER,
            is_material      INTEGER DEFAULT 1,
            is_critical      INTEGER DEFAULT 0,
            confidence_prior REAL DEFAULT 0.75,
            coverage_check   INTEGER DEFAULT 1,
            coverage_ratio   REAL DEFAULT 1.0,
            timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES queries(query_id)
        );

        CREATE TABLE IF NOT EXISTS claim_chunks (
            claim_id        TEXT PRIMARY KEY,
            agent_a_chunks  TEXT,
            agent_b_chunks  TEXT,
            judge_chunks    TEXT,
            FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
        );

        CREATE TABLE IF NOT EXISTS agent_outputs (
            output_id           TEXT PRIMARY KEY,
            claim_id            TEXT,
            agent_role          TEXT,
            round_num           INTEGER,
            verdict             TEXT,
            reasoning           TEXT,
            evidence_cited      TEXT,
            confidence_internal REAL,
            raw_response        TEXT,
            tokens_in           INTEGER DEFAULT 0,
            tokens_out          INTEGER DEFAULT 0,
            latency_ms          INTEGER DEFAULT 0,
            from_cache          INTEGER DEFAULT 0,
            timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
        );

        CREATE TABLE IF NOT EXISTS judge_verdicts (
            verdict_id         TEXT PRIMARY KEY,
            claim_id           TEXT,
            v_label            REAL,
            judge_confidence   REAL,
            judge_reasoning    TEXT,
            evidence_chunk_ids TEXT,
            judge_model        TEXT,
            raw_response       TEXT,
            tokens_in          INTEGER DEFAULT 0,
            tokens_out         INTEGER DEFAULT 0,
            latency_ms         INTEGER DEFAULT 0,
            from_cache         INTEGER DEFAULT 0,
            timestamp          DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
        );

        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key  TEXT PRIMARY KEY,
            response   TEXT,
            model      TEXT,
            timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_claims_query   ON claims(query_id);
        CREATE INDEX IF NOT EXISTS idx_outputs_claim  ON agent_outputs(claim_id);
        CREATE INDEX IF NOT EXISTS idx_outputs_role   ON agent_outputs(agent_role, round_num);
        CREATE INDEX IF NOT EXISTS idx_verdicts_claim ON judge_verdicts(claim_id);
    """)
    conn.commit()


# ── Insert helpers ─────────────────────────────────────────────────────────────

def insert_query(conn, query_id, run_id, user_query, baseline_answer, model, ti, to, latency):
    conn.execute("""
        INSERT OR REPLACE INTO queries
        (query_id, run_id, user_query, baseline_answer, baseline_model, tokens_in, tokens_out, latency_ms)
        VALUES (?,?,?,?,?,?,?,?)
    """, (query_id, run_id, user_query, baseline_answer, model, ti, to, latency))
    conn.commit()


def insert_query_chunks(conn, query_id, chunks: list):
    conn.execute("""
        INSERT OR REPLACE INTO query_chunks (query_id, chunks) VALUES (?,?)
    """, (query_id, json.dumps(chunks)))
    conn.commit()


def insert_claim(conn, claim: dict, query_id: str, coverage_check: bool = True, coverage_ratio: float = 1.0):
    conn.execute("""
        INSERT OR IGNORE INTO claims
        (claim_id, query_id, claim_text, claim_index, is_material, is_critical,
         confidence_prior, coverage_check, coverage_ratio)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        claim["claim_id"], query_id,
        claim["claim_text"], claim.get("claim_index", 0),
        int(claim.get("is_material", True)),
        int(claim.get("is_critical", False)),
        float(claim.get("confidence_prior", 0.75)),
        int(coverage_check), float(coverage_ratio),
    ))
    conn.commit()


def insert_claim_chunks(conn, claim_id: str, assignment: dict):
    conn.execute("""
        INSERT OR REPLACE INTO claim_chunks (claim_id, agent_a_chunks, agent_b_chunks, judge_chunks)
        VALUES (?,?,?,?)
    """, (
        claim_id,
        json.dumps(assignment.get("agent_a", [])),
        json.dumps(assignment.get("agent_b", [])),
        json.dumps(assignment.get("judge",   [])),
    ))
    conn.commit()


def insert_agent_output(conn, claim_id: str, agent_role: str, round_num: int,
                        parsed: dict, raw: str, ti: int, to: int, latency: int, from_cache: bool):
    output_id = f"{claim_id}_{agent_role}_r{round_num}"
    conn.execute("""
        INSERT OR REPLACE INTO agent_outputs
        (output_id, claim_id, agent_role, round_num, verdict, reasoning,
         evidence_cited, confidence_internal, raw_response,
         tokens_in, tokens_out, latency_ms, from_cache)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        output_id, claim_id, agent_role, round_num,
        parsed.get("verdict", "IDK"),
        parsed.get("reasoning", ""),
        json.dumps(parsed.get("evidence_cited", [])),
        float(parsed.get("confidence_internal", 0.5)),
        raw, ti, to, latency, int(from_cache),
    ))
    conn.commit()


def insert_judge_verdict(conn, claim_id: str, parsed: dict, model: str,
                         raw: str, ti: int, to: int, latency: int, from_cache: bool):
    verdict_id = f"judge_{claim_id}"
    conn.execute("""
        INSERT OR REPLACE INTO judge_verdicts
        (verdict_id, claim_id, v_label, judge_confidence, judge_reasoning,
         evidence_chunk_ids, judge_model, raw_response, tokens_in, tokens_out, latency_ms, from_cache)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        verdict_id, claim_id,
        float(parsed.get("v_label", 0.5)),
        float(parsed.get("judge_confidence", 0.5)),
        parsed.get("judge_reasoning", ""),
        json.dumps(parsed.get("evidence_chunk_ids", [])),
        model, raw, ti, to, latency, int(from_cache),
    ))
    conn.commit()


def insert_llm_cache(conn, cache_key: str, response: str, model: str):
    conn.execute("""
        INSERT OR IGNORE INTO llm_cache (cache_key, response, model) VALUES (?,?,?)
    """, (cache_key, response, model))
    conn.commit()


# ── Read helpers ───────────────────────────────────────────────────────────────

def get_llm_cache(conn, cache_key: str) -> Optional[str]:
    row = conn.execute("SELECT response FROM llm_cache WHERE cache_key=?", (cache_key,)).fetchone()
    return row[0] if row else None


def query_has_baseline(conn, query_id: str) -> bool:
    row = conn.execute(
        "SELECT baseline_answer FROM queries WHERE query_id=? AND baseline_answer IS NOT NULL",
        (query_id,)
    ).fetchone()
    return row is not None


def query_has_claims(conn, query_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM claims WHERE query_id=? LIMIT 1", (query_id,)).fetchone()
    return row is not None


def query_has_claim_chunks(conn, query_id: str) -> bool:
    row = conn.execute("""
        SELECT 1 FROM claim_chunks cc
        JOIN claims c ON cc.claim_id = c.claim_id
        WHERE c.query_id=? LIMIT 1
    """, (query_id,)).fetchone()
    return row is not None


def query_has_debate(conn, query_id: str) -> bool:
    row = conn.execute("""
        SELECT 1 FROM agent_outputs ao
        JOIN claims c ON ao.claim_id = c.claim_id
        WHERE c.query_id=? AND ao.round_num=1 LIMIT 1
    """, (query_id,)).fetchone()
    return row is not None


def get_claims_for_query(conn, query_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM claims WHERE query_id=? ORDER BY claim_index", (query_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_claim_chunks(conn, claim_id: str) -> dict:
    row = conn.execute("SELECT * FROM claim_chunks WHERE claim_id=?", (claim_id,)).fetchone()
    if not row:
        return {}
    return {
        "agent_a": json.loads(row["agent_a_chunks"] or "[]"),
        "agent_b": json.loads(row["agent_b_chunks"] or "[]"),
        "judge":   json.loads(row["judge_chunks"]   or "[]"),
    }


def get_agent_output(conn, claim_id: str, agent_role: str, round_num: int) -> Optional[dict]:
    row = conn.execute("""
        SELECT * FROM agent_outputs
        WHERE claim_id=? AND agent_role=? AND round_num=?
    """, (claim_id, agent_role, round_num)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["evidence_cited"] = json.loads(d["evidence_cited"] or "[]")
    return d


def get_query_chunks(conn, query_id: str) -> list[dict]:
    row = conn.execute("SELECT chunks FROM query_chunks WHERE query_id=?", (query_id,)).fetchone()
    return json.loads(row[0]) if row else []
