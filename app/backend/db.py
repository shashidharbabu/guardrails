"""
SQLite database for app sessions and feedback.
Uses a separate DB from gateway_logs.db and mad_store.db.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "app_sessions.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id                   TEXT PRIMARY KEY,
            created_at           TEXT NOT NULL,
            query                TEXT NOT NULL,
            gateway_decision     TEXT NOT NULL,
            gateway_score        REAL NOT NULL,
            gateway_payload      TEXT NOT NULL,
            llm_answer           TEXT,
            mad_routing          TEXT,
            mad_confidence       REAL,
            mad_output_json      TEXT,
            mad_query_id         TEXT,
            mad_rollout_id       TEXT,
            langfuse_trace_id    TEXT,
            pipeline_duration_ms INTEGER
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL REFERENCES sessions(id),
            rating      INTEGER NOT NULL,
            comment     TEXT,
            label       TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_decision ON sessions(gateway_decision);
        CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
        """)


def insert_session(s: dict):
    with _get_conn() as conn:
        conn.execute("""
        INSERT OR REPLACE INTO sessions
          (id, created_at, query, gateway_decision, gateway_score,
           gateway_payload, llm_answer, mad_routing, mad_confidence,
           mad_output_json, mad_query_id, mad_rollout_id,
           langfuse_trace_id, pipeline_duration_ms)
        VALUES
          (:id, :created_at, :query, :gateway_decision, :gateway_score,
           :gateway_payload, :llm_answer, :mad_routing, :mad_confidence,
           :mad_output_json, :mad_query_id, :mad_rollout_id,
           :langfuse_trace_id, :pipeline_duration_ms)
        """, s)


def get_sessions(limit: int = 100) -> list:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_deserialize_session(dict(r)) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return _deserialize_session(dict(row)) if row else None


def update_session_mad(session_id: str, mad_routing: str, mad_confidence: float,
                       mad_output_json: str, pipeline_duration_ms: int,
                       mad_query_id: str = "", mad_rollout_id: str = ""):
    """Patch MAD results into an existing session row (called from background task)."""
    with _get_conn() as conn:
        conn.execute("""
        UPDATE sessions SET
            mad_routing = ?, mad_confidence = ?,
            mad_output_json = ?, mad_query_id = ?,
            mad_rollout_id = ?, pipeline_duration_ms = ?
        WHERE id = ?
        """, (mad_routing, mad_confidence, mad_output_json,
              mad_query_id, mad_rollout_id, pipeline_duration_ms, session_id))


def insert_feedback(f: dict):
    with _get_conn() as conn:
        conn.execute("""
        INSERT OR REPLACE INTO feedback (id, session_id, rating, comment, label, created_at)
        VALUES (:id, :session_id, :rating, :comment, :label, :created_at)
        """, f)


def get_feedback(limit: int = 200) -> list:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_analytics() -> dict:
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        decisions = conn.execute(
            "SELECT gateway_decision, COUNT(*) as cnt FROM sessions GROUP BY gateway_decision"
        ).fetchall()
        avg_ms = conn.execute(
            "SELECT AVG(pipeline_duration_ms) FROM sessions WHERE pipeline_duration_ms IS NOT NULL"
        ).fetchone()[0]
        routing = conn.execute(
            "SELECT mad_routing, COUNT(*) as cnt FROM sessions WHERE mad_routing IS NOT NULL GROUP BY mad_routing"
        ).fetchall()

    d_map = {r["gateway_decision"]: r["cnt"] for r in decisions}
    return {
        "total_sessions":  total,
        "blocked_count":   d_map.get("BLOCK", 0),
        "escalated_count": d_map.get("ESCALATE", 0),
        "passed_count":    d_map.get("PASS", 0),
        "avg_pipeline_ms": int(avg_ms) if avg_ms else 0,
        "decisions":       [{"decision": k, "count": v} for k, v in d_map.items()],
        "mad_routing":     [{"routing": r["mad_routing"], "count": r["cnt"]} for r in routing],
    }


def _deserialize_session(s: dict) -> dict:
    """Parse JSON columns back to dicts/lists."""
    for col in ("gateway_payload", "mad_output_json"):
        val = s.get(col)
        if isinstance(val, str):
            try:
                s[col] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                s[col] = None
    # Rename mad_output_json → mad_output for frontend compatibility
    s["mad_output"] = s.pop("mad_output_json", None)
    return s
