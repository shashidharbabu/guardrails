"""
Event logger — writes all gateway decisions to SQLite for audit trail.
DB file: gateway/gateway_logs.db (auto-created on first run)
"""

import json
import os
import sqlite3
import time
from typing import List

from gateway.decision_engine import GatewayResult


DB_PATH = os.path.join(os.path.dirname(__file__), "gateway_logs.db")


def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gateway_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       REAL    NOT NULL,
            decision        TEXT    NOT NULL,
            gateway_score   REAL    NOT NULL,
            pii_score       REAL    NOT NULL,
            jb_score        REAL    NOT NULL,
            pi_score        REAL    NOT NULL,
            pii_entities    TEXT,
            threat_types    TEXT,
            blocked_reason  TEXT,
            raw_input       TEXT
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_decision ON gateway_events(decision)
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_timestamp ON gateway_events(timestamp)
    """
    )
    conn.commit()
    conn.close()


def log_event(result: GatewayResult):
    """Write a gateway result to the DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO gateway_events
        (timestamp, decision, gateway_score, pii_score, jb_score, pi_score,
         pii_entities, threat_types, blocked_reason, raw_input)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            time.time(),
            result.decision.value,
            round(result.gateway_score, 4),
            round(result.pii_score, 4),
            round(result.jb_score, 4),
            round(result.pi_score, 4),
            json.dumps(result.pii_entities),
            json.dumps(result.threat_types),
            result.blocked_reason,
            result.raw_input[:500],
        ),
    )
    conn.commit()
    conn.close()


def get_recent_events(limit: int = 50) -> List[dict]:
    """Return recent events as list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT * FROM gateway_events ORDER BY timestamp DESC LIMIT ?", (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    for row in rows:
        row["pii_entities"] = json.loads(row.get("pii_entities") or "[]")
        row["threat_types"] = json.loads(row.get("threat_types") or "[]")
    return rows


def get_stats() -> dict:
    """Return decision distribution counts."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT decision, COUNT(*) as count FROM gateway_events GROUP BY decision"
    )
    stats = {row[0]: row[1] for row in cursor.fetchall()}
    total = sum(stats.values())
    conn.close()
    return {"total": total, "distribution": stats}
