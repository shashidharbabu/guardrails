"""
Database layer.
Uses SQLAlchemy Core + an engine derived from DATABASE_URL.
Supports PostgreSQL (production) and SQLite (local dev only).
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column, DateTime, Float, Index, Integer, String, Text,
    create_engine, text, MetaData, Table, select, insert, update,
)
from sqlalchemy.pool import StaticPool

from app.backend.config import get_settings

settings = get_settings()

_db_url = settings.effective_database_url
_is_sqlite = _db_url.startswith("sqlite")

_connect_args = {"check_same_thread": False} if _is_sqlite else {}
_pool_kwargs = {"poolclass": StaticPool} if _is_sqlite else {}

engine = create_engine(
    _db_url,
    connect_args=_connect_args,
    echo=settings.is_development,
    **_pool_kwargs,
)

metadata = MetaData()

# ── Table definitions ─────────────────────────────────────────────────────────

sessions_table = Table(
    "sessions",
    metadata,
    Column("id", String, primary_key=True),
    Column("created_at", String, nullable=False),
    Column("query", Text, nullable=False),
    Column("status", String, nullable=False, default="RECEIVED"),
    Column("gateway_decision", String, nullable=False),
    Column("gateway_score", Float, nullable=False),
    Column("gateway_payload", Text, nullable=False),
    Column("llm_answer", Text),
    Column("llm_model", String),
    Column("mad_routing", String),
    Column("mad_confidence", Float),
    Column("mad_output_json", Text),
    Column("mad_query_id", String),
    Column("mad_rollout_id", String),
    Column("langfuse_trace_id", String),
    Column("pipeline_duration_ms", Integer),
    Column("cse_result_json", Text),
    Column("final_route", String),
    Column("tenant_id", String),
    Column("user_id", String),
    # Indexes defined below
)

feedback_table = Table(
    "feedback",
    metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, nullable=False),
    Column("rating", Integer, nullable=False),
    Column("comment", Text),
    Column("label", String),
    Column("status", String, default="open"),
    Column("category", String),
    Column("severity", String),
    Column("reviewer_notes", Text),
    Column("created_at", String, nullable=False),
    Column("updated_at", String),
    Column("resolved_at", String),
)

audit_logs_table = Table(
    "audit_logs",
    metadata,
    Column("id", String, primary_key=True),
    Column("tenant_id", String),
    Column("request_id", String),
    Column("session_id", String),
    Column("actor_id", String),
    Column("actor_role", String),
    Column("actor_type", String),  # system | user | worker | service
    Column("action", String, nullable=False),
    Column("resource_type", String),
    Column("resource_id", String),
    Column("before_json", Text),
    Column("after_json", Text),
    Column("metadata_json", Text),
    Column("ip_address", String),
    Column("user_agent", String),
    Column("timestamp", String, nullable=False),
)

session_events_table = Table(
    "session_events",
    metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, nullable=False),
    Column("request_id", String),
    Column("tenant_id", String),
    Column("stage", String, nullable=False),
    Column("from_status", String),
    Column("to_status", String, nullable=False),
    Column("timestamp", String, nullable=False),
    Column("actor_type", String, default="system"),
    Column("actor_id", String),
    Column("message", Text),
    Column("metadata_json", Text),
    Column("error_code", String),
    Column("error_message", Text),
)

human_reviews_table = Table(
    "human_reviews",
    metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, nullable=False),
    Column("reviewer_id", String),
    Column("review_status", String, default="pending"),  # pending | in_progress | completed
    Column("review_decision", String),  # approved | rejected | escalated | false_positive | false_negative
    Column("review_notes", Text),
    Column("created_at", String, nullable=False),
    Column("updated_at", String),
    Column("completed_at", String),
)


# ── Schema init ───────────────────────────────────────────────────────────────

def init_db():
    metadata.create_all(engine)
    _migrate_sessions_table()
    _migrate_feedback_table()
    _add_indexes()


def _migrate_sessions_table():
    """Add columns that were introduced after the initial schema was deployed."""
    new_columns = [
        ("status",               "TEXT DEFAULT 'RECEIVED'"),
        ("llm_model",            "TEXT"),
        ("cse_result_json",      "TEXT"),
        ("final_route",          "TEXT"),
        ("tenant_id",            "TEXT"),
        ("user_id",              "TEXT"),
    ]
    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(sessions)")).fetchall()}
        for col_name, col_def in new_columns:
            if col_name not in existing:
                try:
                    conn.execute(text(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_def}"))
                except Exception:
                    pass
        conn.commit()


def _migrate_feedback_table():
    """Add feedback columns introduced after the initial local SQLite schema."""
    if not _is_sqlite:
        return
    new_columns = [
        ("status",         "TEXT DEFAULT 'open'"),
        ("category",       "TEXT"),
        ("severity",       "TEXT"),
        ("reviewer_notes", "TEXT"),
        ("updated_at",     "TEXT"),
        ("resolved_at",    "TEXT"),
    ]
    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(feedback)")).fetchall()}
        for col_name, col_def in new_columns:
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE feedback ADD COLUMN {col_name} {col_def}"))
        conn.commit()


def _add_indexes():
    """Create indexes that aren't expressible in Table() declarations for SQLite compat."""
    index_defs = [
        ("idx_sessions_created_at", "sessions", "created_at"),
        ("idx_sessions_status", "sessions", "status"),
        ("idx_sessions_gateway_decision", "sessions", "gateway_decision"),
        ("idx_sessions_mad_routing", "sessions", "mad_routing"),
        ("idx_sessions_final_route", "sessions", "final_route"),
        ("idx_sessions_tenant_id", "sessions", "tenant_id"),
        ("idx_feedback_session_id", "feedback", "session_id"),
        ("idx_feedback_status", "feedback", "status"),
        ("idx_audit_logs_session_id", "audit_logs", "session_id"),
        ("idx_audit_logs_timestamp", "audit_logs", "timestamp"),
        ("idx_audit_logs_action", "audit_logs", "action"),
        ("idx_session_events_session_id", "session_events", "session_id"),
        ("idx_human_reviews_session_id", "human_reviews", "session_id"),
        ("idx_human_reviews_review_status", "human_reviews", "review_status"),
    ]
    with engine.connect() as conn:
        for idx_name, tbl_name, col_name in index_defs:
            try:
                if _is_sqlite:
                    conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {tbl_name}({col_name})")
                    )
                else:
                    conn.execute(
                        text(
                            f"CREATE INDEX IF NOT EXISTS {idx_name} ON {tbl_name}({col_name})"
                        )
                    )
            except Exception:
                pass  # index may already exist
        conn.commit()


def ping():
    """Lightweight DB connectivity check used by /readyz."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


# ── Sessions ──────────────────────────────────────────────────────────────────

def insert_session(s: dict):
    with engine.begin() as conn:
        row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in s.items()}
        conn.execute(sessions_table.insert().prefix_with("OR REPLACE" if _is_sqlite else "").values(**row))


def get_sessions(limit: int = 100, status: Optional[str] = None, tenant_id: Optional[str] = None) -> list:
    stmt = select(sessions_table).order_by(sessions_table.c.created_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(sessions_table.c.status == status)
    if tenant_id:
        stmt = stmt.where(sessions_table.c.tenant_id == tenant_id)
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return [_deserialize_session(dict(r._mapping)) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    stmt = select(sessions_table).where(sessions_table.c.id == session_id)
    with engine.connect() as conn:
        row = conn.execute(stmt).fetchone()
    return _deserialize_session(dict(row._mapping)) if row else None


def update_session_mad(
    session_id: str,
    mad_routing: str,
    mad_confidence: float,
    mad_output_json: str,
    pipeline_duration_ms: int,
    mad_query_id: str = "",
    mad_rollout_id: str = "",
    cse_result_json: Optional[str] = None,
    langfuse_trace_id: Optional[str] = None,
):
    new_status = _mad_routing_to_status(mad_routing)
    vals = dict(
        mad_routing=mad_routing,
        mad_confidence=mad_confidence,
        mad_output_json=mad_output_json,
        mad_query_id=mad_query_id,
        mad_rollout_id=mad_rollout_id,
        pipeline_duration_ms=pipeline_duration_ms,
        status=new_status,
        cse_result_json=cse_result_json,
    )
    if langfuse_trace_id is not None:
        vals["langfuse_trace_id"] = langfuse_trace_id
    with engine.begin() as conn:
        conn.execute(
            sessions_table.update()
            .where(sessions_table.c.id == session_id)
            .values(**vals)
        )


def update_session_status(session_id: str, new_status: str, error_message: Optional[str] = None):
    vals: dict = {"status": new_status}
    with engine.begin() as conn:
        conn.execute(
            sessions_table.update()
            .where(sessions_table.c.id == session_id)
            .values(**vals)
        )


def _mad_routing_to_status(routing: str) -> str:
    mapping = {
        "DELIVER": "COMPLETED",
        "RETRY": "RETRY_QUEUED",
        "HUMAN_REVIEW": "HUMAN_REVIEW_REQUIRED",
        "HARD_BLOCK": "HARD_BLOCKED",
        "BLOCK": "HARD_BLOCKED",
    }
    return mapping.get(routing, "MAD_COMPLETED")


def _strip_controls(obj):
    """Recursively strip C0 control characters (except \\t \\n \\r) from all strings."""
    import re
    _re = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
    if isinstance(obj, str):
        return _re.sub(' ', obj)
    if isinstance(obj, dict):
        return {k: _strip_controls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_controls(v) for v in obj]
    return obj


def _deserialize_session(s: dict) -> dict:
    import re
    for col in ("gateway_payload", "mad_output_json", "cse_result_json"):
        val = s.get(col)
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                try:
                    parsed = json.loads(re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', val))
                except (json.JSONDecodeError, TypeError):
                    parsed = None
            s[col] = _strip_controls(parsed) if parsed is not None else None
    # Expose mad_output_json as both mad_output (frontend) and keep original key for CSE endpoint
    s["mad_output"] = s.get("mad_output_json")
    return s


# ── Session Events ────────────────────────────────────────────────────────────

def insert_session_event(
    session_id: str,
    stage: str,
    to_status: str,
    from_status: Optional[str] = None,
    actor_type: str = "system",
    actor_id: Optional[str] = None,
    message: Optional[str] = None,
    metadata: Optional[dict] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    request_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
):
    with engine.begin() as conn:
        conn.execute(
            session_events_table.insert().values(
                id=str(uuid.uuid4()),
                session_id=session_id,
                request_id=request_id,
                tenant_id=tenant_id,
                stage=stage,
                from_status=from_status,
                to_status=to_status,
                timestamp=datetime.now(timezone.utc).isoformat(),
                actor_type=actor_type,
                actor_id=actor_id,
                message=message,
                metadata_json=json.dumps(metadata) if metadata else None,
                error_code=error_code,
                error_message=error_message,
            )
        )


def get_session_events(session_id: str) -> list:
    stmt = (
        select(session_events_table)
        .where(session_events_table.c.session_id == session_id)
        .order_by(session_events_table.c.timestamp)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return [dict(r._mapping) for r in rows]


# ── Feedback ──────────────────────────────────────────────────────────────────

def insert_feedback(f: dict):
    with engine.begin() as conn:
        conn.execute(feedback_table.insert().values(**f))


def get_feedback(limit: int = 200, status: Optional[str] = None) -> list:
    stmt = select(feedback_table).order_by(feedback_table.c.created_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(feedback_table.c.status == status)
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return [dict(r._mapping) for r in rows]


def update_feedback(feedback_id: str, updates: dict):
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    with engine.begin() as conn:
        conn.execute(
            feedback_table.update()
            .where(feedback_table.c.id == feedback_id)
            .values(**updates)
        )


# ── Audit Logs ────────────────────────────────────────────────────────────────

def insert_audit_log(
    action: str,
    actor_id: Optional[str] = None,
    actor_role: Optional[str] = None,
    actor_type: str = "system",
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    before: Optional[dict] = None,
    after: Optional[dict] = None,
    metadata: Optional[dict] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
):
    if not settings.ENABLE_AUDIT_LOGGING:
        return
    with engine.begin() as conn:
        conn.execute(
            audit_logs_table.insert().values(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                request_id=request_id,
                session_id=session_id,
                actor_id=actor_id,
                actor_role=actor_role,
                actor_type=actor_type,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                before_json=json.dumps(before) if before else None,
                after_json=json.dumps(after) if after else None,
                metadata_json=json.dumps(metadata) if metadata else None,
                ip_address=ip_address,
                user_agent=user_agent,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )


def get_audit_logs(
    session_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 200,
) -> list:
    stmt = select(audit_logs_table).order_by(audit_logs_table.c.timestamp.desc()).limit(limit)
    if session_id:
        stmt = stmt.where(audit_logs_table.c.session_id == session_id)
    if action:
        stmt = stmt.where(audit_logs_table.c.action == action)
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    result = []
    for r in rows:
        row = dict(r._mapping)
        for col in ("before_json", "after_json", "metadata_json"):
            if isinstance(row.get(col), str):
                try:
                    row[col] = json.loads(row[col])
                except Exception:
                    pass
        result.append(row)
    return result


# ── Human Reviews ─────────────────────────────────────────────────────────────

def insert_human_review(review: dict):
    with engine.begin() as conn:
        conn.execute(human_reviews_table.insert().values(**review))


def get_human_review(review_id: str) -> Optional[dict]:
    stmt = select(human_reviews_table).where(human_reviews_table.c.id == review_id)
    with engine.connect() as conn:
        row = conn.execute(stmt).fetchone()
    return dict(row._mapping) if row else None


def get_human_reviews(
    review_status: Optional[str] = None,
    limit: int = 100,
) -> list:
    stmt = select(human_reviews_table).order_by(human_reviews_table.c.created_at.desc()).limit(limit)
    if review_status:
        stmt = stmt.where(human_reviews_table.c.review_status == review_status)
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return [dict(r._mapping) for r in rows]


def update_human_review(review_id: str, updates: dict):
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    with engine.begin() as conn:
        conn.execute(
            human_reviews_table.update()
            .where(human_reviews_table.c.id == review_id)
            .values(**updates)
        )


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_analytics() -> dict:
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM sessions")).scalar()

        decisions = conn.execute(
            text("SELECT gateway_decision, COUNT(*) as cnt FROM sessions GROUP BY gateway_decision")
        ).fetchall()

        avg_ms = conn.execute(
            text("SELECT AVG(pipeline_duration_ms) FROM sessions WHERE pipeline_duration_ms IS NOT NULL")
        ).scalar()

        routing = conn.execute(
            text("SELECT mad_routing, COUNT(*) as cnt FROM sessions WHERE mad_routing IS NOT NULL GROUP BY mad_routing")
        ).fetchall()

        status_counts = conn.execute(
            text("SELECT status, COUNT(*) as cnt FROM sessions GROUP BY status")
        ).fetchall()

        review_queue_size = conn.execute(
            text("SELECT COUNT(*) FROM human_reviews WHERE review_status = 'pending'")
        ).scalar()

        feedback_count = conn.execute(
            text("SELECT COUNT(*) FROM feedback")
        ).scalar()

        open_feedback = conn.execute(
            text("SELECT COUNT(*) FROM feedback WHERE status = 'open'")
        ).scalar()

        hard_blocks = conn.execute(
            text("SELECT COUNT(*) FROM sessions WHERE status = 'HARD_BLOCKED' OR mad_routing = 'HARD_BLOCK'")
        ).scalar()

    d_map = {r[0]: r[1] for r in decisions}
    return {
        "total_sessions": total,
        "blocked_count": d_map.get("BLOCK", 0),
        "escalated_count": d_map.get("ESCALATE", 0),
        "passed_count": d_map.get("PASS", 0),
        "hard_blocked_count": hard_blocks,
        "avg_pipeline_ms": int(avg_ms) if avg_ms else 0,
        "decisions": [{"decision": k, "count": v} for k, v in d_map.items()],
        "mad_routing": [{"routing": r[0], "count": r[1]} for r in routing],
        "session_statuses": [{"status": r[0], "count": r[1]} for r in status_counts],
        "human_review_queue_size": review_queue_size,
        "feedback_count": feedback_count,
        "open_feedback_count": open_feedback,
    }
