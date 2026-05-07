"""initial_production_schema

Revision ID: caf2cb1bbb0a
Revises:
Create Date: 2026-05-06

Creates the full production schema from scratch.
Handles pre-existing tables by using batch_alter for add_column operations.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = 'caf2cb1bbb0a'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    return inspect(bind).has_table(name)


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    cols = [c["name"] for c in inspect(bind).get_columns(table)]
    return column in cols


def upgrade() -> None:
    # ── sessions ──────────────────────────────────────────────────────────────
    if not _table_exists("sessions"):
        op.create_table(
            "sessions",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("created_at", sa.String(), nullable=False),
            sa.Column("query", sa.Text(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default="RECEIVED"),
            sa.Column("gateway_decision", sa.String(), nullable=False),
            sa.Column("gateway_score", sa.Float(), nullable=False),
            sa.Column("gateway_payload", sa.Text(), nullable=False),
            sa.Column("llm_answer", sa.Text(), nullable=True),
            sa.Column("llm_model", sa.String(), nullable=True),
            sa.Column("mad_routing", sa.String(), nullable=True),
            sa.Column("mad_confidence", sa.Float(), nullable=True),
            sa.Column("mad_output_json", sa.Text(), nullable=True),
            sa.Column("mad_query_id", sa.String(), nullable=True),
            sa.Column("mad_rollout_id", sa.String(), nullable=True),
            sa.Column("langfuse_trace_id", sa.String(), nullable=True),
            sa.Column("pipeline_duration_ms", sa.Integer(), nullable=True),
            sa.Column("cse_result_json", sa.Text(), nullable=True),
            sa.Column("final_route", sa.String(), nullable=True),
            sa.Column("tenant_id", sa.String(), nullable=True),
            sa.Column("user_id", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
    else:
        # Add new columns to existing sessions table
        with op.batch_alter_table("sessions") as batch_op:
            for col, type_ in [
                ("status", sa.String()),
                ("llm_model", sa.String()),
                ("cse_result_json", sa.Text()),
                ("final_route", sa.String()),
                ("tenant_id", sa.String()),
                ("user_id", sa.String()),
            ]:
                if not _column_exists("sessions", col):
                    batch_op.add_column(sa.Column(col, type_, nullable=True))

    # ── feedback ──────────────────────────────────────────────────────────────
    if not _table_exists("feedback"):
        op.create_table(
            "feedback",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("session_id", sa.String(), nullable=False),
            sa.Column("rating", sa.Integer(), nullable=False),
            sa.Column("comment", sa.Text(), nullable=True),
            sa.Column("label", sa.String(), nullable=True),
            sa.Column("status", sa.String(), nullable=True),
            sa.Column("category", sa.String(), nullable=True),
            sa.Column("severity", sa.String(), nullable=True),
            sa.Column("reviewer_notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.String(), nullable=False),
            sa.Column("updated_at", sa.String(), nullable=True),
            sa.Column("resolved_at", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
    else:
        with op.batch_alter_table("feedback") as batch_op:
            for col, type_ in [
                ("status", sa.String()),
                ("category", sa.String()),
                ("severity", sa.String()),
                ("reviewer_notes", sa.Text()),
                ("updated_at", sa.String()),
                ("resolved_at", sa.String()),
            ]:
                if not _column_exists("feedback", col):
                    batch_op.add_column(sa.Column(col, type_, nullable=True))

    # ── audit_logs ────────────────────────────────────────────────────────────
    if not _table_exists("audit_logs"):
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("tenant_id", sa.String(), nullable=True),
            sa.Column("request_id", sa.String(), nullable=True),
            sa.Column("session_id", sa.String(), nullable=True),
            sa.Column("actor_id", sa.String(), nullable=True),
            sa.Column("actor_role", sa.String(), nullable=True),
            sa.Column("actor_type", sa.String(), nullable=True),
            sa.Column("action", sa.String(), nullable=False),
            sa.Column("resource_type", sa.String(), nullable=True),
            sa.Column("resource_id", sa.String(), nullable=True),
            sa.Column("before_json", sa.Text(), nullable=True),
            sa.Column("after_json", sa.Text(), nullable=True),
            sa.Column("metadata_json", sa.Text(), nullable=True),
            sa.Column("ip_address", sa.String(), nullable=True),
            sa.Column("user_agent", sa.String(), nullable=True),
            sa.Column("timestamp", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    # ── session_events ────────────────────────────────────────────────────────
    if not _table_exists("session_events"):
        op.create_table(
            "session_events",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("session_id", sa.String(), nullable=False),
            sa.Column("request_id", sa.String(), nullable=True),
            sa.Column("tenant_id", sa.String(), nullable=True),
            sa.Column("stage", sa.String(), nullable=False),
            sa.Column("from_status", sa.String(), nullable=True),
            sa.Column("to_status", sa.String(), nullable=False),
            sa.Column("timestamp", sa.String(), nullable=False),
            sa.Column("actor_type", sa.String(), nullable=True),
            sa.Column("actor_id", sa.String(), nullable=True),
            sa.Column("message", sa.Text(), nullable=True),
            sa.Column("metadata_json", sa.Text(), nullable=True),
            sa.Column("error_code", sa.String(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    # ── human_reviews ─────────────────────────────────────────────────────────
    if not _table_exists("human_reviews"):
        op.create_table(
            "human_reviews",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("session_id", sa.String(), nullable=False),
            sa.Column("reviewer_id", sa.String(), nullable=True),
            sa.Column("review_status", sa.String(), nullable=True),
            sa.Column("review_decision", sa.String(), nullable=True),
            sa.Column("review_notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.String(), nullable=False),
            sa.Column("updated_at", sa.String(), nullable=True),
            sa.Column("completed_at", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )


def downgrade() -> None:
    op.drop_table("human_reviews")
    op.drop_table("session_events")
    op.drop_table("audit_logs")
    with op.batch_alter_table("feedback") as batch_op:
        for col in ("resolved_at", "updated_at", "reviewer_notes", "severity", "category", "status"):
            batch_op.drop_column(col)
    with op.batch_alter_table("sessions") as batch_op:
        for col in ("user_id", "tenant_id", "final_route", "cse_result_json", "llm_model", "status"):
            batch_op.drop_column(col)
