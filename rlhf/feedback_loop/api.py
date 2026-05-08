"""FastAPI service for human review queue and batch scoring triggers."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Literal, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rlhf.feedback_loop.audit_log import append_audit, read_audit_tail
from rlhf.feedback_loop.config import (
    feedback_api_host,
    feedback_api_port,
    get_db_path,
    human_feedback_log_path,
    triage_ambiguous_high,
    triage_ambiguous_low,
)
from rlhf.feedback_loop.db import connect, init_feedback_schema
from rlhf.feedback_loop.refresh import refresh_advantages
from rlhf.feedback_loop.scorer import run_batch_scoring

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_feedback_schema()
    log.info("Feedback API using DB %s", get_db_path())
    yield


app = FastAPI(title="Guardrails RLHF Feedback", version="0.1.0", lifespan=lifespan)


class ReviewDecision(BaseModel):
    decision: Literal["good", "bad", "skip"] = Field(
        ..., description="good=+0.80, bad=-0.60, skip=keep auto_reward"
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "db": get_db_path()}


@app.post("/rewards/compute")
def rewards_compute() -> Dict[str, Union[int, float]]:
    """Run full batch scoring + GRPO advantage on MAD_DB_PATH."""
    return run_batch_scoring(apply_advantage=True)


@app.get("/review/next")
def review_next() -> Dict[str, Any]:
    """Next item in ambiguous auto_reward band needing human review."""
    low, high = triage_ambiguous_low(), triage_ambiguous_high()
    with connect() as con:
        row = con.execute(
            """
            SELECT r.query_id, r.rollout_id, r.claim_id, r.auto_reward,
                   r.final_reward, r.human_reviewed,
                   q.user_query      AS query_text,
                   ao.reasoning      AS agent_a_context
            FROM rewards r
            JOIN claims c
              ON c.claim_id = r.claim_id
            JOIN queries q
              ON q.query_id = c.query_id
             AND q.run_id = r.rollout_id
            LEFT JOIN agent_outputs ao
              ON ao.claim_id = r.claim_id
             AND ao.agent_role = 'agent_a'
             AND ao.round_num = 1
            WHERE r.human_reviewed = 0
              AND r.auto_reward >= ? AND r.auto_reward <= ?
            ORDER BY r.scored_at ASC
            LIMIT 1
            """,
            (low, high),
        ).fetchone()
    if row is None:
        return {"pending": False}
    return {
        "pending": True,
        "query_id": row["query_id"],
        "rollout_id": row["rollout_id"],
        "claim_id": row["claim_id"],
        "auto_reward": row["auto_reward"],
        "triage_band": [low, high],
        "prompt": row["query_text"],
        "agent_a_context": row["agent_a_context"],
    }


@app.post("/review/{query_id}/{rollout_id}/{claim_id}")
def submit_review(
    query_id: str,
    rollout_id: str,
    claim_id: str,
    body: ReviewDecision,
) -> Dict[str, Any]:
    low, high = triage_ambiguous_low(), triage_ambiguous_high()
    with connect() as con:
        row = con.execute(
            """
            SELECT auto_reward, human_reviewed FROM rewards
            WHERE query_id = ? AND rollout_id = ? AND claim_id = ?
            """,
            (query_id, rollout_id, claim_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Reward row not found")
        if int(row["human_reviewed"] or 0) == 1:
            raise HTTPException(status_code=409, detail="Already reviewed")
        auto = float(row["auto_reward"])
        if not (low <= auto <= high):
            raise HTTPException(
                status_code=400,
                detail=f"auto_reward {auto} outside ambiguous band [{low}, {high}]",
            )
        if body.decision == "good":
            human_r, final_r = 0.80, 0.80
        elif body.decision == "bad":
            human_r, final_r = -0.60, -0.60
        else:
            human_r, final_r = None, auto  # skip: keep auto_reward as final
        con.execute(
            """
            UPDATE rewards
            SET human_reviewed = 1,
                human_reward = ?,
                final_reward = ?,
                grpo_advantage = NULL
            WHERE query_id = ? AND rollout_id = ? AND claim_id = ?
            """,
            (human_r, final_r, query_id, rollout_id, claim_id),
        )
    record = {
        "query_id": query_id,
        "rollout_id": rollout_id,
        "claim_id": claim_id,
        "decision": body.decision,
        "auto_reward": auto,
        "human_reward": human_r,
        "final_reward": final_r,
        "human_reviewed": True,
    }
    append_audit(record, human_feedback_log_path())
    refresh_advantages()
    return record


@app.get("/review/log")
def review_log(tail: int = 50) -> Dict[str, Any]:
    return {"entries": read_audit_tail(human_feedback_log_path(), n=tail)}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "rlhf.feedback_loop.api:app",
        host=feedback_api_host(),
        port=feedback_api_port(),
        reload=False,
    )


if __name__ == "__main__":
    main()
