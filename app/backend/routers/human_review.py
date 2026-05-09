"""
Human Review router.
Manages sessions in HUMAN_REVIEW_REQUIRED state.
Actions: approve, reject, escalate, mark_false_positive, mark_false_negative, add_note.
Each action is audit logged.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.backend import db
from app.backend.auth import UserContext, require_capability

router = APIRouter(prefix="/api/human-review", tags=["human_review"])

VALID_DECISIONS = {
    "approve_delivery",
    "reject_answer",
    "request_regeneration",
    "escalate_to_admin",
    "mark_false_positive",
    "mark_false_negative",
    "add_review_note",
}


class ReviewActionRequest(BaseModel):
    decision: str
    notes: Optional[str] = None


@router.get("/queue")
def get_review_queue(
    user: UserContext = Depends(require_capability("read:sessions")),
):
    """Return all sessions currently awaiting human review."""
    sessions = db.get_sessions(status="HUMAN_REVIEW_REQUIRED", limit=200)
    reviews = db.get_human_reviews(review_status="pending")
    review_map = {r["session_id"]: r for r in reviews}
    for s in sessions:
        s["review"] = review_map.get(s["id"])
    return sessions


@router.get("/sessions")
def list_review_sessions(
    user: UserContext = Depends(require_capability("read:sessions")),
):
    """List all sessions that have passed through human review (any status)."""
    return db.get_sessions(status="HUMAN_REVIEW_REQUIRED", limit=200)


@router.get("/reviews")
def list_reviews(
    review_status: Optional[str] = None,
    user: UserContext = Depends(require_capability("read:sessions")),
):
    return db.get_human_reviews(review_status=review_status)


@router.post("/sessions/{session_id}/action")
def take_review_action(
    session_id: str,
    body: ReviewActionRequest,
    request: Request,
    user: UserContext = Depends(require_capability("write:human_review")),
):
    if body.decision not in VALID_DECISIONS:
        raise HTTPException(
            400,
            f"Invalid decision '{body.decision}'. Valid: {sorted(VALID_DECISIONS)}",
        )

    session = db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if session.get("status") not in ("HUMAN_REVIEW_REQUIRED", "HUMAN_REVIEW_APPROVED", "HUMAN_REVIEW_REJECTED"):
        raise HTTPException(
            400,
            f"Session status is '{session.get('status')}', not eligible for review actions.",
        )

    request_id = getattr(request.state, "request_id", None)

    # Derive new session status from decision
    new_session_status = {
        "approve_delivery": "HUMAN_REVIEW_APPROVED",
        "reject_answer": "HUMAN_REVIEW_REJECTED",
        "request_regeneration": "RETRY_QUEUED",
        "escalate_to_admin": "HUMAN_REVIEW_REQUIRED",
        "mark_false_positive": "HUMAN_REVIEW_APPROVED",
        "mark_false_negative": "HUMAN_REVIEW_REJECTED",
        "add_review_note": session.get("status"),  # no state change
    }.get(body.decision, session.get("status"))

    # Upsert human review record
    existing_reviews = db.get_human_reviews(review_status=None)
    existing = next((r for r in existing_reviews if r["session_id"] == session_id), None)

    now = datetime.now(timezone.utc).isoformat()
    review_decision = body.decision if body.decision != "add_review_note" else None

    if existing:
        updates = {
            "review_status": "completed" if review_decision else "in_progress",
            "reviewer_id": user.user_id,
        }
        if review_decision:
            updates["review_decision"] = review_decision
            updates["completed_at"] = now
        if body.notes:
            updates["review_notes"] = body.notes
        db.update_human_review(existing["id"], updates)
        review_id = existing["id"]
    else:
        review_id = str(uuid.uuid4())
        db.insert_human_review({
            "id": review_id,
            "session_id": session_id,
            "reviewer_id": user.user_id,
            "review_status": "completed" if review_decision else "in_progress",
            "review_decision": review_decision,
            "review_notes": body.notes,
            "created_at": now,
            "updated_at": now,
            "completed_at": now if review_decision else None,
        })

    # Transition session status
    if new_session_status != session.get("status"):
        db.update_session_status(session_id, new_session_status)
        db.insert_session_event(
            session_id=session_id,
            stage="human_review",
            to_status=new_session_status,
            from_status=session.get("status"),
            actor_type="user",
            actor_id=user.user_id,
            message=f"Review action: {body.decision}",
            request_id=request_id,
        )

    # Audit log
    db.insert_audit_log(
        action=f"human_review.{body.decision}",
        actor_id=user.user_id,
        actor_role=user.role,
        actor_type="user",
        resource_type="session",
        resource_id=session_id,
        session_id=session_id,
        request_id=request_id,
        tenant_id=user.tenant_id,
        after={"decision": body.decision, "notes": body.notes, "review_id": review_id},
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("User-Agent"),
    )

    return {
        "session_id": session_id,
        "review_id": review_id,
        "decision": body.decision,
        "new_session_status": new_session_status,
        "reviewer_id": user.user_id,
        "timestamp": now,
    }
