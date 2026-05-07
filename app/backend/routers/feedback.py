import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.backend import db
from app.backend.auth import UserContext, require_capability

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int
    label: Optional[str] = None
    comment: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[str] = None


class FeedbackUpdateRequest(BaseModel):
    status: Optional[str] = None
    reviewer_notes: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[str] = None


@router.get("")
def list_feedback(
    status: Optional[str] = None,
    user: UserContext = Depends(require_capability("read:feedback")),
):
    return db.get_feedback(status=status)


@router.post("")
def create_feedback(
    body: FeedbackRequest,
    request: Request,
    user: UserContext = Depends(require_capability("read:sessions")),
):
    record = {
        "id": str(uuid.uuid4()),
        "session_id": body.session_id,
        "rating": body.rating,
        "label": body.label,
        "comment": body.comment,
        "status": "open",
        "category": body.category,
        "severity": body.severity,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    db.insert_feedback(record)
    db.insert_audit_log(
        action="feedback_created",
        actor_id=user.user_id,
        actor_role=user.role,
        actor_type="user",
        resource_type="feedback",
        resource_id=record["id"],
        session_id=body.session_id,
        request_id=getattr(request.state, "request_id", None),
        tenant_id=user.tenant_id,
        after={"rating": body.rating, "label": body.label},
        ip_address=request.client.host if request.client else None,
    )
    return record


@router.patch("/{feedback_id}")
def update_feedback(
    feedback_id: str,
    body: FeedbackUpdateRequest,
    request: Request,
    user: UserContext = Depends(require_capability("write:feedback")),
):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if body.status == "resolved":
        updates["resolved_at"] = datetime.now(timezone.utc).isoformat()
    db.update_feedback(feedback_id, updates)
    db.insert_audit_log(
        action="feedback_updated",
        actor_id=user.user_id,
        actor_role=user.role,
        resource_type="feedback",
        resource_id=feedback_id,
        request_id=getattr(request.state, "request_id", None),
        after=updates,
    )
    return {"id": feedback_id, **updates}


@router.get("/export")
def export_csv(
    user: UserContext = Depends(require_capability("write:export")),
    request: Request = None,
):
    rows = db.get_feedback()

    def _generate():
        yield "id,session_id,rating,label,status,category,severity,comment,created_at\n"
        for r in rows:
            comment = (r.get("comment") or "").replace('"', '""')
            yield (
                f'{r["id"]},{r["session_id"]},{r["rating"]},'
                f'{r.get("label","")},{r.get("status","")},{r.get("category","")},{r.get("severity","")},'
                f'"{comment}",{r["created_at"]}\n'
            )

    db.insert_audit_log(
        action="feedback_exported",
        actor_id=user.user_id,
        actor_role=user.role,
        resource_type="feedback_export",
        after={"row_count": len(rows)},
    )
    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=feedback.csv"},
    )
