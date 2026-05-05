import uuid
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from app.backend import db

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int
    label: Optional[str] = None
    comment: Optional[str] = None


@router.get("")
def list_feedback():
    return db.get_feedback()


@router.post("")
def create_feedback(body: FeedbackRequest):
    record = {
        "id":         str(uuid.uuid4()),
        "session_id": body.session_id,
        "rating":     body.rating,
        "label":      body.label,
        "comment":    body.comment,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    db.insert_feedback(record)
    return record


@router.get("/export")
def export_csv():
    rows = db.get_feedback()
    def _generate():
        yield "id,session_id,rating,label,comment,created_at\n"
        for r in rows:
            comment = (r.get("comment") or "").replace('"', '""')
            yield f'{r["id"]},{r["session_id"]},{r["rating"]},{r.get("label","")},"{comment}",{r["created_at"]}\n'
    return StreamingResponse(_generate(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=feedback.csv"})
