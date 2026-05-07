"""
Audit log router.
Read-only — logs are append-only from the application perspective.
Requires auditor or admin role.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.backend import db
from app.backend.auth import UserContext, require_capability

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("/logs")
def list_audit_logs(
    session_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    user: UserContext = Depends(require_capability("read:audit_logs")),
):
    return db.get_audit_logs(session_id=session_id, action=action, limit=limit)


@router.get("/logs/{session_id}")
def get_session_audit_logs(
    session_id: str,
    user: UserContext = Depends(require_capability("read:audit_logs")),
):
    return db.get_audit_logs(session_id=session_id, limit=500)
