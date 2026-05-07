from fastapi import APIRouter, Depends
from app.backend import db
from app.backend.auth import UserContext, require_capability

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary")
def summary(user: UserContext = Depends(require_capability("read:analytics"))):
    return db.get_analytics()
