from fastapi import APIRouter
from app.backend import db

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary")
def summary():
    return db.get_analytics()
