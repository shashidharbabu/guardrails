"""
Enterprise Guardrails — App Backend
Run: uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
Production: gunicorn app.backend.main:app -k uvicorn.workers.UvicornWorker
"""

import logging
import sys
import time
import uuid
from pathlib import Path

# Allow multi_agent_debate imports from repo root
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "multi_agent_debate"))

from dotenv import load_dotenv
load_dotenv(_repo_root / ".env")

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.backend.config import get_settings
from app.backend.logging_config import configure_logging
from app.backend import db
from app.backend.routers import gateway, sessions, analytics, feedback
from app.backend.routers.sessions import query_router
from app.backend.routers import system as system_router
from app.backend.routers import audit as audit_router
from app.backend.routers import human_review as human_review_router
from app.backend.routers import copilot as copilot_router
from app.backend.routers import rlhf as rlhf_router
from app.backend.routers import auth as auth_router

settings = get_settings()
configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Validate production requirements on startup
settings.validate_production()

app = FastAPI(
    title="Enterprise Guardrails — App Backend",
    version=settings.APP_VERSION,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

# ── Request ID + structured access logging middleware ─────────────────────────
@app.middleware("http")
async def request_id_and_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.time()
    response: Response = await call_next(request)
    latency_ms = int((time.time() - t0) * 1000)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "http_request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "remote_addr": request.client.host if request.client else "unknown",
        },
    )
    return response

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(gateway.router)
app.include_router(sessions.router)
app.include_router(query_router)
app.include_router(analytics.router)
app.include_router(feedback.router)
app.include_router(system_router.router)
app.include_router(audit_router.router)
app.include_router(human_review_router.router)
app.include_router(copilot_router.router)
app.include_router(rlhf_router.router)
app.include_router(auth_router.router)

# ── Shallow health probes (no auth, no DB — for load balancer liveness) ──────
@app.get("/healthz", tags=["health"], include_in_schema=False)
def healthz():
    return {"status": "ok"}


@app.get("/livez", tags=["health"], include_in_schema=False)
def livez():
    return {"status": "ok"}


@app.get("/readyz", tags=["health"], include_in_schema=False)
def readyz():
    """Readiness — fails if the DB is not reachable."""
    try:
        db.ping()
        return {"status": "ready"}
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(503, detail=f"Not ready: {exc}")


# ── Startup / shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    db.init_db()
    logger.info(
        "startup",
        extra={
            "app_env": settings.APP_ENV.value,
            "app_version": settings.APP_VERSION,
            "cors_origins": settings.CORS_ALLOWED_ORIGINS,
            "disable_auth": settings.DISABLE_AUTH,
        },
    )
