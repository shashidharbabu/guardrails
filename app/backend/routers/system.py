"""
System health router.
GET /api/system/health — detailed component health, requires auth.
"""

import time
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends

from app.backend import db
from app.backend.auth import UserContext, get_current_user
from app.backend.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/api/system", tags=["system"])


def _check(name: str, fn) -> dict:
    t0 = time.time()
    try:
        fn()
        return {
            "name": name,
            "status": "healthy",
            "latency_ms": int((time.time() - t0) * 1000),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "unhealthy",
            "error": str(exc),
            "latency_ms": int((time.time() - t0) * 1000),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


async def _check_async(name: str, fn) -> dict:
    t0 = time.time()
    try:
        await fn()
        return {
            "name": name,
            "status": "healthy",
            "latency_ms": int((time.time() - t0) * 1000),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "unhealthy",
            "error": str(exc),
            "latency_ms": int((time.time() - t0) * 1000),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


async def _ping_http(url: str, timeout: float = 3.0) -> None:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url)
        r.raise_for_status()


def _ollama_base_url() -> str:
    base = settings.LLM_PROVIDER_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


@router.get("/health")
async def system_health(user: UserContext = Depends(get_current_user)):
    """
    Deep health check across all pipeline components.
    Returns { status, components, timestamp, version }.
    """
    results = []

    # Database
    results.append(_check("database", db.ping))

    # Gateway
    results.append(
        await _check_async("gateway", lambda: _ping_http(f"{settings.GATEWAY_URL}/health"))
    )

    # LLM runtime
    results.append(
        await _check_async(
            "llm_runtime",
            lambda: _ping_http(f"{_ollama_base_url()}/api/tags", timeout=5.0),
        )
    )

    # Queue (optional)
    if settings.REDIS_URL or settings.QUEUE_URL:
        def _check_redis():
            import redis
            r = redis.from_url(settings.REDIS_URL or settings.QUEUE_URL)
            r.ping()
        results.append(_check("queue", _check_redis))

    # CSE config loaded
    results.append(
        _check(
            "cse_config",
            lambda: None if settings.CSE_CONFIG_VERSION else (_ for _ in ()).throw(
                Exception("CSE_CONFIG_VERSION not set")
            ),
        )
    )

    statuses = {r["status"] for r in results}
    if "unhealthy" in statuses:
        overall = "unhealthy" if all(r["status"] == "unhealthy" for r in results) else "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "components": {r["name"]: r for r in results},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV.value,
    }
