"""
Gateway proxy router.

All /api/gateway/* calls are forwarded to the gateway server on :8080.
Using httpx for async HTTP with appropriate timeouts.
"""

import httpx
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

GATEWAY_URL = "http://localhost:8080"

router = APIRouter(prefix="/api/gateway", tags=["gateway"])


async def _proxy_get(path: str, params: dict = None, timeout: float = 10.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{GATEWAY_URL}{path}", params=params or {})
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Gateway server not reachable on :8080. Is it running?")
    except httpx.TimeoutException:
        raise HTTPException(504, "Gateway server timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


async def _proxy_post(path: str, body: dict, timeout: float = 60.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{GATEWAY_URL}{path}", json=body)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Gateway server not reachable on :8080. Is it running?")
    except httpx.TimeoutException:
        raise HTTPException(504, "Gateway server timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


async def _proxy_patch(path: str, body: dict, timeout: float = 120.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.patch(f"{GATEWAY_URL}{path}", json=body)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Gateway server not reachable on :8080. Is it running?")
    except httpx.TimeoutException:
        raise HTTPException(504, "Gateway server timed out (model reload may still be in progress)")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/validate")
async def validate(body: dict):
    """Run a query through the full gateway pipeline."""
    return await _proxy_post("/validate", body)


@router.get("/health")
async def health():
    """Gateway health — model load status."""
    return await _proxy_get("/health", timeout=5.0)


@router.get("/logs")
async def logs(limit: int = Query(50, ge=1, le=500)):
    """Recent gateway audit events."""
    return await _proxy_get("/logs", params={"limit": limit})


@router.get("/stats")
async def stats():
    """Decision distribution statistics."""
    return await _proxy_get("/stats")


@router.get("/config")
async def get_config():
    """Current in-memory gateway config (thresholds + model paths)."""
    return await _proxy_get("/config")


@router.patch("/config")
async def update_config(body: dict):
    """
    Update thresholds or model paths.
    Threshold changes: instant.
    Model path changes: ~30-60s warmup on next validate call.
    """
    return await _proxy_patch("/config", body)


@router.post("/config/reset")
async def reset_config():
    """Reset all thresholds and model paths to server defaults."""
    return await _proxy_post("/config/reset", {})
