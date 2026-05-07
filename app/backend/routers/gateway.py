"""
Gateway proxy router.
Forwards /api/gateway/* calls to the internal gateway service.
Service URL comes from settings — no hardcoded localhost.
"""

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.backend.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/api/gateway", tags=["gateway"])


def _gateway_url() -> str:
    return settings.GATEWAY_URL


async def _proxy_get(path: str, params: dict = None, timeout: float = 10.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{_gateway_url()}{path}", params=params or {})
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, f"Gateway not reachable at {_gateway_url()}. Is it running?")
    except httpx.TimeoutException:
        raise HTTPException(504, "Gateway timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


async def _proxy_post(path: str, body: dict, timeout: float = 60.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{_gateway_url()}{path}", json=body)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, f"Gateway not reachable at {_gateway_url()}. Is it running?")
    except httpx.TimeoutException:
        raise HTTPException(504, "Gateway timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


async def _proxy_patch(path: str, body: dict, timeout: float = 120.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.patch(f"{_gateway_url()}{path}", json=body)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, f"Gateway not reachable at {_gateway_url()}. Is it running?")
    except httpx.TimeoutException:
        raise HTTPException(504, "Gateway timed out (model reload may still be in progress)")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


@router.post("/validate")
async def validate(body: dict):
    return await _proxy_post("/validate", body)


@router.get("/health")
async def health():
    return await _proxy_get("/health", timeout=5.0)


@router.get("/logs")
async def logs(limit: int = Query(50, ge=1, le=500)):
    return await _proxy_get("/logs", params={"limit": limit})


@router.get("/stats")
async def stats():
    return await _proxy_get("/stats")


@router.get("/config")
async def get_config():
    return await _proxy_get("/config")


@router.patch("/config")
async def update_config(body: dict):
    return await _proxy_patch("/config", body)


@router.post("/config/reset")
async def reset_config():
    return await _proxy_post("/config/reset", {})
