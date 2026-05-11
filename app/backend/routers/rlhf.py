"""
rlhf.py — Proxy router to the RLHF feedback loop service (:8002).

Exposes the feedback loop's human review queue and reward scoring
through the main backend API so the frontend can reach it via /api/rlhf/*.

All heavy lifting (SQLite reads/writes, GRPO advantage computation) is done
by the standalone rlhf.feedback_loop.api service.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.backend.auth import UserContext, require_capability
from app.backend.config import get_settings

router = APIRouter(prefix="/api/rlhf", tags=["rlhf"])
logger = logging.getLogger(__name__)


def _feedback_url(path: str) -> str:
    base = get_settings().FEEDBACK_API_URL.rstrip("/")
    return f"{base}{path}"


async def _proxy_get(path: str) -> Any:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(_feedback_url(path))
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, detail="Feedback loop service unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)


async def _proxy_post(path: str, body: dict | None = None) -> Any:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(_feedback_url(path), json=body or {})
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, detail="Feedback loop service unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
async def rlhf_health():
    """Liveness check — pings the feedback loop service."""
    return await _proxy_get("/health")


@router.post("/rewards/compute")
async def compute_rewards(
    user: UserContext = Depends(require_capability("write:feedback")),
):
    """Trigger full batch reward scoring + GRPO advantage on MAD DB."""
    return await _proxy_post("/rewards/compute")


@router.get("/review/next")
async def review_next(
    user: UserContext = Depends(require_capability("read:feedback")),
):
    """Get next item from the ambiguous human review queue."""
    return await _proxy_get("/review/next")


class ReviewDecision(BaseModel):
    decision: str  # good | bad | skip


@router.post("/review/{query_id}/{rollout_id}/{claim_id}")
async def submit_review(
    query_id: str,
    rollout_id: str,
    claim_id: str,
    body: ReviewDecision,
    user: UserContext = Depends(require_capability("write:feedback")),
):
    """Submit a human review decision for a specific reward row."""
    return await _proxy_post(
        f"/review/{query_id}/{rollout_id}/{claim_id}",
        body={"decision": body.decision},
    )


@router.get("/review/log")
async def review_log(
    tail: int = 50,
    user: UserContext = Depends(require_capability("read:feedback")),
):
    """Get the last N human review audit log entries."""
    return await _proxy_get(f"/review/log?tail={tail}")
