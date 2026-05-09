"""
auth.py — Login endpoint that issues a signed JWT.

Single built-in admin account for the research demo.
Credentials come from env vars ADMIN_USERNAME / ADMIN_PASSWORD.
"""

import os
import time

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.backend.config import get_settings

router = APIRouter(prefix="/api/auth", tags=["auth"])
settings = get_settings()

_ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "spartanguard2026")


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: str
    name: str
    email: str
    org: str


def _issue_jwt(user_id: str, role: str, email: str) -> str:
    exp = int(time.time()) + settings.JWT_EXPIRE_MINUTES * 60
    payload = {
        "sub": user_id,
        "role": role,
        "email": email,
        "exp": exp,
        "iat": int(time.time()),
    }
    try:
        from jose import jwt as jose_jwt
        return jose_jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    except ImportError:
        pass
    try:
        import jwt as pyjwt
        return pyjwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    except ImportError:
        raise HTTPException(status_code=500, detail="No JWT library available")


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest):
    if body.username != _ADMIN_USERNAME or body.password != _ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = _issue_jwt(
        user_id="admin-001",
        role="admin",
        email="admin@spartanai.edu",
    )
    return LoginResponse(
        access_token=token,
        user_id="admin-001",
        role="admin",
        name="Northstar AI Governance",
        email="admin@spartanai.edu",
        org="SJSU SpartanGuard",
    )


@router.get("/me")
def me():
    """Public endpoint — returns auth config so frontend knows if auth is enabled."""
    return {
        "auth_enabled": not settings.DISABLE_AUTH,
        "method": "jwt",
    }
