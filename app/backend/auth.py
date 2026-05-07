"""
Authentication and RBAC.

Production: validates a Bearer JWT signed by OIDC issuer or internal secret.
Development: DISABLE_AUTH=true bypasses all checks (never in production).

Roles: admin | reviewer | analyst | viewer | auditor | service_account

WARNING: When DISABLE_AUTH=true the app accepts any request as an anonymous
admin. This must NEVER be used in production or staging environments.
"""

import logging
from enum import Enum
from typing import Optional, Set

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_bearer = HTTPBearer(auto_error=False)


class Role(str, Enum):
    admin = "admin"
    reviewer = "reviewer"
    analyst = "analyst"
    viewer = "viewer"
    auditor = "auditor"
    service_account = "service_account"


# Capabilities per role
ROLE_CAPABILITIES: dict[str, Set[str]] = {
    Role.admin: {
        "read:sessions", "read:analytics", "read:feedback", "read:audit_logs",
        "read:gateway_config", "read:system_health",
        "write:gateway_config", "write:settings", "write:feedback",
        "write:human_review", "write:export", "admin:users",
    },
    Role.reviewer: {
        "read:sessions", "read:analytics", "read:feedback",
        "read:system_health", "write:human_review", "write:feedback",
    },
    Role.analyst: {
        "read:sessions", "read:analytics", "read:feedback",
        "read:system_health", "write:export",
    },
    Role.viewer: {
        "read:sessions", "read:analytics", "read:feedback",
        "read:system_health",
    },
    Role.auditor: {
        "read:sessions", "read:audit_logs", "read:analytics",
        "write:export",
    },
    Role.service_account: {
        "read:sessions", "write:sessions",
    },
}


class UserContext:
    def __init__(
        self,
        user_id: str,
        role: str,
        tenant_id: Optional[str] = None,
        email: Optional[str] = None,
    ):
        self.user_id = user_id
        self.role = role
        self.tenant_id = tenant_id
        self.email = email
        self.capabilities: Set[str] = ROLE_CAPABILITIES.get(role, set())

    def can(self, capability: str) -> bool:
        return capability in self.capabilities

    def require(self, capability: str):
        if not self.can(capability):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{self.role}' does not have capability '{capability}'",
            )


def _decode_jwt(token: str) -> dict:
    """
    Decode and verify a JWT.
    Uses python-jose if available; falls back to pyjwt.
    In production configure OIDC_ISSUER_URL for issuer validation.
    """
    try:
        from jose import jwt as jose_jwt, JWTError
        payload = jose_jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.OIDC_AUDIENCE,
        )
        return payload
    except ImportError:
        pass

    try:
        import jwt as pyjwt
        payload = pyjwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="No JWT library available. Install python-jose or PyJWT.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> UserContext:
    if settings.DISABLE_AUTH:
        # Development only — never in production
        return UserContext(
            user_id="dev-user",
            role=Role.admin,
            tenant_id=None,
            email="dev@localhost",
        )

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = _decode_jwt(credentials.credentials)
    role = payload.get("role") or payload.get("guardrails_role", Role.viewer)
    return UserContext(
        user_id=payload.get("sub", "unknown"),
        role=role,
        tenant_id=payload.get("tenant_id"),
        email=payload.get("email"),
    )


# Convenience dependency factories

def require_role(*roles: str):
    """Return a FastAPI dependency that enforces one of the given roles."""
    async def _dep(user: UserContext = Depends(get_current_user)) -> UserContext:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {roles}. Your role: {user.role}",
            )
        return user
    return _dep


def require_capability(capability: str):
    """Return a FastAPI dependency that enforces a specific capability."""
    async def _dep(user: UserContext = Depends(get_current_user)) -> UserContext:
        user.require(capability)
        return user
    return _dep
