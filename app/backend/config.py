"""
Production settings module.
All runtime configuration comes from environment variables.
No defaults reference localhost in production — use APP_ENV to guard dev-only paths.
"""

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional


class AppEnv(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"


class Settings:
    # ── Core ──────────────────────────────────────────────────────────────────
    APP_ENV: AppEnv = AppEnv(os.environ.get("APP_ENV", "development"))
    APP_VERSION: str = os.environ.get("APP_VERSION", "1.0.0")
    DEPLOYMENT_REGION: str = os.environ.get("DEPLOYMENT_REGION", "local")

    # ── Database ──────────────────────────────────────────────────────────────
    # In production, set DATABASE_URL to a postgres:// connection string.
    # SQLite is only allowed when APP_ENV=development.
    DATABASE_URL: Optional[str] = os.environ.get("DATABASE_URL")
    SQLITE_DB_PATH: str = os.environ.get(
        "SQLITE_DB_PATH",
        str(Path(__file__).resolve().parent.parent / "app_sessions.db"),
    )

    @property
    def effective_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        if self.APP_ENV == AppEnv.production:
            raise RuntimeError(
                "DATABASE_URL must be set in production. "
                "SQLite is not permitted in production environments."
            )
        return f"sqlite:///{self.SQLITE_DB_PATH}"

    # ── Services ──────────────────────────────────────────────────────────────
    GATEWAY_URL: str = os.environ.get("GATEWAY_URL", "http://localhost:8080")
    LLM_PROVIDER_TYPE: str = os.environ.get("LLM_PROVIDER_TYPE", "ollama")
    LLM_PROVIDER_URL: str = os.environ.get(
        "LLM_PROVIDER_URL",
        os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    DEFAULT_LLM_MODEL: str = os.environ.get("DEFAULT_LLM_MODEL", "qwen2.5:7b")

    # ── MAD ───────────────────────────────────────────────────────────────────
    MAD_MODE: str = os.environ.get("MAD_MODE", "local")  # local | api | disabled
    MAD_API_URL: Optional[str] = os.environ.get("MAD_API_URL")
    MAD_TIMEOUT_SECONDS: int = int(os.environ.get("MAD_TIMEOUT_SECONDS", "600"))

    # ── Feedback Loop ─────────────────────────────────────────────────────────
    FEEDBACK_API_URL: str = os.environ.get("FEEDBACK_API_URL", "http://localhost:8002")

    # ── Queue (optional for production durable execution) ─────────────────────
    REDIS_URL: Optional[str] = os.environ.get("REDIS_URL")
    QUEUE_URL: Optional[str] = os.environ.get("QUEUE_URL")

    # ── CSE ───────────────────────────────────────────────────────────────────
    CSE_CONFIG_VERSION: str = os.environ.get("CSE_CONFIG_VERSION", "1.0.0")
    CSE_DELIVER_THRESHOLD: float = float(os.environ.get("CSE_DELIVER_THRESHOLD", "0.75"))
    CSE_HUMAN_REVIEW_THRESHOLD: float = float(os.environ.get("CSE_HUMAN_REVIEW_THRESHOLD", "0.45"))
    CSE_MAX_RETRY_COUNT: int = int(os.environ.get("CSE_MAX_RETRY_COUNT", "1"))

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Production: comma-separated list of allowed origins, e.g.
    # CORS_ALLOWED_ORIGINS=https://guardrails.company.com
    CORS_ALLOWED_ORIGINS: List[str] = [
        o.strip()
        for o in os.environ.get(
            "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:4173"
        ).split(",")
        if o.strip()
    ]

    # ── Auth / OIDC ───────────────────────────────────────────────────────────
    # Set DISABLE_AUTH=true ONLY for local development. Never in production.
    DISABLE_AUTH: bool = os.environ.get("DISABLE_AUTH", "true").lower() == "true"
    JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
    JWT_ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))
    OIDC_ISSUER_URL: Optional[str] = os.environ.get("OIDC_ISSUER_URL")
    OIDC_CLIENT_ID: Optional[str] = os.environ.get("OIDC_CLIENT_ID")
    OIDC_AUDIENCE: Optional[str] = os.environ.get("OIDC_AUDIENCE")

    # ── Observability ─────────────────────────────────────────────────────────
    ENABLE_AUDIT_LOGGING: bool = os.environ.get("ENABLE_AUDIT_LOGGING", "true").lower() == "true"
    ENABLE_LANGFUSE: bool = os.environ.get("ENABLE_LANGFUSE", "false").lower() == "true"
    LANGFUSE_HOST: Optional[str] = os.environ.get("LANGFUSE_HOST")
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

    # ── Data ─────────────────────────────────────────────────────────────────
    DATA_RETENTION_DAYS: int = int(os.environ.get("DATA_RETENTION_DAYS", "365"))

    # ── Multi-tenancy ─────────────────────────────────────────────────────────
    TENANT_MODE: str = os.environ.get("TENANT_MODE", "single")  # single | multi

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == AppEnv.production

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == AppEnv.development

    def validate_production(self):
        """Raise on startup if required production config is missing."""
        if not self.is_production:
            return
        errors = []
        if not self.DATABASE_URL:
            errors.append("DATABASE_URL required in production")
        if self.DISABLE_AUTH:
            errors.append("DISABLE_AUTH must not be true in production")
        if self.JWT_SECRET_KEY == "CHANGE_ME_IN_PRODUCTION":
            errors.append("JWT_SECRET_KEY must be set in production")
        if self.CORS_ALLOWED_ORIGINS == ["http://localhost:5173", "http://localhost:4173"]:
            errors.append("CORS_ALLOWED_ORIGINS must not use localhost in production")
        if errors:
            raise RuntimeError(
                "Production config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
