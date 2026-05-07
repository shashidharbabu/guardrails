"""
Langfuse (self-hosted / cloud) instrumentation for debate MAD pipeline.

Uses an explicit Langfuse(...) client built from env vars so traces are not dropped by
get_client() multi-project edge cases. Normalizes quoted values from .env files.

Env:
  LANGFUSE_ENABLED — app-level gate (default true)
  LANGFUSE_TRACING_ENABLED — SDK gate (default true; same as Langfuse Python SDK)
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
  LANGFUSE_DEBUG — verbose SDK logs when true
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, MutableMapping, Optional

# Single client per process — avoids get_client() returning a disabled stub.
_langfuse_client: Any = None
_client_init_error: Optional[str] = None


def _strip_val(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    s = raw.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s


def enabled() -> bool:
    if os.getenv("LANGFUSE_ENABLED", "true").lower() not in ("1", "true", "yes"):
        return False
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() in (
        "0",
        "false",
        "no",
    ):
        return False
    return True


def _truthy_credentials() -> bool:
    pk = _strip_val(os.getenv("LANGFUSE_PUBLIC_KEY"))
    sk = _strip_val(os.getenv("LANGFUSE_SECRET_KEY"))
    base = _strip_val(os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST"))
    return bool(pk and sk and base)


def get_tracing_client():
    """
    Return the Langfuse client or None if misconfigured.
    Prefer this over langfuse.get_client() to avoid disabled multi-project stubs.
    """
    global _langfuse_client, _client_init_error
    if not enabled() or not _truthy_credentials():
        return None
    if _langfuse_client is not None:
        return _langfuse_client
    if _client_init_error is not None:
        return None
    try:
        from langfuse import Langfuse

        pk = _strip_val(os.getenv("LANGFUSE_PUBLIC_KEY"))
        sk = _strip_val(os.getenv("LANGFUSE_SECRET_KEY"))
        base = _strip_val(os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST"))
        base = base.rstrip("/")
        _langfuse_client = Langfuse(
            public_key=pk,
            secret_key=sk,
            base_url=base,
        )
        print(
            f"[Langfuse] Client initialized (base_url={base!r}, public_key_prefix={pk[:12]}...)",
            flush=True,
        )
        return _langfuse_client
    except Exception as exc:
        _client_init_error = str(exc)
        print(f"[Langfuse] Client init failed: {exc}", flush=True)
        return None


def observability_status() -> dict:
    """Safe diagnostics for GET /mad/observability/status (no secrets)."""
    base = _strip_val(os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST"))
    pk = _strip_val(os.getenv("LANGFUSE_PUBLIC_KEY"))
    host_hint = ""
    if base:
        if "://" in base:
            try:
                host_hint = base.split("//", 1)[1].split("/")[0]
            except IndexError:
                host_hint = base[:48]
        else:
            host_hint = base[:48]
    return {
        "langfuse_enabled": enabled(),
        "credentials_present": _truthy_credentials(),
        "base_url_host": host_hint,
        "public_key_prefix": (pk[:14] + "…") if len(pk) > 14 else pk,
        "client_ready": get_tracing_client() is not None,
        "last_init_error": _client_init_error,
        "try_ping": "POST /mad/observability/ping — then open Langfuse Traces (last 15 min)",
    }


@contextmanager
def observe(
    as_type: str,
    name: str,
    *,
    input_payload: Optional[Mapping[str, Any]] = None,
    **extra: Any,
) -> Iterator[Any]:
    """Create a Langfuse observation (span or generation). Yields observation or None."""
    client = get_tracing_client()
    if client is None:
        yield None
        return
    kwargs: MutableMapping[str, Any] = {"as_type": as_type, "name": name}
    kwargs.update(extra)
    if input_payload is not None:
        kwargs["input"] = dict(input_payload)
    try:
        with client.start_as_current_observation(**kwargs) as observation:
            yield observation
    except Exception as exc:
        print(f"[Langfuse] observation '{name}' skipped: {exc}", flush=True)
        yield None


def flush() -> None:
    client = get_tracing_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        print(f"[Langfuse] flush warning: {exc}", flush=True)


def shutdown() -> None:
    """Call on process exit so batches are exported (Docker / uvicorn reload)."""
    client = get_tracing_client()
    if client is None:
        return
    try:
        client.flush()
        if hasattr(client, "shutdown"):
            client.shutdown()
    except Exception:
        pass


def ping() -> dict:
    """Minimal trace + flush — use to verify Langfuse UI without running full MAD."""
    out: dict = {"ok": False, "detail": ""}
    with observe(
        "span",
        "mad.observability.ping",
        input_payload={"source": "manual_ping"},
    ) as obs:
        if obs is not None:
            try:
                obs.update(output={"pong": True})
            except Exception:
                pass
            out["ok"] = True
            out["detail"] = "observation_created"
        else:
            st = observability_status()
            if not st["credentials_present"]:
                out["detail"] = "missing_LANGFUSE_PUBLIC_KEY_SECRET_OR_BASE_URL"
            elif not st["langfuse_enabled"]:
                out["detail"] = "LANGFUSE_ENABLED_or_TRACING_disabled"
            elif not st["client_ready"]:
                out["detail"] = st.get("last_init_error") or "client_not_ready"
            else:
                out["detail"] = "observe_returned_none"
    flush()
    return out
