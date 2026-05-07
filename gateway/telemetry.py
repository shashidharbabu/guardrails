"""
Gateway observability: Datadog APM when available; optional stdout JSON mirrors.

- DD_TRACE_ENABLED: default true — use ddtrace spans when the package is installed.
- GATEWAY_TELEMETRY_STDOUT: default false — if true, also emit one-line JSON for
  span_start/span_end (useful when the Agent is down or for local piping).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_DDTRACE_ENABLED = os.getenv("DD_TRACE_ENABLED", "true").lower() in ("1", "true", "yes")
_STDOUT_MIRROR = os.getenv("GATEWAY_TELEMETRY_STDOUT", "false").lower() in (
    "1",
    "true",
    "yes",
)

_tracer = None
try:
    if _DDTRACE_ENABLED:
        from ddtrace import tracer as _tracer  # noqa: WPS433
except ImportError:
    _tracer = None


def ddtrace_active() -> bool:
    return _tracer is not None and _DDTRACE_ENABLED


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@contextmanager
def span(name: str, *, trace_id: Optional[str] = None, **tags: Any) -> Iterator[None]:
    tid = trace_id or str(uuid.uuid4())
    service = os.getenv("DD_SERVICE", "gateway")

    stdout = _STDOUT_MIRROR or not ddtrace_active()
    start_payload = {"event": "span_start", "component": "gateway", "name": name, "trace_id": tid}
    for k, v in tags.items():
        start_payload[k] = _json_safe(v)
    if stdout:
        logger.info(json.dumps(start_payload))

    if ddtrace_active():
        with _tracer.trace(name, service=service) as dd_span:
            dd_span.set_tag("trace_id", tid)
            env = os.getenv("DD_ENV")
            if env:
                dd_span.set_tag("env", env)
            for k, v in tags.items():
                dd_span.set_tag(k, _json_safe(v))
            yield
    else:
        yield

    if stdout:
        logger.info(json.dumps({"event": "span_end", "component": "gateway", "name": name, "trace_id": tid}))


def tag_current_span(**kwargs: Any) -> None:
    if not ddtrace_active():
        return
    try:
        current = _tracer.current_span()
        if current is None:
            return
        for k, v in kwargs.items():
            current.set_tag(k, _json_safe(v))
    except Exception:
        pass
