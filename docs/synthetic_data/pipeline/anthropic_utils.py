from __future__ import annotations

import json
import time
from typing import Any, Literal, Optional


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # remove leading ```lang? and trailing ```
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s[: -len("```")]
    return s.strip()


def _extract_json_substring(s: str) -> str:
    """
    Best-effort extraction for cases where the model returns extra text.
    We prefer the first top-level JSON array/object.
    """
    s = _strip_code_fences(s)
    first_obj = s.find("{")
    first_arr = s.find("[")
    if first_obj == -1 and first_arr == -1:
        return s

    if first_arr != -1 and (first_obj == -1 or first_arr < first_obj):
        start = first_arr
        end = s.rfind("]")
        return s[start : end + 1] if end != -1 else s[start:]

    start = first_obj
    end = s.rfind("}")
    return s[start : end + 1] if end != -1 else s[start:]


def loads_loose_json(s: str) -> Any:
    return json.loads(_extract_json_substring(s))


def call_claude_text(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None = None,
    retry_limit: int = 3,
    backoff_base_seconds: float = 1.0,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(retry_limit + 1):
        try:
            kwargs = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return resp.content[0].text.strip()
        except Exception as e:  # noqa: BLE001 - caller wants resilience in Colab
            last_err = e
            if attempt >= retry_limit:
                raise
            time.sleep(backoff_base_seconds * (2**attempt))
    raise last_err or RuntimeError("Claude call failed without exception")


def call_claude_json(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None = None,
    retry_limit: int = 3,
) -> Any:
    raw = call_claude_text(
        client=client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        retry_limit=retry_limit,
    )
    return loads_loose_json(raw)

