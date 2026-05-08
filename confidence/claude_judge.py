"""
confidence/claude_judge.py — DeepEvalBaseLLM adapter for Anthropic Claude.

Drop-in replacement for OllamaJudge.  Set JUDGE_BACKEND=claude in .env
and provide ANTHROPIC_API_KEY to route DeepEval metrics through Claude
instead of the local qwen2.5:7b model.

Advantages over qwen2.5:7b for evaluation:
  - Graded scores (not binary 0/1 that the small model produces)
  - Stronger regulatory / legal reasoning
  - Faster per-call than loading a local 7B model

Usage:
    from confidence.claude_judge import ClaudeJudge
    judge = ClaudeJudge()
    metric = FaithfulnessMetric(model=judge, threshold=0.5)

Env vars:
    ANTHROPIC_API_KEY    required
    CLAUDE_JUDGE_MODEL   default: claude-haiku-4-5  (fast + cheap for eval)
    CLAUDE_JUDGE_TIMEOUT default: 60

DeepEval v0.21 schema protocol:
    DeepEval calls generate(prompt, schema=PydanticModel) when it needs
    structured output.  We detect the kwarg, append a JSON instruction to the
    prompt, ask Claude to reply with valid JSON only, then parse + return the
    Pydantic model instance.  Plain string calls (no schema) work as before.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Type

import anthropic
from deepeval.models.base_model import DeepEvalBaseLLM

def _default_model() -> str:
    try:
        from app.backend.config import get_settings
        return get_settings().CLAUDE_JUDGE_MODEL
    except Exception:
        return os.getenv("CLAUDE_JUDGE_MODEL", "claude-haiku-4-5-20251001")


def _default_timeout() -> float:
    try:
        from app.backend.config import get_settings
        return float(get_settings().CLAUDE_JUDGE_TIMEOUT)
    except Exception:
        return float(os.getenv("CLAUDE_JUDGE_TIMEOUT", "60"))


def _extract_json(text: str) -> str:
    """Extract the first JSON object or array from a string."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find first { or [
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        idx = text.find(start_char)
        if idx != -1:
            # Find the matching close bracket
            depth = 0
            for i, ch in enumerate(text[idx:], idx):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[idx:i + 1]
    return text


def _parse_with_schema(text: str, schema: type) -> Any:
    """Parse Claude's text response into the requested Pydantic schema."""
    raw = _extract_json(text)
    data = json.loads(raw)
    return schema(**data) if isinstance(data, dict) else schema.parse_raw(raw)


def _schema_instruction(schema: type) -> str:
    """Build a concise JSON-only instruction for the given Pydantic schema."""
    try:
        example = schema.schema()
        props = list(example.get("properties", {}).keys())
        fields_hint = ", ".join(f'"{p}"' for p in props)
        return (
            f"\n\nRespond with ONLY a valid JSON object containing these fields: "
            f"{fields_hint}. No markdown, no explanation, no extra text."
        )
    except Exception:
        return "\n\nRespond with ONLY a valid JSON object. No markdown, no explanation."


class ClaudeJudge(DeepEvalBaseLLM):
    """
    DeepEval-compatible LLM wrapper for Anthropic Claude.

    Implements both sync `generate()` and async `a_generate()` so
    DeepEval can run metrics in either mode.

    When DeepEval passes a `schema` kwarg (Pydantic model class), we append
    a JSON-only instruction to the prompt and parse the response into that
    schema before returning.
    """

    def __init__(
        self,
        model:   Optional[str]   = None,
        timeout: Optional[float] = None,
    ) -> None:
        model   = model   or _default_model()
        timeout = timeout or _default_timeout()
        self.model   = model
        self.timeout = timeout
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to .env or export it before using ClaudeJudge."
            )
        self._client       = anthropic.Anthropic(api_key=api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key)

    # ── Required DeepEvalBaseLLM interface ────────────────────────────────────

    def load_model(self) -> Any:
        return self

    def generate(self, prompt: str, schema: Optional[type] = None, *args, **kwargs) -> Any:
        """Synchronous generation via Anthropic Messages API.

        When `schema` is a Pydantic model class, appends a JSON instruction,
        calls Claude, and returns the parsed Pydantic instance.
        Otherwise returns the plain text response.
        """
        full_prompt = prompt
        if schema is not None:
            full_prompt = prompt + _schema_instruction(schema)

        message = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": full_prompt}],
        )
        text = message.content[0].text

        if schema is not None:
            return _parse_with_schema(text, schema)
        return text

    async def a_generate(self, prompt: str, schema: Optional[type] = None, *args, **kwargs) -> Any:
        """Async generation via Anthropic Messages API."""
        full_prompt = prompt
        if schema is not None:
            full_prompt = prompt + _schema_instruction(schema)

        message = await self._async_client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": full_prompt}],
        )
        text = message.content[0].text

        if schema is not None:
            return _parse_with_schema(text, schema)
        return text

    def get_model_name(self) -> str:
        return f"claude/{self.model}"
