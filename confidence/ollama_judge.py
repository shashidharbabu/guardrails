"""
confidence/ollama_judge.py — DeepEvalBaseLLM adapter for local Ollama.

Wraps qwen2.5:7b (or any Ollama model) so DeepEval metrics can use it
as their evaluation judge instead of OpenAI GPT.

Usage:
    from confidence.ollama_judge import OllamaJudge
    judge = OllamaJudge(model="qwen2.5:7b")
    metric = FaithfulnessMetric(model=judge, threshold=0.5)
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
from deepeval.models.base_model import DeepEvalBaseLLM

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL:    str = os.getenv("VERIFIER_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT:  float = float(os.getenv("VERIFIER_TIMEOUT", "120"))


class OllamaJudge(DeepEvalBaseLLM):
    """
    DeepEval-compatible LLM wrapper for local Ollama.

    Implements both sync `generate()` and async `a_generate()` so
    DeepEval can run metrics in either mode.

    DeepEval passes a plain string prompt and expects a plain string
    response — the Ollama /api/generate endpoint is the simplest fit.
    """

    def __init__(
        self,
        model:    str   = OLLAMA_MODEL,
        base_url: str   = OLLAMA_BASE_URL,
        timeout:  float = OLLAMA_TIMEOUT,
    ) -> None:
        self.model    = model
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    # ── Required DeepEvalBaseLLM interface ────────────────────────────────────

    def load_model(self) -> Any:
        """DeepEval calls this to initialise the model. We return self."""
        return self

    def generate(self, prompt: str, *args, **kwargs) -> str:
        """Synchronous generation via Ollama /api/generate."""
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},  # deterministic for evaluation
        }
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        """Async generation via Ollama /api/generate."""
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

    def get_model_name(self) -> str:
        return f"ollama/{self.model}"
