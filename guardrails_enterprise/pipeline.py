"""
guardrails_enterprise/pipeline.py — GuardrailsPipeline (library mode).

Runs the full pipeline in-process — no separate servers needed.
Ollama must be running locally (default: http://localhost:11434).

PIPELINE STAGES
---------------
1. Gateway      — PII / jailbreak / prompt-injection validation
                  (requires: pip install guardrails-enterprise[gateway])
2. LLM          — Ollama chat completion
3. MAD          — Multi-Agent Debate output verification
4. CSE          — Confidence Scoring Engine (DeepEval + MAD judge aggregate)
                  (DeepEval requires: pip install guardrails-enterprise[eval])

USAGE
-----
    from guardrails_enterprise import GuardrailsPipeline

    # Minimal — Ollama running locally, no RAG
    pipe = GuardrailsPipeline()
    result = pipe.run("What does HIPAA require for PHI encryption?")
    print(result.routing)          # DELIVER
    print(result.confidence)       # 0.87
    print(result.cse)              # CSEResult(...)

    # Full config
    pipe = GuardrailsPipeline(
        ollama_url="http://localhost:11434",
        ollama_model="qwen2.5:7b",
        skip_gateway=False,
        skip_deepeval=False,
    )

IMPORTS
-------
Gateway validators, MAD pipeline, and CSE are all lazy-imported so the SDK
is importable even if optional extras are not installed. Errors only surface
when you actually call run().
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from guardrails_enterprise.types import (
    CSEResult,
    GatewayResult,
    MADSummary,
    PipelineResult,
)

logger = logging.getLogger(__name__)

# Ensure repo root is on sys.path so internal packages resolve correctly
# when installed in editable mode (pip install -e .).
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "multi_agent_debate"), str(_REPO_ROOT / "rag_folder")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("VERIFIER_MODEL",    "qwen2.5:7b")
SYSTEM_PROMPT    = (
    "You are an enterprise compliance assistant. Answer questions about regulatory "
    "requirements accurately and concisely based on your knowledge of healthcare, "
    "banking, and legal regulations. Be concise — 2-4 sentences maximum."
)


class GuardrailsPipeline:
    """
    Full end-to-end guardrails pipeline running entirely in-process.

    Parameters
    ----------
    ollama_url      : Base URL for the Ollama server.
    ollama_model    : Ollama model to use for LLM generation.
    system_prompt   : System prompt sent to the LLM.
    skip_gateway    : If True, skip PII/jailbreak/PI validation entirely.
    skip_deepeval   : If True, CSE runs in judge-only v0.1 mode (no DeepEval).
    gateway_kwargs  : Extra kwargs forwarded to GuardrailGateway.__init__().
    """

    def __init__(
        self,
        ollama_url:     str  = OLLAMA_BASE_URL,
        ollama_model:   str  = OLLAMA_MODEL,
        system_prompt:  str  = SYSTEM_PROMPT,
        skip_gateway:   bool = False,
        skip_deepeval:  bool = False,
        **gateway_kwargs,
    ) -> None:
        self.ollama_url    = ollama_url.rstrip("/")
        self.ollama_model  = ollama_model
        self.system_prompt = system_prompt
        self.skip_gateway  = skip_gateway
        self.skip_deepeval = skip_deepeval
        self._gateway_kwargs = gateway_kwargs
        self._gateway = None   # lazy-init

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, query: str) -> PipelineResult:
        """
        Run the full pipeline for a single query.

        1. Gateway validation (unless skip_gateway=True)
        2. LLM generation via Ollama
        3. Multi-Agent Debate (MAD) verification
        4. Confidence Scoring Engine (CSE)

        Returns a PipelineResult with all intermediate results attached.
        """
        if not query.strip():
            raise ValueError("query must not be empty")

        t0 = time.time()

        # ── Step 1: Gateway ───────────────────────────────────────────────────
        gateway_result = self._run_gateway(query)
        logger.info("[Pipeline] Gateway: %s (score=%.3f)", gateway_result.decision, gateway_result.gateway_score)

        if not gateway_result.is_allowed:
            logger.info("[Pipeline] BLOCKED by gateway — skipping LLM + MAD")
            return PipelineResult(query=query, gateway=gateway_result)

        # ── Step 2: LLM ───────────────────────────────────────────────────────
        llm_answer = self._call_llm(query)
        logger.info("[Pipeline] LLM answer: %s...", llm_answer[:80])

        if not llm_answer:
            return PipelineResult(query=query, gateway=gateway_result)

        # ── Step 3 + 4: MAD + CSE (integrated) ───────────────────────────────
        mad_output = self._run_mad(query, llm_answer)

        if mad_output is None:
            return PipelineResult(
                query=query, gateway=gateway_result, llm_answer=llm_answer
            )

        mad_summary = MADSummary.from_mad_output(mad_output)
        cse = CSEResult.from_dict(mad_output.cse_result) if mad_output.cse_result else None

        elapsed = round((time.time() - t0) * 1000)
        logger.info(
            "[Pipeline] Done in %dms — routing=%s confidence=%.4f",
            elapsed, mad_summary.routing_decision, mad_summary.aggregate_confidence,
        )

        return PipelineResult(
            query=query,
            gateway=gateway_result,
            llm_answer=llm_answer,
            mad=mad_summary,
            cse=cse,
        )

    # ── Gateway ───────────────────────────────────────────────────────────────

    def _run_gateway(self, query: str) -> GatewayResult:
        if self.skip_gateway:
            return GatewayResult(
                decision="PASS", gateway_score=0.0,
                pii_score=0.0, jb_score=0.0, pi_score=0.0,
            )
        try:
            gateway = self._get_gateway()
            raw = gateway.process(query)
            return GatewayResult.from_gateway_result(raw)
        except ImportError as exc:
            logger.warning(
                "[Pipeline] Gateway not available (%s). "
                "Install with: pip install guardrails-enterprise[gateway]\n"
                "Treating as PASS.", exc
            )
            return GatewayResult(
                decision="PASS", gateway_score=0.0,
                pii_score=0.0, jb_score=0.0, pi_score=0.0,
            )

    def _get_gateway(self):
        if self._gateway is None:
            from gateway.gateway import GuardrailGateway
            self._gateway = GuardrailGateway(**self._gateway_kwargs)
        return self._gateway

    # ── LLM ──────────────────────────────────────────────────────────────────

    def _call_llm(self, query: str) -> Optional[str]:
        import httpx

        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": query},
            ],
            "stream":     False,
            "max_tokens": 512,
        }
        try:
            resp = httpx.post(
                f"{self.ollama_url}/v1/chat/completions",
                json=payload,
                timeout=300.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("[Pipeline] LLM call failed: %s", exc)
            return None

    # ── MAD + CSE ─────────────────────────────────────────────────────────────

    def _run_mad(self, query: str, llm_answer: str):
        try:
            from multi_agent.mad_pipeline import run_mad

            if self.skip_deepeval:
                # Monkey-patch ConfidenceScorer to skip DeepEval calls
                _patch_skip_deepeval()

            return run_mad(query, llm_answer)
        except ImportError as exc:
            logger.error(
                "[Pipeline] MAD pipeline not available: %s\n"
                "Ensure multi_agent_debate/ is on your path or install the full SDK.", exc
            )
            return None
        except Exception as exc:
            logger.error("[Pipeline] MAD pipeline error: %s", exc, exc_info=True)
            return None


# ── Convenience function ──────────────────────────────────────────────────────

def run_pipeline(
    query:          str,
    ollama_url:     str  = OLLAMA_BASE_URL,
    ollama_model:   str  = OLLAMA_MODEL,
    skip_gateway:   bool = False,
    skip_deepeval:  bool = False,
    **gateway_kwargs,
) -> PipelineResult:
    """
    Run the full guardrails pipeline in one call.

    Convenience wrapper around GuardrailsPipeline — creates a fresh pipeline
    instance and runs it for a single query.

    Parameters
    ----------
    query         : User query to process.
    ollama_url    : Ollama server URL (default: http://localhost:11434).
    ollama_model  : LLM model name (default: qwen2.5:7b).
    skip_gateway  : Skip PII/jailbreak validation (useful for testing).
    skip_deepeval : Use judge-only v0.1 CSE mode (no DeepEval LLM calls).
    **gateway_kwargs : Extra args forwarded to GuardrailGateway.

    Returns
    -------
    PipelineResult

    Example
    -------
        from guardrails_enterprise import run_pipeline

        result = run_pipeline("What does GDPR Article 17 require?")
        print(result.routing)     # DELIVER / RETRY / HARD_BLOCK / HUMAN_REVIEW
        print(result.confidence)  # 0.87
        print(result.cse)         # CSEResult(...)
    """
    pipe = GuardrailsPipeline(
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        skip_gateway=skip_gateway,
        skip_deepeval=skip_deepeval,
        **gateway_kwargs,
    )
    return pipe.run(query)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _patch_skip_deepeval() -> None:
    """
    Monkey-patch ConfidenceScorer._run_deepeval to return neutral defaults.
    Called when skip_deepeval=True to avoid DeepEval/Ollama calls for CSE.
    """
    try:
        from confidence.scorer import ConfidenceScorer, _DEFAULT_F_LLM, _DEFAULT_H_LLM, _DEFAULT_RELEVANCY

        def _neutral_deepeval(self, *args, **kwargs):
            return _DEFAULT_F_LLM, _DEFAULT_H_LLM, _DEFAULT_RELEVANCY, "skip_deepeval", "v0.1"

        ConfidenceScorer._run_deepeval = _neutral_deepeval
    except ImportError:
        pass
