"""
guardrails_enterprise.pipeline — High-level SDK pipeline.

This is the primary entry point for most SDK users.

Quickstart::

    import asyncio
    from guardrails_enterprise import GuardrailPipeline, SDKConfig

    cfg = SDKConfig(llm_model="qwen2.5:7b")
    pipeline = GuardrailPipeline(config=cfg)

    result = asyncio.run(pipeline.run("Does HIPAA require AES-256 encryption?"))
    print(result.gateway.decision)     # "PASS"
    print(result.llm_answer)           # LLM response
    print(result.mad.routing_decision) # "DELIVER" (after MAD completes)

Blocking-only use (no LLM)::

    gw_result = asyncio.run(pipeline.check("Ignore all previous instructions..."))
    print(gw_result.decision)  # "BLOCK"
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx

from guardrails_enterprise.config import SDKConfig
from guardrails_enterprise.exceptions import (
    GatewayBlockedError,
    GatewayConnectionError,
    LLMError,
    MADError,
    RAGError,
)
from guardrails_enterprise.types import GatewayResult, MADResult, PipelineResult, RAGResult

logger = logging.getLogger(__name__)

_mad_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mad-worker")


class GuardrailPipeline:
    """
    End-to-end enterprise guardrails pipeline.

    Wraps:
      1. Input Gateway  — blocks threats, PII, jailbreaks
      2. LLM call       — Ollama OpenAI-compatible endpoint
      3. MAD pipeline   — verifies LLM output against regulatory corpus
    """

    def __init__(self, config: Optional[SDKConfig] = None):
        self.config = config or SDKConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, query: str) -> PipelineResult:
        """
        Run the full pipeline: gateway → LLM → MAD.

        Returns a PipelineResult immediately after gateway + LLM complete.
        If `config.mad_in_background` is True (default), MAD runs as an asyncio
        background task and the `mad` field will be None until it finishes
        (suitable for the app backend which polls/updates the DB separately).

        If `config.mad_in_background` is False, this coroutine awaits MAD and
        the returned `mad` field is populated.

        Raises:
            GatewayBlockedError: if `config.raise_on_block` is True and decision is BLOCK.
            LLMError:            if the LLM call fails.
            GatewayConnectionError: if the gateway service is unreachable.
        """
        session_id = str(uuid.uuid4())
        t0 = time.time()

        # 1. Gateway
        gw_result = await self.check(query)

        if gw_result.decision == "BLOCK" and self.config.raise_on_block:
            raise GatewayBlockedError(
                decision=gw_result.decision,
                score=gw_result.gateway_score,
                reason=gw_result.blocked_reason,
            )

        # 2. LLM (skip if blocked)
        llm_answer: Optional[str] = None
        if gw_result.allowed:
            try:
                llm_answer = await self._call_llm(query)
            except Exception as exc:
                raise LLMError(str(exc)) from exc

        duration_ms = int((time.time() - t0) * 1000)

        # 3. MAD
        mad_result: Optional[MADResult] = None
        if self.config.run_mad and llm_answer:
            if self.config.mad_in_background:
                asyncio.create_task(
                    self._run_mad_background(session_id, query, llm_answer)
                )
            else:
                mad_result = await self._run_mad(query, llm_answer)

        return PipelineResult(
            session_id=session_id,
            query=query,
            gateway=gw_result,
            llm_answer=llm_answer,
            mad=mad_result,
            duration_ms=duration_ms,
        )

    async def check(self, text: str) -> GatewayResult:
        """
        Run only the input gateway check (no LLM, no MAD).

        Useful for pre-screening before routing to any downstream service.

        Raises:
            GatewayConnectionError: if the gateway service is unreachable.
        """
        try:
            async with httpx.AsyncClient(timeout=self.config.gateway_timeout_s) as client:
                r = await client.post(
                    f"{self.config.gateway_url}/validate",
                    json={"text": text},
                )
                r.raise_for_status()
                return GatewayResult.from_api(r.json())
        except httpx.ConnectError as exc:
            raise GatewayConnectionError(
                f"Gateway unreachable at {self.config.gateway_url}: {exc}"
            ) from exc

    async def verify(self, query: str, llm_answer: str) -> MADResult:
        """
        Run only the MAD output verification (no gateway, no LLM call).

        Useful when you already have an LLM answer and want to verify it.

        Raises:
            MADError: if the MAD pipeline fails.
        """
        return await self._run_mad(query, llm_answer)

    async def retrieve(self, query: str) -> RAGResult:
        """
        Run RAG retrieval + SLM verification for a query.

        Embeds the query with Nemotron-8B, retrieves the top candidates from
        Qdrant, then runs the Qwen SLM verifier to select the best chunks and
        assess evidence sufficiency.

        Use this standalone when you want to inspect what evidence the corpus
        contains before running the full pipeline, or to ground your own prompts.

        Example::

            rag = await pipeline.retrieve("Does HIPAA require AES-256 encryption?")
            print(rag.sufficient_context)  # True / False
            print(rag.confidence)          # 0.73
            print(rag.grounded_summary)    # "HIPAA addresses encryption..."
            for chunk in rag.top_chunks:
                print(chunk["chunk_id"], chunk["why_selected"])

        Raises:
            RAGError: if Qdrant is unreachable and the TF-IDF fallback also fails.
        """
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                _mad_executor,
                _run_rag_sync,
                query,
                self.config.rag_top_k,
            )
            return RAGResult.from_pipeline(result)
        except Exception as exc:
            raise RAGError(f"RAG pipeline error: {exc}") from exc

    async def retrieve_and_verify(
        self,
        query: str,
        llm_answer: str,
    ) -> tuple[RAGResult, MADResult]:
        """
        Retrieve evidence for a query AND verify an existing LLM answer.

        Combines RAG retrieval and MAD verification into a single call.
        Useful for batch offline evaluation or when you have a pre-generated answer.

        Returns:
            (RAGResult, MADResult) tuple — both completed before returning.

        Example::

            rag, mad = await pipeline.retrieve_and_verify(
                "Does HIPAA require AES-256?",
                "HIPAA mandates AES-256 encryption for all ePHI.",
            )
            print(rag.sufficient_context)      # True
            print(mad.routing_decision)        # "HARD_BLOCK" — fabricated mandate
        """
        rag_task = asyncio.create_task(self.retrieve(query))
        mad_task = asyncio.create_task(self.verify(query, llm_answer))
        rag_result, mad_result = await asyncio.gather(rag_task, mad_task)
        return rag_result, mad_result

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _call_llm(self, query: str) -> str:
        chat_url = self.config.ollama_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model":      self.config.llm_model,
            "messages": [
                {"role": "system", "content": self.config.llm_system_prompt},
                {"role": "user",   "content": query},
            ],
            "stream":     False,
            "max_tokens": self.config.llm_max_tokens,
        }
        async with httpx.AsyncClient(timeout=self.config.llm_timeout_s) as client:
            r = await client.post(chat_url, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def _run_mad(self, query: str, llm_answer: str) -> MADResult:
        loop = asyncio.get_event_loop()
        try:
            mad_out = await loop.run_in_executor(
                _mad_executor, _run_mad_sync, query, llm_answer
            )
            if mad_out is None:
                raise MADError("MAD pipeline returned None")
            return MADResult.from_mad_output(mad_out)
        except MADError:
            raise
        except Exception as exc:
            raise MADError(f"MAD pipeline error: {exc}") from exc

    async def _run_mad_background(self, session_id: str, query: str, llm_answer: str):
        try:
            result = await self._run_mad(query, llm_answer)
            logger.info(
                "MAD complete [session=%s routing=%s confidence=%.3f]",
                session_id, result.routing_decision, result.aggregate_confidence,
            )
        except MADError as exc:
            logger.error("MAD background task failed [session=%s]: %s", session_id, exc)


def _run_mad_sync(query: str, llm_answer: str):
    """Thread-pool worker: imports and runs the MAD pipeline synchronously."""
    try:
        from multi_agent.mad_pipeline import run_mad  # noqa: PLC0415
        return run_mad(query, llm_answer)
    except Exception as exc:
        logger.error("MAD sync worker error: %s", exc, exc_info=True)
        return None


def _run_rag_sync(query: str, top_k: int) -> dict:
    """
    Thread-pool worker: imports and runs the RAG pipeline synchronously.

    Tries the full Qdrant + SLM verifier pipeline first.
    On any failure returns a best-effort dict from the TF-IDF fallback.
    """
    try:
        from rag.pipeline import run_rag_pipeline  # noqa: PLC0415
        return run_rag_pipeline(query, top_k=top_k)
    except Exception as exc:
        logger.warning("RAG full pipeline failed (%s) — using TF-IDF fallback", exc)

    # TF-IDF fallback: return minimal dict that RAGResult.from_pipeline() handles
    try:
        from multi_agent.rag_stub import retrieve  # noqa: PLC0415
        chunks = retrieve(query, top_k=top_k)
        top_chunks = [
            {
                "rank":       i + 1,
                "chunk_id":   c.chunk_id,
                "text":       c.text,
                "doc_id":     c.source,
                "why_selected": "TF-IDF fallback",
                "vector_score": 0.0,
            }
            for i, c in enumerate(chunks)
        ]
        return {
            "query":               query,
            "retrieved_count":     len(top_chunks),
            "retrieved_candidates": top_chunks,
            "verified_output": {
                "sufficient_context": False,
                "confidence":         0.0,
                "grounded_summary":   "Qdrant unavailable — TF-IDF fallback used.",
                "top_chunks":         top_chunks,
                "rejected_chunks":    [],
            },
        }
    except Exception as exc2:
        logger.error("RAG TF-IDF fallback also failed: %s", exc2, exc_info=True)
        return {
            "query":               query,
            "retrieved_count":     0,
            "retrieved_candidates": [],
            "verified_output": {
                "sufficient_context": False,
                "confidence":         0.0,
                "grounded_summary":   "RAG pipeline and fallback both failed.",
                "top_chunks":         [],
                "rejected_chunks":    [],
            },
        }
