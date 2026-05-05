"""
rag/verifier.py — SLM retrieval verifier for the RAG pipeline.
===============================================================

Ports notebook Cells 18-26 (Qwen2.5-3B-Instruct verifier) to run via Ollama
instead of loading transformers locally. Ollama uses Metal GPU on Apple Silicon,
giving equivalent speed without a separate 3B model load.

WHAT IT DOES
------------
Given a user query and a list of retrieved candidate chunks from Qdrant, the
verifier uses an SLM to:

1. Select the best 3 chunks that directly support answering the query
2. Reject weak, redundant, or loosely-related chunks
3. Decide whether the evidence is sufficient to answer the query
4. Assign a confidence score 0.0–1.0
5. Produce a grounded summary

OUTPUT FORMAT (matches notebook exactly)
----------------------------------------
{
  "query": "...",
  "sufficient_context": true,
  "confidence": 0.0,
  "grounded_summary": "...",
  "top_chunks": [
    {
      "rank": 1,
      "chunk_id": "...",
      "doc_id": "...",
      "tier": "...",
      "document_name": "...",
      "source_url": "...",
      "pdf_url": "...",
      "vector_score": 0.0,
      "why_selected": "...",
      "text": "..."
    }
  ],
  "rejected_chunks": [
    { "chunk_id": "...", "reason": "..." }
  ]
}

FALLBACK BEHAVIOUR
------------------
If Ollama is unreachable or returns malformed JSON, `run_verifier()` returns a
best-effort fallback that selects the top-scored chunks by vector score and
marks sufficient_context=False. The pipeline always gets a usable result.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from rag.config import (
    OLLAMA_BASE_URL,
    VERIFIER_MODEL,
    VERIFIER_TIMEOUT,
    TOP_K_VERIFIED,
)

logger = logging.getLogger(__name__)

# ── Prompt template (exact match to final notebook version) ───────────────────

_VERIFIER_SYSTEM = "You are a strict JSON-only retrieval verifier."

_VERIFIER_PROMPT_TEMPLATE = """\
You are a retrieval verification engine for a regulatory, governance, policy, and security RAG pipeline.

Your job is to verify retrieved evidence, not just summarize it.

Tasks:
1. Read the user query
2. Review all retrieved chunks
3. Select the best {top_k} chunks that most directly support answering the query
4. Prefer evidence coverage across different aspects of the query
5. Avoid selecting redundant chunks if another chunk adds broader or complementary support
6. Mark weak, duplicate, loosely related, or unnecessary chunks as rejected
7. Decide whether the evidence is sufficient to answer the query
8. Assign a confidence score from 0.0 to 1.0
9. Return ONLY valid JSON

Rules:
- Use only the provided chunks
- Do not invent facts
- Prefer direct answer-supporting chunks
- Prefer stronger and more specific evidence
- Prefer better coverage over redundancy
- Every retrieved chunk that is not selected must appear in rejected_chunks with a reason
- If the evidence is weak, incomplete, or ambiguous, set sufficient_context to false
- Keep the output both human-readable and agent-consumable

Return JSON in exactly this format:

{{
  "query": "...",
  "sufficient_context": true,
  "confidence": 0.0,
  "grounded_summary": "...",
  "top_chunks": [
    {{
      "rank": 1,
      "chunk_id": "...",
      "doc_id": "...",
      "tier": "...",
      "document_name": "...",
      "source_url": "...",
      "pdf_url": "...",
      "vector_score": 0.0,
      "why_selected": "...",
      "text": "..."
    }}
  ],
  "rejected_chunks": [
    {{
      "chunk_id": "...",
      "reason": "..."
    }}
  ]
}}

User Query:
{query}

Retrieved Chunks:
{chunks_json}"""


# ── Public API ─────────────────────────────────────────────────────────────────

def build_verifier_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the verifier prompt from a query and prepared chunk dicts.

    retrieved_chunks should be the output of prepare_chunks_for_verifier().
    """
    return _VERIFIER_PROMPT_TEMPLATE.format(
        top_k=TOP_K_VERIFIED,
        query=query,
        chunks_json=json.dumps(retrieved_chunks, indent=2, ensure_ascii=False),
    )


def prepare_chunks_for_verifier(qdrant_results: list[Any]) -> list[dict]:
    """
    Convert Qdrant PointStruct results into dicts for the verifier prompt.

    Handles both Qdrant PointStruct objects (from qdrant_client) and plain
    dicts (for unit testing / fallback paths).
    """
    prepared = []
    for r in qdrant_results:
        if hasattr(r, "payload"):
            # Qdrant PointStruct
            p = r.payload
            score = float(r.score)
        elif isinstance(r, dict):
            p = r
            score = float(r.get("score", 0.0))
        else:
            continue

        source_obj = p.get("source", {})
        if not isinstance(source_obj, dict):
            source_obj = {}

        prepared.append({
            "chunk_id":      p.get("chunk_id", ""),
            "doc_id":        p.get("doc_id", ""),
            "tier":          p.get("tier", ""),
            "document_name": source_obj.get("document_name"),
            "source_url":    p.get("source_url", ""),
            "pdf_url":       source_obj.get("pdf_url"),
            "text":          p.get("text", ""),
            "vector_score":  round(score, 4),
        })
    return prepared


def extract_json_from_text(text: str) -> dict:
    """
    Robustly extract the first complete JSON object from model output.

    Handles leading/trailing prose, markdown code fences, and partial output.
    Raises ValueError if no valid JSON is found.
    """
    # Strip markdown code fences first
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    start = text.find("{")
    end   = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in verifier output")

    json_str = text[start : end + 1]
    return json.loads(json_str)


def run_verifier(
    query: str,
    qdrant_results: list[Any],
) -> dict:
    """
    Run the SLM verifier over retrieved Qdrant results.

    Sends the verifier prompt to Ollama and parses the structured JSON response.
    On any error (Ollama down, JSON parse failure, timeout) returns a
    best-effort fallback dict so the pipeline never hard-fails.

    Args:
        query:          The original user query text.
        qdrant_results: List of Qdrant PointStruct objects (or plain dicts).

    Returns:
        Verified output dict with top_chunks, rejected_chunks, confidence, etc.
    """
    retrieved_chunks = prepare_chunks_for_verifier(qdrant_results)

    if not retrieved_chunks:
        return _empty_result(query)

    prompt = build_verifier_prompt(query, retrieved_chunks)
    messages = [
        {"role": "system", "content": _VERIFIER_SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    try:
        raw = _call_ollama(messages)
        result = extract_json_from_text(raw)
        # Ensure required keys are present
        result.setdefault("query", query)
        result.setdefault("sufficient_context", False)
        result.setdefault("confidence", 0.0)
        result.setdefault("grounded_summary", "")
        result.setdefault("top_chunks", [])
        result.setdefault("rejected_chunks", [])
        return result

    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.warning("[Verifier] Ollama unreachable: %s — using score-based fallback", exc)
        return _score_fallback(query, retrieved_chunks)

    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("[Verifier] JSON parse failed: %s — using score-based fallback", exc)
        return _score_fallback(query, retrieved_chunks)

    except Exception as exc:
        logger.error("[Verifier] Unexpected error: %s", exc, exc_info=True)
        return _score_fallback(query, retrieved_chunks)


# ── Ollama call ────────────────────────────────────────────────────────────────

def _call_ollama(messages: list[dict]) -> str:
    """POST to Ollama's OpenAI-compatible chat completions endpoint."""
    url = OLLAMA_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model":       VERIFIER_MODEL,
        "messages":    messages,
        "temperature": 0.0,
        "stream":      False,
    }
    with httpx.Client(timeout=VERIFIER_TIMEOUT) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


# ── Fallback helpers ───────────────────────────────────────────────────────────

def _empty_result(query: str) -> dict:
    return {
        "query":             query,
        "sufficient_context": False,
        "confidence":        0.0,
        "grounded_summary":  "No chunks retrieved.",
        "top_chunks":        [],
        "rejected_chunks":   [],
    }


def _score_fallback(query: str, prepared_chunks: list[dict]) -> dict:
    """
    Best-effort fallback when the SLM verifier cannot run.

    Selects the top TOP_K_VERIFIED chunks by vector_score and marks the rest
    as rejected. sufficient_context=False signals to the caller that this
    is a degraded response.
    """
    sorted_chunks = sorted(
        prepared_chunks, key=lambda c: c.get("vector_score", 0.0), reverse=True
    )
    top    = sorted_chunks[:TOP_K_VERIFIED]
    bottom = sorted_chunks[TOP_K_VERIFIED:]

    top_chunks = [
        {
            "rank":          i + 1,
            "chunk_id":      c["chunk_id"],
            "doc_id":        c["doc_id"],
            "tier":          c["tier"],
            "document_name": c.get("document_name"),
            "source_url":    c["source_url"],
            "pdf_url":       c.get("pdf_url"),
            "vector_score":  c["vector_score"],
            "why_selected":  "selected by vector score (verifier fallback)",
            "text":          c["text"],
        }
        for i, c in enumerate(top)
    ]

    rejected = [
        {"chunk_id": c["chunk_id"], "reason": "below top-k cutoff (verifier fallback)"}
        for c in bottom
    ]

    return {
        "query":             query,
        "sufficient_context": False,  # conservative — verifier didn't confirm sufficiency
        "confidence":        round(top[0]["vector_score"] if top else 0.0, 4),
        "grounded_summary":  "Verifier unavailable — chunks selected by vector score only.",
        "top_chunks":        top_chunks,
        "rejected_chunks":   rejected,
    }
