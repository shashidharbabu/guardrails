"""
rag/pipeline.py — Full RAG pipeline: Qdrant retrieval → SLM verification.
==========================================================================

This module is the canonical entry point for all RAG calls in the system.
It combines:
  1. Qdrant semantic search via nvidia/llama-embed-nemotron-8b
  2. Qwen SLM verifier (via Ollama) that selects + ranks the best chunks

PUBLIC API
----------

    run_rag_pipeline(query, top_k=7, tier=None) -> dict
        Full pipeline. Returns a rich dict with candidates, verified output,
        and metadata. Use this for evaluation, debugging, and the app backend.

    build_agent_handoff(rag_result) -> dict
        Formats run_rag_pipeline() output for MAD agent consumption.

    retrieve_verified(query, top_k=7) -> List[EvidenceChunk]
        Drop-in replacement for rag_stub.retrieve(). Returns only the
        verified top chunks as EvidenceChunk objects.
        Falls back gracefully to TF-IDF stub if Qdrant/verifier unavailable.

FLOW
----
    Query
      → embed_query()  [Nemotron-8B, instruction prefix]
      → Qdrant cosine search  [top_k=7 candidates]
      → run_verifier()  [Qwen SLM selects top 3 + produces JSON]
      → List[EvidenceChunk]  [returned to MAD agents]
"""
from __future__ import annotations

import logging
from typing import List, Optional

from rag.config import TOP_K_RETRIEVE, TOP_K_VERIFIED

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_rag_pipeline(
    query: str,
    top_k: int = TOP_K_RETRIEVE,
    tier: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> dict:
    """
    Execute the full RAG pipeline for a query.

    Steps:
      1. Retrieve top_k candidate chunks from Qdrant
      2. Run SLM verifier to select the best chunks and assess sufficiency
      3. Return a structured result dict

    Args:
        query:  User query or claim text.
        top_k:  Number of candidates to retrieve from Qdrant before verification.
        tier:   Optional tier filter for Qdrant (e.g. "T0"). Usually None.
        doc_id: Optional doc_id filter. Usually None.

    Returns:
        {
          "query": str,
          "retrieved_count": int,
          "retrieved_candidates": List[dict],   # all Qdrant hits with metadata
          "verified_output": {                  # SLM verifier JSON
              "sufficient_context": bool,
              "confidence": float,
              "grounded_summary": str,
              "top_chunks": List[dict],
              "rejected_chunks": List[dict],
          }
        }
    """
    # Step 1: Qdrant retrieval
    qdrant_results = _qdrant_retrieve(query, top_k=top_k, tier=tier, doc_id=doc_id)

    # Step 2: SLM verification
    from rag.verifier import run_verifier
    verified_output = run_verifier(query, qdrant_results)

    # Step 3: Build candidate list for the result dict
    retrieved_candidates = _format_candidates(qdrant_results)

    return {
        "query":               query,
        "retrieved_count":     len(qdrant_results),
        "retrieved_candidates": retrieved_candidates,
        "verified_output":     verified_output,
    }


def build_agent_handoff(rag_result: dict) -> dict:
    """
    Format run_rag_pipeline() output for consumption by MAD agents.

    Returns the canonical agent handoff format from the notebook:
    {
      "query": str,
      "ground_truth": {
        "sufficient_context": bool,
        "confidence": float,
        "grounded_summary": str,
        "top_chunks": List[dict],
        "rejected_chunks": List[dict],
      }
    }
    """
    verified = rag_result["verified_output"]
    return {
        "query": rag_result["query"],
        "ground_truth": {
            "sufficient_context": verified.get("sufficient_context", False),
            "confidence":         verified.get("confidence", 0.0),
            "grounded_summary":   verified.get("grounded_summary", ""),
            "top_chunks":         verified.get("top_chunks", []),
            "rejected_chunks":    verified.get("rejected_chunks", []),
        },
    }


def retrieve_verified(
    query: str,
    top_k: int = TOP_K_RETRIEVE,
) -> "List[EvidenceChunk]":
    """
    Drop-in replacement for rag_stub.retrieve().

    Runs the full pipeline (Qdrant → SLM verifier) and returns only the
    verified top chunks as EvidenceChunk objects, ready for MAD agents.

    Falls back to TF-IDF stub (via rag_stub) if:
      - Qdrant env vars are not set
      - qdrant-client or transformers are not installed
      - Ollama is unreachable (verifier uses score-based fallback internally)

    Args:
        query:  Claim text or user query.
        top_k:  Qdrant candidate count before verification (default: TOP_K_RETRIEVE).

    Returns:
        List[EvidenceChunk] — top verified chunks (at most TOP_K_VERIFIED).
    """
    try:
        rag_result = run_rag_pipeline(query, top_k=top_k)
        return _verified_to_evidence_chunks(rag_result)
    except Exception as exc:
        logger.warning(
            "[RAG pipeline] Full pipeline failed (%s) — delegating to TF-IDF stub.", exc
        )
        return _tfidf_fallback(query, top_k)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _qdrant_retrieve(
    query: str,
    top_k: int,
    tier: Optional[str],
    doc_id: Optional[str],
) -> list:
    """
    Run Qdrant vector search with optional payload filters.

    Returns raw Qdrant PointStruct objects (or empty list on failure).
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    from urllib.parse import urlparse

    from rag.config import (
        QDRANT_URL, QDRANT_API_KEY, QDRANT_TIMEOUT, COLLECTION_NAME,
    )
    from rag.embedder import embed_query

    # Parse URL explicitly so Cloud (https/443) and self-hosted (http/6333)
    # both get the right port — QdrantClient defaults to 6333 even for https URLs.
    _parsed = urlparse(QDRANT_URL)
    _is_https = _parsed.scheme == "https"
    client = QdrantClient(
        host=_parsed.hostname or QDRANT_URL,
        port=_parsed.port or (443 if _is_https else 6333),
        https=_is_https,
        api_key=QDRANT_API_KEY or None,
        timeout=QDRANT_TIMEOUT,
        check_compatibility=False,
    )
    query_vector = embed_query(query)

    # Build optional filter
    search_filter = None
    conditions = []
    if tier:
        conditions.append(rest.FieldCondition(key="tier", match=rest.MatchValue(value=tier)))
    if doc_id:
        conditions.append(rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=doc_id)))
    if conditions:
        search_filter = rest.Filter(must=conditions)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=search_filter,
    ).points

    return results


def _format_candidates(qdrant_results: list) -> list[dict]:
    """Convert Qdrant PointStructs to plain dicts for serialisation."""
    out = []
    for r in qdrant_results:
        p = r.payload
        source_obj = p.get("source", {})
        if not isinstance(source_obj, dict):
            source_obj = {}
        out.append({
            "chunk_id":      p.get("chunk_id", ""),
            "doc_id":        p.get("doc_id", ""),
            "tier":          p.get("tier", ""),
            "source_url":    p.get("source_url", ""),
            "document_name": source_obj.get("document_name"),
            "pdf_url":       source_obj.get("pdf_url"),
            "vector_score":  round(float(r.score), 4),
            "text":          p.get("text", ""),
        })
    return out


def _verified_to_evidence_chunks(rag_result: dict) -> "List[EvidenceChunk]":
    """
    Convert the verifier's top_chunks list to EvidenceChunk Pydantic objects.

    Imports EvidenceChunk from multi_agent.models (canonical schema).
    Falls back to rag.retriever's raw chunks if top_chunks is empty.
    """
    # Lazy import to avoid circular dependency — pipeline can run standalone
    try:
        from multi_agent.models import EvidenceChunk
    except ImportError:
        # If multi_agent is not on sys.path, use a simple dataclass fallback
        from dataclasses import dataclass

        @dataclass
        class EvidenceChunk:  # type: ignore[no-redef]
            chunk_id: str
            text: str
            source: str
            tier: int = 1

    verified = rag_result["verified_output"]
    top_chunks = verified.get("top_chunks", [])

    if not top_chunks:
        # Verifier rejected all chunks — fall back to raw Qdrant candidates
        candidates = rag_result.get("retrieved_candidates", [])
        top_chunks = [
            {
                "chunk_id":      c.get("chunk_id", ""),
                "doc_id":        c.get("doc_id", ""),
                "document_name": c.get("document_name", ""),
                "text":          c.get("text", ""),
            }
            for c in candidates[:TOP_K_VERIFIED]
        ]

    evidence = []
    for chunk in top_chunks:
        source_str = (
            chunk.get("document_name")
            or chunk.get("doc_id")
            or chunk.get("chunk_id", "unknown")
        )
        evidence.append(EvidenceChunk(
            chunk_id=chunk.get("chunk_id", ""),
            text=chunk.get("text", ""),
            source=source_str,
            tier=1,  # tier labels in corpus are batch labels, not authority levels
        ))

    return evidence


def _tfidf_fallback(query: str, top_k: int) -> "List[EvidenceChunk]":
    """Delegate to rag_stub TF-IDF when Qdrant pipeline is unavailable."""
    try:
        from multi_agent.rag_stub import _load_chunks, _tokenize, _compute_idf, _to_evidence
        import math

        chunks = _load_chunks()
        if not chunks:
            return []

        query_terms = _tokenize(query)
        idf = _compute_idf(chunks, query_terms)
        scored = []
        for i, chunk in enumerate(chunks):
            from multi_agent.rag_stub import _get_text
            text        = _get_text(chunk)
            chunk_terms = _tokenize(text)
            if not chunk_terms:
                continue
            score = sum(
                (chunk_terms.count(t) / len(chunk_terms)) * idf.get(t, 0.0)
                for t in query_terms
            )
            if score > 0:
                scored.append((score, _to_evidence(chunk, i)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ec for _, ec in scored[:top_k]] or [
            _to_evidence(c, i) for i, c in enumerate(chunks[:top_k])
        ]
    except Exception as exc:
        logger.error("[RAG pipeline] TF-IDF fallback also failed: %s", exc)
        return []
