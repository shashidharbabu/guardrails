"""
rag/retriever.py — Real Qdrant retriever for the MAD pipeline.
==============================================================

DROP-IN REPLACEMENT for multi_agent/rag_stub.retrieve().

Interface is identical:
    retrieve(query: str, top_k: int) -> List[EvidenceChunk]

The only difference from rag_stub is what happens inside retrieve():
  - rag_stub: keyword TF-IDF search over a local JSONL file
  - this file: semantic vector search against Qdrant cloud

Everything that calls retrieve() — agent_a, agent_b, judge, debate_engine —
works without any changes. The interface contract is preserved exactly.

PAYLOAD FIELDS IN QDRANT (confirmed from notebook cell 11/16):
  chunk_id    — string e.g. "t0__iso__27001_2022_infosec__chunk_0000"
  doc_id      — string e.g. "t0__iso__27001_2022_infosec"
  tier        — string e.g. "T0", "T1"  (batch label — NOT authority ranking)
  chunk_index — int
  token_count — int
  chunk_method — string
  source_url  — string (direct URL)
  source      — dict {"document_name": "...", "pdf_url": "..."}
  text        — string (chunk content)

HOW TO ACTIVATE:
  In multi_agent/rag_stub.py, replace the retrieve() function body with:
    from rag.retriever import retrieve
  See the one-line swap instructions at the bottom of this file.
"""
from __future__ import annotations

from typing import List
from urllib.parse import urlparse

from qdrant_client import QdrantClient

from rag.config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_TIMEOUT,
    COLLECTION_NAME,
)
from rag.embedder import embed_query

# Import EvidenceChunk from multi_agent — same schema, no duplication
from multi_agent.models import EvidenceChunk

# ── Singleton Qdrant client ────────────────────────────────────────────────────
_client: QdrantClient = None


def _make_qdrant_client(url: str, api_key: str, timeout: int) -> QdrantClient:
    """
    Build a QdrantClient that works correctly for both Cloud (HTTPS/443)
    and self-hosted (HTTP/6333) instances.

    When `url` contains `https://`, QdrantClient's default port (6333) is wrong
    for Qdrant Cloud which expects port 443. Parsing explicitly avoids that.
    """
    parsed = urlparse(url)
    is_https = parsed.scheme == "https"
    host = parsed.hostname or url
    port = parsed.port or (443 if is_https else 6333)
    return QdrantClient(
        host=host,
        port=port,
        https=is_https,
        api_key=api_key or None,
        timeout=timeout,
        check_compatibility=False,  # suppress version-check 404 warning on Cloud
    )


def _get_client() -> QdrantClient:
    """Return the Qdrant client, creating it on first call."""
    global _client
    if _client is None:
        print(f"[RAG] Connecting to Qdrant at {QDRANT_URL} ...")
        _client = _make_qdrant_client(QDRANT_URL, QDRANT_API_KEY, QDRANT_TIMEOUT)
        print(f"[RAG] Connected. Collection: {COLLECTION_NAME}")
    return _client


def retrieve(query: str, top_k: int = 5) -> List[EvidenceChunk]:
    """
    Retrieve top_k regulatory chunks from Qdrant for the given query.

    Steps:
      1. Embed the query using Nemotron-8B WITH the instruction prefix
      2. Run cosine similarity search against the Qdrant collection
      3. Map results to EvidenceChunk objects

    This is a drop-in replacement for rag_stub.retrieve().
    The signature and return type are identical.

    Args:
        query  — the text to search for (claim text or user query)
        top_k  — number of chunks to return

    Returns:
        List[EvidenceChunk] ordered by relevance (most relevant first)
    """
    # Step 1: embed query with prefix
    query_vector = embed_query(query)

    # Step 2: search Qdrant
    client = _get_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points

    # Step 3: map to EvidenceChunk
    chunks: List[EvidenceChunk] = []
    for r in results:
        p = r.payload

        # source field is a dict in this corpus: {"document_name": ..., "pdf_url": ...}
        # We store document_name as the source string for display/citation purposes.
        # Falls back to source_url or doc_id if source dict is not present.
        source_obj = p.get("source", {})
        if isinstance(source_obj, dict):
            source_str = source_obj.get("document_name") or p.get("doc_id", "unknown")
        else:
            source_str = str(source_obj) if source_obj else p.get("doc_id", "unknown")

        chunks.append(EvidenceChunk(
            chunk_id=str(p.get("chunk_id", r.id)),
            text=p.get("text", ""),
            source=source_str,
            tier=1,   # tier field ("T0"/"T1") is a batch label, not authority ranking
                      # stored as 1 (neutral) — not used for filtering in MAD v0.1
        ))

    return chunks


def health_check() -> dict:
    """
    Verify Qdrant connection and collection are reachable.
    Call this at startup before running MAD to catch config errors early.

    Returns dict with status and collection info.
    """
    try:
        client = _get_client()
        info = client.get_collection(COLLECTION_NAME)
        return {
            "status": "ok",
            "collection": COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ── HOW TO ACTIVATE THIS IN MAD ───────────────────────────────────────────────
#
# In multi_agent/rag_stub.py, replace the entire retrieve() function with:
#
#     from rag.retriever import retrieve  # noqa: F401
#
# That single import makes rag_stub.retrieve point to this real implementation.
# agent_a, agent_b, judge, debate_engine call rag_stub.retrieve() unchanged.
# Nothing else needs to change anywhere in the MAD codebase.
