from __future__ import annotations
"""
RAG retrieval client.

TODO: The original implementation uses:
  - Qdrant vector store (must be running, default host.docker.internal:6333)
  - BM25 index (bm25_combined.pkl)
  - Dense embedder (retrieval_v2.EMBED_MODEL_NAME)
  - Hybrid re-ranking pipeline (retrieval_v2.RetrievalPipeline)

To enable live RAG retrieval:
  1. Set RAG_SERVICE_PATH env var to the directory containing rag_service.py and retrieval_v2.py
  2. Set BM25_PATH_OVERRIDE to the absolute path of bm25_combined.pkl
  3. Ensure Qdrant is running and reachable at QDRANT_HOST:QDRANT_PORT
  4. Set QDRANT_MODE=server (or local for embedded Qdrant)

For pipeline testing without Qdrant, use load_chunks_from_jsonl() to supply chunks directly.
"""

import json
import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)

_svc = None


def load_chunks_from_jsonl(path: str | Path) -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_chunks_from_json(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def _normalize_chunk(c: dict) -> dict:
    return {
        "chunk_id": c.get("chunk_id") or f"{c.get('source_file', 'unk')}_{c.get('rank', 0)}",
        "text": c.get("text", ""),
        "source_file": c.get("source_file", ""),
        "domain": c.get("domain", ""),
        "rerank_score": c.get("rerank_score", 0.0),
    }


async def retrieve(query: str) -> list[dict]:
    """
    Retrieve chunks for a query using the RAG service.

    Requires RAG_SERVICE_PATH env var pointing to the NewRAG directory.
    Falls back to empty list if RAG is not configured (use load_chunks_from_jsonl instead).
    """
    global _svc
    rag_service_path = os.getenv("RAG_SERVICE_PATH", "")
    if not rag_service_path:
        logger.warning(
            "RAG_SERVICE_PATH not set. Returning empty chunks. "
            "Set RAG_SERVICE_PATH or pass chunks directly to the pipeline."
        )
        return []

    if _svc is None:
        import sys
        sys.path.insert(0, rag_service_path)
        os.environ.setdefault("BASE_DIR", rag_service_path)
        os.environ.setdefault("BM25_PATH", os.getenv("BM25_PATH_OVERRIDE", ""))
        os.environ.setdefault("COLLECTION_NAME", "guardrails_rag_v2")
        os.environ.setdefault("QDRANT_MODE", os.getenv("QDRANT_MODE", "server"))
        os.environ.setdefault("QDRANT_HOST", os.getenv("QDRANT_HOST", "host.docker.internal"))
        os.environ.setdefault("QDRANT_PORT", os.getenv("QDRANT_PORT", "6333"))
        from rag_service import get_service
        _svc = get_service(verbose=False)

    result = _svc.retrieve_for_llm(query)
    return [_normalize_chunk(c) for c in result.get("chunks", [])]
