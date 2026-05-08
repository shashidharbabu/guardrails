"""
check_qdrant.py — Verify Qdrant cloud connection and collection status.

Usage:
    cd multi_agent_debate
    python -m rag.check_qdrant

Requires QDRANT_API_KEY env var (or set in rag/config.py for dev).
"""
import json
import os
import sys
from pathlib import Path

# Ensure the rag package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import QDRANT_API_KEY, QDRANT_URL, COLLECTION_NAME
from rag.retriever import health_check, retrieve


def main() -> None:
    print(f"Qdrant URL  : {QDRANT_URL}")
    print(f"Collection  : {COLLECTION_NAME}")
    print(f"API key set : {'yes' if QDRANT_API_KEY else 'NO — set QDRANT_API_KEY env var'}")
    print()

    if not QDRANT_API_KEY:
        print("ERROR: QDRANT_API_KEY is not set. Export it and re-run.")
        sys.exit(1)

    print("Running health_check()...")
    result = health_check()
    print(json.dumps(result, indent=2))

    if result.get("status") != "ok":
        print("\nERROR: Qdrant connection failed. Check URL and API key.")
        sys.exit(1)

    points = result.get("points_count", 0)
    if points == 0:
        print("\nWARNING: Collection exists but has 0 points — ingestion may not have run.")
        sys.exit(1)

    print(f"\nCollection has {points:,} points. Running test retrieval...")
    chunks = retrieve("HIPAA encryption requirements for ePHI", top_k=3)
    print(f"Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        preview = c.text[:120].replace("\n", " ")
        print(f"  [{i}] {c.chunk_id} | {c.source}")
        print(f"       {preview}...")
    print("\nQdrant RAG is operational.")


if __name__ == "__main__":
    main()
