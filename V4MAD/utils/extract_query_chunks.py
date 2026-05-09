"""
One-time utility: extract query-level top-5 chunks from the old MAD v3 DB.

Run this ONCE before starting the step pipeline to create:
  data/queries_50.json       — 50 query_id + user_query pairs
  data/query_chunks_50.json  — query_id → top-5 stored chunks

Source: mad_before_phase1_5090_ragfix_01_.db (or any MAD v3 DB with rag_chunks column)

Run: python utils/extract_query_chunks.py
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OLD_DB = Path(
    "/content/drive/MyDrive/V4MAD/mad_before_phase1_5090_ragfix_01_.db"
)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    if not OLD_DB.exists():
        print(f"[ERROR] Old DB not found: {OLD_DB}")
        print("Update OLD_DB path in this script to point to your MAD v3 database.")
        return

    DATA_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(OLD_DB)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT query_id, user_query, rag_chunks FROM queries ORDER BY query_id"
    ).fetchall()

    queries      = []
    query_chunks = []

    for row in rows:
        query_id   = row["query_id"]
        user_query = row["user_query"]
        raw_chunks = row["rag_chunks"] or "[]"

        try:
            chunks = json.loads(raw_chunks)
        except json.JSONDecodeError:
            chunks = []

        # Normalize chunk format
        normalized = []
        for c in chunks:
            normalized.append({
                "chunk_id":     c.get("chunk_id", f"{query_id}_chunk_{len(normalized)}"),
                "text":         c.get("text", ""),
                "source_file":  c.get("source_file", ""),
                "domain":       c.get("domain", ""),
                "rerank_score": float(c.get("rerank_score", 0.0)),
            })

        queries.append({
            "query_id":   query_id,
            "user_query": user_query,
        })
        query_chunks.append({
            "query_id":   query_id,
            "user_query": user_query,
            "chunks":     normalized,
        })

    queries_path = DATA_DIR / "queries_50.json"
    chunks_path  = DATA_DIR / "query_chunks_50.json"

    queries_path.write_text(json.dumps(queries, indent=2))
    chunks_path.write_text(json.dumps(query_chunks, indent=2))

    avg_chunks = sum(len(q["chunks"]) for q in query_chunks) / max(len(query_chunks), 1)
    print(f"Extracted {len(queries)} queries → {queries_path}")
    print(f"Extracted chunk pools   → {chunks_path}")
    print(f"Avg chunks per query: {avg_chunks:.1f}")


if __name__ == "__main__":
    main()
