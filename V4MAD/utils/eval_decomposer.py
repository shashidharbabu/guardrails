"""
Offline utility: evaluate decomposer quality by comparing baseline LLM answers
with reconstructed claims text using TF-IDF cosine similarity.

Run any time after step2 has completed:
  python utils/eval_decomposer.py

No GPU required — pure sklearn TF-IDF.

Output columns:
  query_id | num_claims | coverage_ratio | tfidf_cosine | reconstruction_quality
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH


def _reconstruct(conn: sqlite3.Connection, query_id: str) -> str:
    rows = conn.execute(
        "SELECT claim_text FROM claims WHERE query_id=? ORDER BY claim_index",
        (query_id,),
    ).fetchall()
    return " ".join(r[0] for r in rows if r[0])


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([text_a, text_b])
    return float(cosine_similarity(vec[0], vec[1])[0][0])


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT query_id, user_query, baseline_answer FROM queries ORDER BY query_id"
    ).fetchall()

    if not rows:
        print("No queries found in DB. Run step1 and step2 first.")
        return

    results = []
    for row in rows:
        qid      = row["query_id"]
        baseline = row["baseline_answer"] or ""
        recon    = _reconstruct(conn, qid)

        num_claims = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE query_id=?", (qid,)
        ).fetchone()[0]

        coverage_ratio = conn.execute(
            "SELECT AVG(coverage_ratio) FROM claims WHERE query_id=?", (qid,)
        ).fetchone()[0] or 0.0

        sim = _tfidf_cosine(baseline, recon)

        quality = (
            "GOOD"    if sim >= 0.70 else
            "OK"      if sim >= 0.50 else
            "WEAK"    if sim >= 0.30 else
            "POOR"
        )

        results.append({
            "query_id":          qid,
            "num_claims":        num_claims,
            "coverage_ratio":    round(coverage_ratio, 3),
            "tfidf_cosine":      round(sim, 3),
            "reconstruction_quality": quality,
        })

        print(f"[{qid}] claims={num_claims} coverage={coverage_ratio:.1%} "
              f"cosine={sim:.3f} → {quality}")

    if results:
        scores = [r["tfidf_cosine"] for r in results]
        print(f"\n── Summary ──────────────────────────────────────")
        print(f"  Queries evaluated  : {len(results)}")
        print(f"  Avg TF-IDF cosine  : {np.mean(scores):.3f}")
        print(f"  Median             : {np.median(scores):.3f}")
        print(f"  Min / Max          : {np.min(scores):.3f} / {np.max(scores):.3f}")
        good  = sum(1 for r in results if r["reconstruction_quality"] == "GOOD")
        ok    = sum(1 for r in results if r["reconstruction_quality"] == "OK")
        weak  = sum(1 for r in results if r["reconstruction_quality"] == "WEAK")
        poor  = sum(1 for r in results if r["reconstruction_quality"] == "POOR")
        print(f"  GOOD (≥0.70)       : {good}")
        print(f"  OK   (≥0.50)       : {ok}")
        print(f"  WEAK (≥0.30)       : {weak}")
        print(f"  POOR (<0.30)       : {poor}")

        if np.mean(scores) < 0.50:
            print("\n  ⚠ Average cosine < 0.50 — consider a larger decomposer model")
            print("    or tighten the DECOMPOSER_SYSTEM prompt to require full coverage.")

    out_path = Path(DB_PATH).parent / "eval_decomposer.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    main()
