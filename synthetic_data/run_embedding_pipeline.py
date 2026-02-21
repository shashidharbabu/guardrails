from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import List

import anthropic

from synthetic_data.pipeline.embedding_triplets import generate_triplets_for_chunk
from synthetic_data.pipeline.io import write_jsonl
from synthetic_data.pipeline.preprocess import preprocess_and_enrich
from synthetic_data.pipeline.ragas_eval import evaluate_embedding_with_ragas
from synthetic_data.pipeline.validation import validate_embedding_triplets


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run embedding triplet synthetic-data pipeline.")
    ap.add_argument("--chunks-dir", required=True, help="Folder containing input JSONL chunk files.")
    ap.add_argument(
        "--out-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "_embedding_run")),
        help="Output folder for pipeline artifacts.",
    )
    ap.add_argument("--model", default="claude-sonnet-4-20250514")
    ap.add_argument("--min-tokens", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=420)
    ap.add_argument("--retry-limit", type=int, default=3)
    ap.add_argument("--enrich-limit", type=int, default=200, help="Max chunks to enrich.")
    ap.add_argument("--triplets", type=int, default=100, help="Target number of embedding triplets.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--debug", action="store_true", help="Enable detailed step-by-step debug logs.")
    ap.add_argument("--with-ragas", action="store_true", help="Run RAGAS evaluation after validation.")
    ap.add_argument(
        "--ragas-sample-limit",
        type=int,
        default=None,
        help="Optional cap on number of samples passed to RAGAS.",
    )
    args = ap.parse_args()

    debug = bool(args.debug)

    def dbg(msg: str) -> None:
        if debug:
            print(f"[DEBUG] {msg}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY env var.")
    client = anthropic.Anthropic(api_key=api_key)

    out_dir = os.path.abspath(args.out_dir)
    synthetic_dir = os.path.join(out_dir, "synthetic")
    embedding_dir = os.path.join(synthetic_dir, "embedding")
    rejected_dir = os.path.join(embedding_dir, "rejected")
    _ensure_dirs(out_dir, synthetic_dir, embedding_dir, rejected_dir)

    dbg(
        "Args: "
        f"chunks_dir={os.path.abspath(args.chunks_dir)} "
        f"triplets={args.triplets} enrich_limit={args.enrich_limit} model={args.model}"
    )

    shared = preprocess_and_enrich(
        chunks_dir=os.path.abspath(args.chunks_dir),
        artifact_dir=embedding_dir,
        rejected_dir=rejected_dir,
        client=client,
        model=args.model,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        retry_limit=args.retry_limit,
        enrich_limit=args.enrich_limit,
        seed=args.seed,
        debug=debug,
        logger=dbg,
    )

    print(
        f"Loaded {len(shared['all_chunks'])} | Kept {len(shared['kept'])} | Rejected {len(shared['rejected'])}"
    )
    failed = sum(1 for c in shared["enriched"] if c.get("metadata_status") == "failed")
    print(f"Enriched {len(shared['enriched']) - failed} | Failed {failed}")

    dbg("Stage 3 start: triplet generation")
    enriched = shared["enriched"]
    triplets_raw: List[dict] = []
    for idx, c in enumerate(enriched, start=1):
        if len(triplets_raw) >= int(args.triplets):
            break
        if not c.get("threat_categories"):
            dbg(f"Stage 3 [{idx}/{len(enriched)}] skip chunk_id={c.get('chunk_id')} reason=no_threat_categories")
            continue
        new = generate_triplets_for_chunk(
            c,
            all_chunks=enriched,
            client=client,
            model=args.model,
            retry_limit=args.retry_limit,
            n_queries=4,
            debug=debug,
        )
        dbg(
            f"Stage 3 [{idx}/{len(enriched)}] chunk_id={c.get('chunk_id')} "
            f"generated={len(new)} cumulative={len(triplets_raw) + len(new)}"
        )
        for t in new:
            triplets_raw.append(t)
            if len(triplets_raw) >= int(args.triplets):
                break

    write_jsonl(os.path.join(embedding_dir, "embedding_triplets_raw.jsonl"), triplets_raw)
    print(f"Triplets generated (raw): {len(triplets_raw)}")

    trip_ok, trip_rej = validate_embedding_triplets(triplets_raw)
    write_jsonl(os.path.join(embedding_dir, "embedding_triplets_validated.jsonl"), trip_ok)
    write_jsonl(os.path.join(rejected_dir, "embedding_triplets_rejected.jsonl"), trip_rej)

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Triplets: {len(trip_ok)} kept | {len(trip_rej)} rejected")
    if debug:
        trip_rej_counter = Counter()
        for r in trip_rej:
            for rr in r.get("reject_reasons", []) or []:
                trip_rej_counter[str(rr)] += 1
        dbg(f"Stage 4 triplet reject reasons: {dict(trip_rej_counter)}")

    if trip_ok:
        print("\n--- Triplet sample ---")
        print(json.dumps(trip_ok[0], indent=2)[:2000])

    if args.with_ragas:
        print("\n=== RAGAS EVAL ===")
        eval_input = trip_ok if trip_ok else triplets_raw
        ragas_summary = evaluate_embedding_with_ragas(
            triplets_for_eval=eval_input,
            eval_dir=os.path.join(embedding_dir, "eval"),
            sample_limit=args.ragas_sample_limit,
            debug=debug,
        )
        print(f"RAGAS summary: {json.dumps(ragas_summary)}")


if __name__ == "__main__":
    main()

