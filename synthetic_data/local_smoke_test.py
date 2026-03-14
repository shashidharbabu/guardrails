from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import anthropic

from synthetic_data.pipeline.agent_pairs import generate_agent_pair
from synthetic_data.pipeline.embedding_triplets import generate_triplets_for_chunk
from synthetic_data.pipeline.enrichment import enrich_chunk
from synthetic_data.pipeline.filtering import filter_chunks, load_all_chunks
from synthetic_data.pipeline.io import write_jsonl
from synthetic_data.pipeline.validation import validate_agent_pairs, validate_embedding_triplets


def _pick_decision(rng: random.Random, label_dist: Dict[str, float]) -> str:
    items = list(label_dist.items())
    total = sum(float(w) for _, w in items) or 1.0
    r = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += float(w)
        if r <= acc:
            return k
    return items[-1][0]


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local smoke test for synthetic data pipeline (10 samples).")
    ap.add_argument("--chunks-dir", required=True, help="Folder containing input JSONL chunk files.")
    ap.add_argument(
        "--out-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "_local_run")),
        help="Output folder for checkpoints and samples.",
    )
    ap.add_argument("--model", default="claude-sonnet-4-20250514")
    ap.add_argument("--min-tokens", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=420)
    ap.add_argument("--retry-limit", type=int, default=3)
    ap.add_argument("--enrich-limit", type=int, default=200, help="Max chunks to enrich (for negatives/context).")
    ap.add_argument("--triplets", type=int, default=10, help="Target number of embedding triplets to generate.")
    ap.add_argument("--pairs", type=int, default=10, help="Target number of agent pairs to generate.")
    ap.add_argument(
        "--label-dist",
        default='{"BLOCK": 0.5, "ALLOW": 0.3, "ESCALATE": 0.2}',
        help="JSON dict for decision distribution.",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--debug", action="store_true", help="Enable detailed step-by-step debug logs.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    debug = bool(args.debug)

    def dbg(msg: str) -> None:
        if debug:
            print(f"[DEBUG] {msg}")

    # API key: use env var; avoid pasting secrets into code or notebooks.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY env var.")

    client = anthropic.Anthropic(api_key=api_key)
    label_dist = json.loads(args.label_dist)
    dbg(
        "Args: "
        f"chunks_dir={os.path.abspath(args.chunks_dir)} "
        f"triplets={args.triplets} pairs={args.pairs} "
        f"enrich_limit={args.enrich_limit} model={args.model}"
    )

    out_dir = os.path.abspath(args.out_dir)
    synthetic_dir = os.path.join(out_dir, "synthetic")
    rejected_dir = os.path.join(out_dir, "synthetic", "rejected")
    _ensure_dirs(out_dir, synthetic_dir, rejected_dir)

    # Stage 1: load + filter
    dbg("Stage 1 start: loading chunks")
    all_chunks = load_all_chunks(os.path.abspath(args.chunks_dir))
    dbg(f"Stage 1 loaded {len(all_chunks)} raw chunks")
    kept, rejected = filter_chunks(all_chunks, min_tokens=args.min_tokens, max_tokens=args.max_tokens)
    write_jsonl(os.path.join(synthetic_dir, "filtered_chunks.jsonl"), kept)
    write_jsonl(os.path.join(rejected_dir, "rejected_chunks.jsonl"), rejected)
    print(f"Loaded {len(all_chunks)} | Kept {len(kept)} | Rejected {len(rejected)}")
    if debug and rejected:
        rej_counter = Counter()
        for r in rejected:
            for rr in r.get("reject_reasons", []) or []:
                rej_counter[str(rr)] += 1
        dbg(f"Stage 1 rejection reasons: {dict(rej_counter)}")

    # Stage 2: enrich a subset (enough diversity for hard negatives + context)
    # Prefer diverse docs: shuffle and take first N.
    dbg("Stage 2 start: enrichment")
    rng.shuffle(kept)
    to_enrich = kept[: max(0, min(len(kept), int(args.enrich_limit)))]
    dbg(f"Stage 2 selected {len(to_enrich)} chunks for enrichment")
    enriched: List[dict] = []
    for idx, c in enumerate(to_enrich, start=1):
        out = enrich_chunk(
            c,
            client=client,
            model=args.model,
            retry_limit=args.retry_limit,
        )
        enriched.append(out)
        dbg(
            f"Stage 2 [{idx}/{len(to_enrich)}] "
            f"chunk_id={c.get('chunk_id')} "
            f"status={out.get('metadata_status')} "
            f"cats={len(out.get('threat_categories') or [])}"
        )
    write_jsonl(os.path.join(synthetic_dir, "enriched_chunks.jsonl"), enriched)
    failed = sum(1 for c in enriched if c.get("metadata_status") == "failed")
    print(f"Enriched {len(enriched) - failed} | Failed {failed}")

    # Stage 3: embedding triplets (attempt until we hit target or run out)
    dbg("Stage 3 start: triplet generation")
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
    write_jsonl(os.path.join(synthetic_dir, "embedding_triplets_raw.jsonl"), triplets_raw)
    print(f"Triplets generated (raw): {len(triplets_raw)}")

    # Stage 4: agent pairs (sample decisions and pick an available threat category)
    dbg("Stage 4 start: agent pair generation")
    by_cat: Dict[str, List[dict]] = {}
    for c in enriched:
        for cat in (c.get("threat_categories") or []):
            by_cat.setdefault(cat, []).append(c)

    cats = [c for c in by_cat.keys() if by_cat[c]]
    dbg(f"Stage 4 threat categories available: {cats}")
    pairs_raw: List[dict] = []
    attempts = 0
    while len(pairs_raw) < int(args.pairs) and attempts < int(args.pairs) * 10:
        attempts += 1
        if not cats:
            dbg("Stage 4 no categories available; breaking")
            break
        decision = _pick_decision(rng, label_dist)
        threat_cat = rng.choice(cats)
        pair = generate_agent_pair(
            by_cat[threat_cat],
            decision=decision,
            threat_category=threat_cat,
            client=client,
            model=args.model,
            max_tokens_response=2000,
            retry_limit=args.retry_limit,
            debug=debug,
        )
        if pair:
            pairs_raw.append(pair)
            dbg(
                f"Stage 4 attempt={attempts} accepted pair_id={pair.get('id')} "
                f"decision={decision} threat_cat={threat_cat} total={len(pairs_raw)}"
            )
        else:
            dbg(f"Stage 4 attempt={attempts} failed decision={decision} threat_cat={threat_cat}")
    write_jsonl(os.path.join(synthetic_dir, "agent_pairs_raw.jsonl"), pairs_raw)
    print(f"Agent pairs generated (raw): {len(pairs_raw)}")

    # Stage 5: validate (same checks as pipeline)
    trip_ok, trip_rej = validate_embedding_triplets(triplets_raw)
    write_jsonl(os.path.join(synthetic_dir, "embedding_triplets_validated.jsonl"), trip_ok)
    write_jsonl(os.path.join(rejected_dir, "embedding_triplets_rejected.jsonl"), trip_rej)

    pair_ok, pair_rej, stats = validate_agent_pairs(
        pairs_raw,
        target_distribution=label_dist,
    )
    write_jsonl(os.path.join(synthetic_dir, "agent_pairs_validated.jsonl"), pair_ok)
    write_jsonl(os.path.join(rejected_dir, "agent_pairs_rejected.jsonl"), pair_rej)

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Triplets: {len(trip_ok)} kept | {len(trip_rej)} rejected")
    print(f"Pairs:    {len(pair_ok)} kept | {len(pair_rej)} rejected")
    if debug:
        trip_rej_counter = Counter()
        for r in trip_rej:
            for rr in r.get("reject_reasons", []) or []:
                trip_rej_counter[str(rr)] += 1
        pair_rej_counter = Counter()
        for r in pair_rej:
            for rr in r.get("reject_reasons", []) or []:
                pair_rej_counter[str(rr)] += 1
        dbg(f"Stage 5 triplet reject reasons: {dict(trip_rej_counter)}")
        dbg(f"Stage 5 pair reject reasons: {dict(pair_rej_counter)}")
    if stats:
        print(f"Pair stats: {stats}")

    # Print 1 sample of each, if available
    if trip_ok:
        print("\n--- Triplet sample ---")
        print(json.dumps(trip_ok[0], indent=2)[:2000])
    if pair_ok:
        print("\n--- Agent pair sample ---")
        print(json.dumps(pair_ok[0], indent=2)[:2000])


if __name__ == "__main__":
    main()

