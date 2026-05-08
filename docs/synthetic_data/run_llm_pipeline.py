from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List

import anthropic

from synthetic_data.pipeline.agent_pairs import generate_agent_pair
from synthetic_data.pipeline.io import write_jsonl
from synthetic_data.pipeline.preprocess import preprocess_and_enrich
from synthetic_data.pipeline.ragas_eval import evaluate_llm_with_ragas
from synthetic_data.pipeline.validation import validate_agent_pairs


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
    ap = argparse.ArgumentParser(description="Run LLM pair synthetic-data pipeline.")
    ap.add_argument("--chunks-dir", required=True, help="Folder containing input JSONL chunk files.")
    ap.add_argument(
        "--out-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "_llm_run")),
        help="Output folder for pipeline artifacts.",
    )
    ap.add_argument("--model", default="claude-sonnet-4-20250514")
    ap.add_argument("--min-tokens", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=420)
    ap.add_argument("--retry-limit", type=int, default=3)
    ap.add_argument("--enrich-limit", type=int, default=200, help="Max chunks to enrich.")
    ap.add_argument("--pairs", type=int, default=100, help="Target number of LLM pairs.")
    ap.add_argument(
        "--label-dist",
        default='{"BLOCK": 0.5, "ALLOW": 0.3, "ESCALATE": 0.2}',
        help="JSON dict for decision distribution.",
    )
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
    label_dist = json.loads(args.label_dist)
    rng = random.Random(args.seed)

    out_dir = os.path.abspath(args.out_dir)
    synthetic_dir = os.path.join(out_dir, "synthetic")
    llm_dir = os.path.join(synthetic_dir, "llm")
    rejected_dir = os.path.join(llm_dir, "rejected")
    _ensure_dirs(out_dir, synthetic_dir, llm_dir, rejected_dir)

    dbg(
        "Args: "
        f"chunks_dir={os.path.abspath(args.chunks_dir)} "
        f"pairs={args.pairs} enrich_limit={args.enrich_limit} model={args.model}"
    )

    shared = preprocess_and_enrich(
        chunks_dir=os.path.abspath(args.chunks_dir),
        artifact_dir=llm_dir,
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

    dbg("Stage 3 start: LLM pair generation")
    by_cat: Dict[str, List[dict]] = {}
    for c in shared["enriched"]:
        for cat in (c.get("threat_categories") or []):
            by_cat.setdefault(cat, []).append(c)
    cats = [c for c in by_cat.keys() if by_cat[c]]
    dbg(f"Stage 3 threat categories available: {cats}")

    pairs_raw: List[dict] = []
    attempts = 0
    while len(pairs_raw) < int(args.pairs) and attempts < int(args.pairs) * 10:
        attempts += 1
        if not cats:
            dbg("Stage 3 no categories available; breaking")
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
                f"Stage 3 attempt={attempts} accepted pair_id={pair.get('id')} "
                f"decision={decision} threat_cat={threat_cat} total={len(pairs_raw)}"
            )
        else:
            dbg(f"Stage 3 attempt={attempts} failed decision={decision} threat_cat={threat_cat}")

    write_jsonl(os.path.join(llm_dir, "agent_pairs_raw.jsonl"), pairs_raw)
    print(f"Agent pairs generated (raw): {len(pairs_raw)}")

    pair_ok, pair_rej, stats = validate_agent_pairs(
        pairs_raw,
        target_distribution=label_dist,
    )
    write_jsonl(os.path.join(llm_dir, "agent_pairs_validated.jsonl"), pair_ok)
    write_jsonl(os.path.join(rejected_dir, "agent_pairs_rejected.jsonl"), pair_rej)

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Pairs: {len(pair_ok)} kept | {len(pair_rej)} rejected")
    if debug:
        pair_rej_counter = Counter()
        for r in pair_rej:
            for rr in r.get("reject_reasons", []) or []:
                pair_rej_counter[str(rr)] += 1
        dbg(f"Stage 4 pair reject reasons: {dict(pair_rej_counter)}")
    if stats:
        print(f"Pair stats: {stats}")

    if pair_ok:
        print("\n--- LLM pair sample ---")
        print(json.dumps(pair_ok[0], indent=2)[:2000])

    if args.with_ragas:
        print("\n=== RAGAS EVAL ===")
        eval_input = pair_ok if pair_ok else pairs_raw
        ragas_summary = evaluate_llm_with_ragas(
            pairs_for_eval=eval_input,
            eval_dir=os.path.join(llm_dir, "eval"),
            sample_limit=args.ragas_sample_limit,
            debug=debug,
        )
        print(f"RAGAS summary: {json.dumps(ragas_summary)}")


if __name__ == "__main__":
    main()

