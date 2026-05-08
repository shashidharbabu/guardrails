from __future__ import annotations

import os
import random
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple

from .enrichment import enrich_chunk
from .filtering import filter_chunks, load_all_chunks
from .io import write_jsonl


def preprocess_and_enrich(
    *,
    chunks_dir: str,
    artifact_dir: str,
    rejected_dir: str,
    client: Any,
    model: str,
    min_tokens: int,
    max_tokens: int | None,
    retry_limit: int,
    enrich_limit: int,
    seed: int,
    debug: bool = False,
    logger: Callable[[str], None] | None = None,
) -> Dict[str, List[dict]]:
    rng = random.Random(seed)

    def dbg(msg: str) -> None:
        if debug and logger:
            logger(msg)

    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    dbg("Stage 1 start: loading chunks")
    all_chunks = load_all_chunks(os.path.abspath(chunks_dir))
    dbg(f"Stage 1 loaded {len(all_chunks)} raw chunks")

    kept, rejected = filter_chunks(all_chunks, min_tokens=min_tokens, max_tokens=max_tokens)
    write_jsonl(os.path.join(artifact_dir, "filtered_chunks.jsonl"), kept)
    write_jsonl(os.path.join(rejected_dir, "rejected_chunks.jsonl"), rejected)
    dbg(f"Stage 1 done: kept={len(kept)} rejected={len(rejected)}")
    if debug and rejected:
        rej_counter = Counter()
        for r in rejected:
            for rr in r.get("reject_reasons", []) or []:
                rej_counter[str(rr)] += 1
        dbg(f"Stage 1 rejection reasons: {dict(rej_counter)}")

    dbg("Stage 2 start: enrichment")
    rng.shuffle(kept)
    to_enrich = kept[: max(0, min(len(kept), int(enrich_limit)))]
    dbg(f"Stage 2 selected {len(to_enrich)} chunks for enrichment")
    enriched: List[dict] = []
    for idx, c in enumerate(to_enrich, start=1):
        out = enrich_chunk(
            c,
            client=client,
            model=model,
            retry_limit=retry_limit,
        )
        enriched.append(out)
        dbg(
            f"Stage 2 [{idx}/{len(to_enrich)}] "
            f"chunk_id={c.get('chunk_id')} "
            f"status={out.get('metadata_status')} "
            f"cats={len(out.get('threat_categories') or [])}"
        )
    write_jsonl(os.path.join(artifact_dir, "enriched_chunks.jsonl"), enriched)
    failed = sum(1 for c in enriched if c.get("metadata_status") == "failed")
    dbg(f"Stage 2 done: enriched={len(enriched) - failed} failed={failed}")

    return {
        "all_chunks": all_chunks,
        "kept": kept,
        "rejected": rejected,
        "enriched": enriched,
    }

