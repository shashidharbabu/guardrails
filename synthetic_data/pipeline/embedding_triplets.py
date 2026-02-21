from __future__ import annotations

import json
import random
import time
from typing import Any, List, Optional

from tqdm import tqdm

from .anthropic_utils import call_claude_json, call_claude_text
from .io import load_checkpoint, save_checkpoint
from .prompts import HARD_NEGATIVE_PROMPT, TRIPLET_GENERATION_PROMPT


def _safe_json_array_of_strings(x: Any) -> Optional[List[str]]:
    if not isinstance(x, list):
        return None
    out: List[str] = []
    for item in x:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def generate_triplets_for_chunk(
    chunk: dict,
    *,
    all_chunks: List[dict],
    client: Any,
    model: str,
    retry_limit: int,
    n_queries: int = 4,
    debug: bool = False,
) -> List[dict]:
    triplets: List[dict] = []
    chunk_id = str(chunk.get("chunk_id"))

    def dbg(msg: str) -> None:
        if debug:
            print(f"[DEBUG][triplets][{chunk_id}] {msg}")

    query_prompt = TRIPLET_GENERATION_PROMPT.format(
        positive_text=str(chunk.get("text", "") or "")[:800],
        doc_id=chunk.get("doc_id", ""),
        threat_categories=chunk.get("threat_categories", []),
        n_queries=n_queries,
    )

    try:
        queries_raw = call_claude_json(
            client=client,
            model=model,
            prompt=query_prompt,
            max_tokens=600,
            retry_limit=retry_limit,
        )
        queries = _safe_json_array_of_strings(queries_raw)
        if not queries:
            dbg("query_generation_returned_empty_or_invalid_array")
            return []
        dbg(f"query_generation_ok count={len(queries)}")
    except Exception:
        dbg("query_generation_exception")
        return []

    pos_cats = set(chunk.get("threat_categories", []) or [])
    candidates = [
        c
        for c in all_chunks
        if c.get("chunk_id") != chunk.get("chunk_id")
        and c.get("doc_id") != chunk.get("doc_id")
        and (set(c.get("threat_categories", []) or []) & pos_cats)
    ]
    if len(candidates) < 5:
        dbg(f"insufficient_negative_candidates count={len(candidates)} min_required=5")
        return []
    dbg(f"negative_candidate_pool count={len(candidates)}")

    sample_candidates = random.sample(candidates, min(8, len(candidates)))
    candidate_text = "\n".join(
        f'[{c.get("chunk_id")}]: {c.get("summary") or str(c.get("text", ""))[:100]}'
        for c in sample_candidates
    )

    for query in queries:
        neg_prompt = HARD_NEGATIVE_PROMPT.format(
            query=query,
            positive_summary=chunk.get("summary") or str(chunk.get("text", ""))[:200],
            candidates=candidate_text,
        )
        try:
            neg_id = call_claude_text(
                client=client,
                model=model,
                prompt=neg_prompt,
                max_tokens=100,
                retry_limit=retry_limit,
            ).strip()
            neg_chunk = next(
                (c for c in sample_candidates if str(c.get("chunk_id")) == neg_id),
                None,
            )
            if not neg_chunk:
                dbg(f"hard_negative_not_found returned_id={neg_id!r}")
                continue

            triplets.append(
                {
                    "id": f'emb_{chunk["chunk_id"]}_{len(triplets):03d}',
                    "query": query,
                    "positive": {
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "text": chunk["text"],
                        "tier": chunk["tier"],
                    },
                    "hard_negative": {
                        "chunk_id": neg_chunk["chunk_id"],
                        "doc_id": neg_chunk["doc_id"],
                        "text": neg_chunk["text"],
                        "tier": neg_chunk["tier"],
                        "threat_categories": neg_chunk.get("threat_categories", []),
                    },
                    "metadata": {
                        "threat_categories": chunk.get("threat_categories", []),
                        "domain": chunk.get("tier"),
                        "positive_doc": chunk["doc_id"],
                        "negative_doc": neg_chunk["doc_id"],
                        "query_type": "mixed",
                    },
                }
            )
        except Exception:
            dbg("hard_negative_selection_exception")
            continue

    dbg(f"triplets_created count={len(triplets)}")
    return triplets


def generate_embedding_triplets(
    *,
    synthetic_dir: str,
    client: Any,
    model: str,
    retry_limit: int,
    target_total: int | None = None,
    save_every_chunks: int = 50,
    inter_save_sleep_seconds: float = 2.0,
    seed: int = 13,
) -> List[dict]:
    random.seed(seed)

    enriched = load_checkpoint(synthetic_dir, "enriched_chunks.jsonl")
    all_triplets = load_checkpoint(synthetic_dir, "embedding/triplets_progress.jsonl")

    done_ids = {t.get("positive", {}).get("chunk_id") for t in all_triplets}
    remaining = [
        c
        for c in enriched
        if c.get("chunk_id") not in done_ids and len(c.get("threat_categories", [])) > 0
    ]

    for i, chunk in enumerate(tqdm(remaining, desc="Generating triplets")):
        if target_total is not None and len(all_triplets) >= target_total:
            break

        triplets = generate_triplets_for_chunk(
            chunk,
            all_chunks=enriched,
            client=client,
            model=model,
            retry_limit=retry_limit,
        )
        all_triplets.extend(triplets)

        if (i + 1) % save_every_chunks == 0:
            save_checkpoint(all_triplets, synthetic_dir, "embedding/triplets_progress.jsonl")
            time.sleep(inter_save_sleep_seconds)

    save_checkpoint(all_triplets, synthetic_dir, "embedding/triplets_raw.jsonl")
    return all_triplets

