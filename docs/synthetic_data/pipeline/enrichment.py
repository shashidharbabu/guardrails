from __future__ import annotations

import time
from typing import Any, Dict, List

from tqdm import tqdm

from .anthropic_utils import call_claude_json
from .io import load_checkpoint, save_checkpoint
from .prompts import METADATA_PROMPT


def enrich_chunk(
    chunk: dict,
    *,
    client: Any,
    model: str,
    retry_limit: int,
) -> dict:
    prompt = METADATA_PROMPT.format(
        chunk_text=str(chunk.get("text", "") or "")[:1500],
        doc_id=chunk.get("doc_id", ""),
        tier=chunk.get("tier", ""),
    )
    out = dict(chunk)
    try:
        metadata = call_claude_json(
            client=client,
            model=model,
            prompt=prompt,
            max_tokens=500,
            retry_limit=retry_limit,
        )
        if not isinstance(metadata, dict):
            raise ValueError("Metadata response was not an object")
        out.update(metadata)
        out["metadata_status"] = "enriched"
        return out
    except Exception:
        out["metadata_status"] = "failed"
        out["threat_categories"] = []
        out["cross_references"] = []
        out["applicable_scenarios"] = []
        out["severity"] = "unknown"
        out["summary"] = ""
        return out


def enrich_all_chunks(
    chunks: List[dict],
    *,
    synthetic_dir: str,
    client: Any,
    model: str,
    retry_limit: int,
    save_every: int = 100,
    rate_limit_sleep_seconds: float = 1.0,
) -> List[dict]:
    enriched = load_checkpoint(synthetic_dir, "enriched_chunks.jsonl")
    done_ids = {c.get("chunk_id") for c in enriched if c.get("chunk_id")}
    remaining = [c for c in chunks if c.get("chunk_id") not in done_ids]

    for i, chunk in enumerate(tqdm(remaining, desc="Enriching chunks")):
        enriched_chunk = enrich_chunk(
            chunk,
            client=client,
            model=model,
            retry_limit=retry_limit,
        )
        enriched.append(enriched_chunk)

        if (i + 1) % save_every == 0:
            save_checkpoint(enriched, synthetic_dir, "enriched_chunks.jsonl")
            time.sleep(rate_limit_sleep_seconds)

    save_checkpoint(enriched, synthetic_dir, "enriched_chunks.jsonl")
    return enriched

