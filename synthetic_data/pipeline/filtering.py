from __future__ import annotations

import glob
import os
from typing import List, Tuple

import jsonlines
from tqdm import tqdm


def load_all_chunks(chunks_dir: str) -> List[dict]:
    all_chunks: List[dict] = []
    files = sorted(glob.glob(f"{chunks_dir}/*.jsonl"))
    for f in files:
        with jsonlines.open(f) as reader:
            for chunk in reader:
                chunk = dict(chunk)
                chunk["source_file"] = os.path.basename(f)
                all_chunks.append(chunk)
    return all_chunks


def is_toc_chunk(text: str) -> bool:
    # Heuristic from spec: table-of-contents chunks are dot/number heavy and short-ish.
    if not text:
        return True
    dot_ratio = text.count(".") / max(len(text), 1)
    return dot_ratio > 0.15 and len(text) < 1000


def filter_chunks(
    chunks: List[dict],
    *,
    min_tokens: int,
    max_tokens: int | None = None,
) -> Tuple[List[dict], List[dict]]:
    kept: List[dict] = []
    rejected: List[dict] = []
    seen_texts: set[str] = set()

    for chunk in tqdm(chunks, desc="Filtering chunks"):
        reasons: List[str] = []

        token_count = int(chunk.get("token_count", 0) or 0)
        if token_count < min_tokens:
            reasons.append("too_short")
        if max_tokens is not None and token_count > max_tokens:
            reasons.append("too_long")

        text = str(chunk.get("text", "") or "")
        if is_toc_chunk(text):
            reasons.append("toc_page")

        # Near-duplicate: signature by prefix (spec)
        text_sig = text[:200].strip()
        if text_sig in seen_texts:
            reasons.append("duplicate")
        else:
            seen_texts.add(text_sig)

        if reasons:
            out = dict(chunk)
            out["reject_reasons"] = reasons
            rejected.append(out)
            continue

        out = dict(chunk)
        out["flagged_truncation"] = str(out.get("chunk_method", "")).endswith("_trunc")
        kept.append(out)

    return kept, rejected

