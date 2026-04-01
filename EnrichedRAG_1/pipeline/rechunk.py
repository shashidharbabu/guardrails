#!/usr/bin/env python3
"""
Sentence-aware, structure-aware rechunker for regulatory RAG documents.
Reads old JSONL chunks, reconstructs per-document text, then rechunks
with spaCy sentence boundaries and structural heading awareness.
"""

import json
import re
import os
import hashlib
import statistics
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import spacy
import tiktoken

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
CHUNKS_DIR     = BASE_DIR / "chunks_with_links"
OUTPUT_DIR     = BASE_DIR / "rechunked_output"
TARGET_MIN     = 150
TARGET_LO      = 300
TARGET_HI      = 480
TARGET_MAX     = 512
DROP_UNDER     = 50
OVERLAP_TOKENS = 50

ENC = tiktoken.get_encoding("cl100k_base")

def tok_count(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))

def tok_encode(text: str) -> list[int]:
    return ENC.encode(text, disallowed_special=())

def tok_decode(tokens: list[int]) -> str:
    return ENC.decode(tokens)

# ── Tier mapping ────────────────────────────────────────────────────────────
# Use the tier already present in the source data; this is the fallback
# classifier based on filename patterns the user described.
T0_PATTERNS = ["nist", "owasp", "iso__27001", "iso_27001"]
T1_PATTERNS = ["gdpr", "hipaa", "eu_ai_act", "ccpa", "nis2", "cpra",
               "pipl", "cra_2024", "dsa_2022", "eo_14110", "appi",
               "saudi_pdpl", "online_safety", "coppa", "convention_108",
               "sg_pdpa"]
T3_PATTERNS = ["t3__"]

def classify_tier(filename: str) -> str:
    fn = filename.lower()
    if fn.startswith("t0") or any(p in fn for p in T0_PATTERNS):
        return "T0"
    if fn.startswith("t1") or any(p in fn for p in T1_PATTERNS):
        return "T1"
    if fn.startswith("t3") or any(p in fn for p in T3_PATTERNS):
        return "T3"
    return "T2"

# ── Structure heading patterns ──────────────────────────────────────────────
# MAJOR headings: force a new chunk unconditionally
MAJOR_HEADING_RE = re.compile(
    r"^("
    r"(?:Article|Section|Chapter|Part|Annex|Schedule|CHAPTER|PART|ARTICLE|SECTION|Appendix|ANNEX)\s+\d+"
    r"|(?:TITLE|Title)\s+[IVXLCDM\d]+"
    r"|Recital\s+\d+"
    r"|Rule\s+\d+"
    r"|Regulation\s+\d+"
    r"|Clause\s+\d+"
    r")",
    re.MULTILINE,
)

# MINOR headings: prefer to break here when chunk is already >= TARGET_MIN,
# but do NOT force a break (avoids tiny chunks from list items)
MINOR_HEADING_RE = re.compile(
    r"^("
    r"\d{1,3}\.\d{0,3}\.?\d{0,3}\.?\s+[A-Z]"           # 4.1 Understanding…
    r"|[A-Z]{2,}[\s.]+\d"                                  # GV.OC-01, PR.PS-01
    r"|\(\d+\)\s"                                           # (1) …
    r"|\([a-z]\)\s"                                         # (a) …
    r")",
    re.MULTILINE,
)

# Combined: used for metadata tagging
HEADING_RE = re.compile(
    r"^("
    r"(?:Article|Section|Chapter|Part|Annex|Schedule|CHAPTER|PART|ARTICLE|SECTION|Appendix|ANNEX)\s+\d+"
    r"|(?:TITLE|Title)\s+[IVXLCDM\d]+"
    r"|\d{1,3}\.\d{0,3}\.?\d{0,3}\.?\s+[A-Z]"
    r"|[A-Z]{2,}[\s.]+\d"
    r"|Recital\s+\d+"
    r"|Rule\s+\d+"
    r"|Regulation\s+\d+"
    r"|Clause\s+\d+"
    r")",
    re.MULTILINE,
)

ARTICLE_NUM_RE = re.compile(
    r"(?:Article|Section|Clause|Rule|Regulation|Recital)\s+(\d+[A-Za-z]?)",
    re.IGNORECASE,
)

# ── TOC / junk detection ───────────────────────────────────────────────────
def is_toc_or_junk(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    # Mostly dots + page numbers
    if re.search(r"\.{4,}", stripped):
        return True
    # Pure article cross-ref tables (the longest chunk problem from analysis)
    lines = stripped.split("\n")
    ref_lines = sum(1 for l in lines if re.match(r"^\s*Article\s+\d+", l.strip()))
    if len(lines) > 5 and ref_lines / len(lines) > 0.6:
        return True
    # All numbers/whitespace
    if re.match(r"^[\d\s.\-–—]+$", stripped):
        return True
    words = re.findall(r"[a-zA-Z]{2,}", stripped)
    if len(words) < 3 and tok_count(stripped) < DROP_UNDER:
        return True
    return False

# ── Load documents ──────────────────────────────────────────────────────────
def load_documents():
    """Load all JSONL files, reconstruct per-document full text + metadata."""
    docs = {}  # doc_id -> {text, tier, source_url, source, filename}
    old_chunks_all = []

    for fp in sorted(CHUNKS_DIR.glob("*.jsonl")):
        file_chunks = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                file_chunks.append(obj)
                old_chunks_all.append(obj)

        if not file_chunks:
            continue

        file_chunks.sort(key=lambda c: c.get("chunk_index", 0))
        doc_id = file_chunks[0]["doc_id"]
        full_text = "\n\n".join(c["text"] for c in file_chunks)

        tier = file_chunks[0].get("tier") or classify_tier(fp.name)
        source_url = file_chunks[0].get("source_url", "")
        source = file_chunks[0].get("source", {})

        docs[doc_id] = {
            "text": full_text,
            "tier": tier,
            "source_url": source_url,
            "source": source,
            "filename": fp.name,
        }

    return docs, old_chunks_all

# ── Sentence splitting with spaCy ──────────────────────────────────────────
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3_000_000  # some docs are large

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy, preserving structure."""
    # First split on structural headings to preserve them as boundaries
    segments = re.split(r"(\n\s*\n)", text)  # split on blank lines
    sentences = []
    for seg in segments:
        seg_stripped = seg.strip()
        if not seg_stripped:
            continue
        if len(seg_stripped) < 5:
            if sentences:
                sentences[-1] += seg
            continue
        # Use spaCy for sentence segmentation
        doc = nlp(seg_stripped)
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                sentences.append(s)
    return sentences

# ── Detect if a sentence is a structural heading ───────────────────────────
def is_heading(sent: str) -> bool:
    return bool(HEADING_RE.match(sent.strip()))

def extract_article_number(text: str) -> str | None:
    m = ARTICLE_NUM_RE.search(text)
    return m.group(1) if m else None

# ── Punctuation-based splitting for oversized sentences ─────────────────────
def split_long_sentence_at_punctuation(sent: str, max_tokens: int = 512) -> list[tuple[str, bool]]:
    """
    Split a sentence that exceeds max_tokens at the nearest comma or semicolon.
    Returns list of (text, was_split_at_punctuation) tuples.
    """
    tc = tok_count(sent)
    if tc <= max_tokens:
        return [(sent, False)]

    # Find all comma/semicolon positions
    split_points = []
    for i, ch in enumerate(sent):
        if ch in ",;":
            split_points.append(i)

    if not split_points:
        # No punctuation to split on — return as-is (rare edge case)
        return [(sent, False)]

    # Greedy: accumulate text up to max_tokens, split at last viable punctuation
    result = []
    start = 0
    while start < len(sent):
        remaining = sent[start:]
        if tok_count(remaining) <= max_tokens:
            result.append((remaining.strip(), len(result) > 0))
            break

        # Find the last split point that keeps us under max_tokens
        best_split = None
        for sp in split_points:
            if sp <= start:
                continue
            candidate = sent[start:sp + 1]
            if tok_count(candidate) <= max_tokens:
                best_split = sp
            else:
                break

        if best_split is not None:
            chunk_text = sent[start:best_split + 1].strip()
            if chunk_text:
                result.append((chunk_text, len(result) > 0))
            start = best_split + 1
        else:
            # No valid split point found under limit — take up to next punct anyway
            next_points = [sp for sp in split_points if sp > start]
            if next_points:
                chunk_text = sent[start:next_points[0] + 1].strip()
                if chunk_text:
                    result.append((chunk_text, len(result) > 0))
                start = next_points[0] + 1
            else:
                # Absolute fallback: take the rest
                result.append((remaining.strip(), len(result) > 0))
                break

    return result

# ── Core rechunking logic ──────────────────────────────────────────────────
def rechunk_document(doc_id: str, text: str) -> list[dict]:
    """
    Sentence-aware, structure-aware chunking.
    - Force new chunk at headings
    - Target 300-400 tokens, min 150, max 450
    - 50-token overlap between consecutive chunks
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_sents = []
    current_tokens = 0

    def finalize_chunk(sents, is_last=False):
        """Create a chunk from accumulated sentences."""
        chunk_text = " ".join(sents)
        tc = tok_count(chunk_text)

        # Drop junk
        if tc < DROP_UNDER or is_toc_or_junk(chunk_text):
            return None

        first_sent = sents[0].strip() if sents else ""
        return {
            "text": chunk_text,
            "token_count": tc,
            "starts_at_boundary": is_heading(first_sent),
            "article_number": extract_article_number(chunk_text),
        }

    # Pre-process: split any oversized sentences at punctuation
    processed_sents = []  # list of (text, split_at_punctuation)
    for sent in sentences:
        parts = split_long_sentence_at_punctuation(sent, max_tokens=TARGET_MAX)
        processed_sents.extend(parts)

    punct_split_count = 0

    for sent_text, was_punct_split in processed_sents:
        sent_tokens = tok_count(sent_text)
        if was_punct_split:
            punct_split_count += 1

        is_major = bool(MAJOR_HEADING_RE.match(sent_text.strip()))
        is_minor = bool(MINOR_HEADING_RE.match(sent_text.strip()))

        # MAJOR heading: always force a new chunk
        if is_major and current_sents:
            result = finalize_chunk(current_sents)
            if result:
                chunks.append(result)
            current_sents = []
            current_tokens = 0

        # MINOR heading: only break if we already have >= TARGET_MIN tokens
        elif is_minor and current_sents and current_tokens >= TARGET_MIN:
            result = finalize_chunk(current_sents)
            if result:
                chunks.append(result)
            current_sents = []
            current_tokens = 0

        # If adding this sentence would exceed max, finalize first
        if current_tokens + sent_tokens > TARGET_MAX and current_sents:
            result = finalize_chunk(current_sents)
            if result:
                chunks.append(result)
            current_sents = []
            current_tokens = 0

        current_sents.append(sent_text)
        current_tokens += sent_tokens

        # If we've reached the target range, check if we should finalize
        if current_tokens >= TARGET_LO:
            result = finalize_chunk(current_sents)
            if result:
                chunks.append(result)
            current_sents = []
            current_tokens = 0

    # Don't forget the last chunk
    if current_sents:
        result = finalize_chunk(current_sents)
        if result:
            # If the last chunk is too small, merge with previous
            if result["token_count"] < TARGET_MIN and chunks:
                prev = chunks[-1]
                merged_text = prev["text"] + " " + result["text"]
                merged_tc = tok_count(merged_text)
                if merged_tc <= TARGET_MAX + 50:  # allow slight overflow for final merge
                    prev["text"] = merged_text
                    prev["token_count"] = merged_tc
                    if result["article_number"] and not prev["article_number"]:
                        prev["article_number"] = result["article_number"]
                else:
                    chunks.append(result)
            else:
                chunks.append(result)

    # Tag chunks that contain punctuation-split fragments
    for c in chunks:
        c["split_at_punctuation"] = False
    if punct_split_count > 0:
        # Mark chunks whose text contains fragments from punct-split sentences
        # Simple heuristic: if the chunk core text ends with , or ; it was likely split
        for c in chunks:
            core = c.get("text", "").strip()
            if core and core[-1] in ",;":
                c["split_at_punctuation"] = True

    # ── Apply overlap ───────────────────────────────────────────────────
    # Store core_text (without overlap) for boundary quality checking
    if len(chunks) > 1:
        for i in range(len(chunks)):
            chunks[i]["core_text"] = chunks[i]["text"]
        for i in range(1, len(chunks)):
            prev_tokens = tok_encode(chunks[i - 1]["text"])
            if len(prev_tokens) > OVERLAP_TOKENS:
                overlap_text = tok_decode(prev_tokens[-OVERLAP_TOKENS:])
                chunks[i]["text"] = overlap_text.strip() + " " + chunks[i]["text"]
                chunks[i]["token_count"] = tok_count(chunks[i]["text"])
                chunks[i]["has_overlap"] = True
            else:
                chunks[i]["has_overlap"] = False
        chunks[0]["has_overlap"] = False
    else:
        for c in chunks:
            c["core_text"] = c["text"]
            c["has_overlap"] = False

    # ── Deduplicate ───────────────────────────────────────────────────
    seen = set()
    deduped = []
    for c in chunks:
        h = hashlib.md5(c["core_text"].strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(c)
    chunks = deduped

    return chunks

# ── Build final output ─────────────────────────────────────────────────────
def build_output(docs):
    all_chunks = []
    per_doc_stats = {}

    for doc_id, doc_info in sorted(docs.items()):
        print(f"  Rechunking: {doc_id} ...", end=" ", flush=True)
        raw_chunks = rechunk_document(doc_id, doc_info["text"])

        doc_chunks = []
        for idx, c in enumerate(raw_chunks):
            chunk = {
                "chunk_id": f"{doc_id}__rechunk_{idx:04d}",
                "doc_id": doc_id,
                "doc_name": doc_info["source"].get("document_name", doc_id),
                "tier": doc_info["tier"],
                "chunk_index": idx,
                "token_count": c["token_count"],
                "article_number": c["article_number"],
                "has_overlap": c.get("has_overlap", False),
                "starts_at_boundary": c["starts_at_boundary"],
                "split_at_punctuation": c.get("split_at_punctuation", False),
                "source_url": doc_info["source_url"],
                "text": c["text"],
                # Internal: used for report boundary checks, stripped before saving
                "_core_text": c.get("core_text", c["text"]),
            }
            doc_chunks.append(chunk)

        per_doc_stats[doc_id] = {
            "count": len(doc_chunks),
            "tier": doc_info["tier"],
            "tokens": [c["token_count"] for c in doc_chunks],
        }
        all_chunks.extend(doc_chunks)
        print(f"{len(doc_chunks)} chunks")

    return all_chunks, per_doc_stats

# ── Comparison report ──────────────────────────────────────────────────────
def generate_report(old_chunks, new_chunks, per_doc_stats):
    lines = []
    w = lines.append

    w("=" * 72)
    w("       RECHUNKING REPORT — OLD vs NEW")
    w(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 72)

    # ── Overall counts ──
    w(f"\n{'':>30s} {'OLD':>10s} {'NEW':>10s}")
    w(f"{'Total chunks':>30s} {len(old_chunks):>10d} {len(new_chunks):>10d}")
    w(f"{'Documents':>30s} {len(set(c['doc_id'] for c in old_chunks)):>10d} {len(per_doc_stats):>10d}")

    # ── Token statistics ──
    old_tc = [tok_count(c["text"]) for c in old_chunks]
    new_tc = [c["token_count"] for c in new_chunks]

    w(f"\n{'--- Token Statistics ---':^72s}")
    w(f"{'':>30s} {'OLD':>10s} {'NEW':>10s}")
    w(f"{'Min tokens':>30s} {min(old_tc):>10d} {min(new_tc):>10d}")
    w(f"{'Max tokens':>30s} {max(old_tc):>10d} {max(new_tc):>10d}")
    w(f"{'Mean tokens':>30s} {statistics.mean(old_tc):>10.1f} {statistics.mean(new_tc):>10.1f}")
    w(f"{'Median tokens':>30s} {statistics.median(old_tc):>10.1f} {statistics.median(new_tc):>10.1f}")
    w(f"{'Std dev':>30s} {statistics.stdev(old_tc):>10.1f} {statistics.stdev(new_tc):>10.1f}")

    # ── Distribution ──
    buckets = [
        ("0-50", 0, 50), ("51-100", 51, 100), ("101-150", 101, 150),
        ("151-200", 151, 200), ("201-300", 201, 300), ("301-400", 301, 400),
        ("401-450", 401, 450), ("451-512", 451, 512), ("512+", 513, 999999),
    ]
    w(f"\n{'--- Token Distribution ---':^72s}")
    w(f"  {'Bucket':>10s}  {'OLD cnt':>8s} {'OLD %':>7s}  {'NEW cnt':>8s} {'NEW %':>7s}")
    for label, lo, hi in buckets:
        o_cnt = sum(1 for t in old_tc if lo <= t <= hi)
        n_cnt = sum(1 for t in new_tc if lo <= t <= hi)
        o_pct = o_cnt / len(old_tc) * 100
        n_pct = n_cnt / len(new_tc) * 100
        w(f"  {label:>10s}  {o_cnt:>8d} {o_pct:>6.1f}%  {n_cnt:>8d} {n_pct:>6.1f}%")

    # ── Size compliance ──
    w(f"\n{'--- Size Compliance ---':^72s}")
    old_under100 = sum(1 for t in old_tc if t < 100)
    new_under100 = sum(1 for t in new_tc if t < 100)
    old_over500 = sum(1 for t in old_tc if t > 500)
    new_over500 = sum(1 for t in new_tc if t > 500)
    old_in_target = sum(1 for t in old_tc if TARGET_LO <= t <= TARGET_HI)
    new_in_target = sum(1 for t in new_tc if TARGET_LO <= t <= TARGET_HI)

    w(f"{'':>30s} {'OLD':>10s} {'NEW':>10s}")
    w(f"{'Under 100 tokens':>30s} {old_under100:>9d}  {new_under100:>9d}")
    w(f"{'Over 500 tokens':>30s} {old_over500:>9d}  {new_over500:>9d}")
    w(f"{'In target (300-400)':>30s} {old_in_target:>9d}  {new_in_target:>9d}")
    w(f"{'% in target':>30s} {old_in_target/len(old_tc)*100:>8.1f}%  {new_in_target/len(new_tc)*100:>8.1f}%")

    # ── Boundary quality ──
    w(f"\n{'--- Boundary Quality ---':^72s}")

    def boundary_stats(chunks_list, use_core=False):
        bad_start = 0
        bad_end = 0
        for c in chunks_list:
            # For new chunks, check core_text (without overlap prefix)
            text = (c.get("_core_text", c["text"]) if use_core else c["text"]).strip()
            if not text:
                continue
            if text[0].islower() and text[0] not in "•–—-":
                bad_start += 1
            if text[-1] not in ".?!])\"'":
                bad_end += 1
        return bad_start, bad_end

    o_bs, o_be = boundary_stats(old_chunks, use_core=False)
    n_bs, n_be = boundary_stats(new_chunks, use_core=True)

    w(f"{'':>30s} {'OLD':>10s} {'NEW':>10s}")
    w(f"{'Bad start (mid-sentence)':>30s} {o_bs:>9d}  {n_bs:>9d}")
    w(f"{'% bad start':>30s} {o_bs/len(old_chunks)*100:>8.1f}%  {n_bs/len(new_chunks)*100:>8.1f}%")
    w(f"{'Bad end (mid-sentence)':>30s} {o_be:>9d}  {n_be:>9d}")
    w(f"{'% bad end':>30s} {o_be/len(old_chunks)*100:>8.1f}%  {n_be/len(new_chunks)*100:>8.1f}%")

    # ── Duplication ──
    w(f"\n{'--- Duplication ---':^72s}")
    old_hashes = [hashlib.md5(c["text"].strip().encode()).hexdigest() for c in old_chunks]
    new_hashes = [hashlib.md5(c.get("_core_text", c["text"]).strip().encode()).hexdigest() for c in new_chunks]
    w(f"{'':>30s} {'OLD':>10s} {'NEW':>10s}")
    w(f"{'Exact duplicates':>30s} {len(old_hashes)-len(set(old_hashes)):>9d}  {len(new_hashes)-len(set(new_hashes)):>9d}")

    # ── Content quality ──
    w(f"\n{'--- Content Quality ---':^72s}")
    def content_stats(chunks_list):
        low_content = sum(1 for c in chunks_list if len(re.findall(r"[a-zA-Z]{2,}", c["text"])) < 20)
        toc = sum(1 for c in chunks_list if re.search(r"\.{4,}", c["text"]))
        return low_content, toc

    o_lc, o_toc = content_stats(old_chunks)
    n_lc, n_toc = content_stats(new_chunks)
    w(f"{'':>30s} {'OLD':>10s} {'NEW':>10s}")
    w(f"{'Low content (<20 words)':>30s} {o_lc:>9d}  {n_lc:>9d}")
    w(f"{'TOC/junk chunks':>30s} {o_toc:>9d}  {n_toc:>9d}")

    # ── Overlap stats (new only) ──
    w(f"\n{'--- Overlap (NEW only) ---':^72s}")
    overlap_count = sum(1 for c in new_chunks if c.get("has_overlap"))
    w(f"  Chunks with 50-token overlap: {overlap_count} / {len(new_chunks)} ({overlap_count/len(new_chunks)*100:.1f}%)")

    # ── Structural boundary starts (new only) ──
    boundary_starts = sum(1 for c in new_chunks if c.get("starts_at_boundary"))
    w(f"  Chunks starting at structural boundary: {boundary_starts} / {len(new_chunks)} ({boundary_starts/len(new_chunks)*100:.1f}%)")

    # ── Article coverage ──
    articles_found = sum(1 for c in new_chunks if c.get("article_number"))
    w(f"  Chunks with article_number metadata: {articles_found} / {len(new_chunks)} ({articles_found/len(new_chunks)*100:.1f}%)")

    # ── Per-doc summary ──
    w(f"\n{'--- Per-Document Summary ---':^72s}")
    w(f"  {'Document':50s} {'Tier':>4s} {'Chunks':>7s} {'Avg Tok':>8s}")
    w(f"  {'-'*50} {'-'*4} {'-'*7} {'-'*8}")
    for doc_id, stats in sorted(per_doc_stats.items()):
        avg = statistics.mean(stats["tokens"]) if stats["tokens"] else 0
        w(f"  {doc_id:50s} {stats['tier']:>4s} {stats['count']:>7d} {avg:>8.0f}")

    # ── Sample new chunks ──
    w(f"\n{'--- 5 Sample New Chunks ---':^72s}")
    random.seed(42)
    samples = random.sample(new_chunks, min(5, len(new_chunks)))
    for i, c in enumerate(samples, 1):
        w(f"\n  Sample {i}: [{c['chunk_id']}]")
        w(f"  Tier: {c['tier']} | Tokens: {c['token_count']} | "
          f"Article: {c['article_number'] or 'N/A'} | "
          f"Overlap: {c['has_overlap']} | Boundary: {c['starts_at_boundary']}")
        w(f"  {'─'*60}")
        text_preview = c["text"][:500]
        for line in text_preview.split("\n"):
            w(f"  │ {line}")
        if len(c["text"]) > 500:
            w(f"  │ ... [{len(c['text'])-500} more chars]")
        w(f"  {'─'*60}")

    # ── Verdict ──
    w(f"\n{'='*72}")
    w(f"  OVERALL IMPROVEMENT SUMMARY")
    w(f"{'='*72}")
    w(f"  Bad start boundaries:  {o_bs/len(old_chunks)*100:.1f}% → {n_bs/len(new_chunks)*100:.1f}%  "
      f"({'↓ IMPROVED' if n_bs/len(new_chunks) < o_bs/len(old_chunks) else '→ same/worse'})")
    w(f"  Bad end boundaries:    {o_be/len(old_chunks)*100:.1f}% → {n_be/len(new_chunks)*100:.1f}%  "
      f"({'↓ IMPROVED' if n_be/len(new_chunks) < o_be/len(old_chunks) else '→ same/worse'})")
    w(f"  In-target size:        {old_in_target/len(old_tc)*100:.1f}% → {new_in_target/len(new_tc)*100:.1f}%  "
      f"({'↑ IMPROVED' if new_in_target/len(new_tc) > old_in_target/len(old_tc) else '→ same/worse'})")
    w(f"  TOC/junk chunks:       {o_toc} → {n_toc}  "
      f"({'↓ IMPROVED' if n_toc < o_toc else '→ same/worse'})")
    w(f"  Under-100 chunks:      {old_under100} → {new_under100}  "
      f"({'↓ IMPROVED' if new_under100 < old_under100 else '→ same/worse'})")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  RECHUNKER — Sentence-aware, Structure-aware               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Load
    print("\n[1/4] Loading documents from old chunks...")
    docs, old_chunks = load_documents()
    print(f"  Loaded {len(docs)} documents, {len(old_chunks)} old chunks")

    # Rechunk
    print(f"\n[2/4] Rechunking {len(docs)} documents...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    new_chunks, per_doc_stats = build_output(docs)
    print(f"\n  Total new chunks: {len(new_chunks)}")

    # Save (strip internal fields)
    print("\n[3/4] Saving output...")
    chunks_path = OUTPUT_DIR / "chunks.json"
    save_chunks = [{k: v for k, v in c.items() if not k.startswith("_")} for c in new_chunks]
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(save_chunks, f, indent=2, ensure_ascii=False)
    print(f"  Saved {chunks_path} ({len(save_chunks)} chunks)")

    # Report
    print("\n[4/4] Generating comparison report...")
    report = generate_report(old_chunks, new_chunks, per_doc_stats)
    report_path = OUTPUT_DIR / "rechunk_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved {report_path}")

    # Print full report to stdout
    print("\n")
    print(report)

    # Print compact summary of the 4 key metrics
    print("\n" + "=" * 50)
    print("  KEY METRICS (quick check)")
    print("=" * 50)
    # Recompute from new_chunks directly
    new_tc = [c["token_count"] for c in new_chunks]
    def _boundary(chunks_list):
        bs = be = 0
        for c in chunks_list:
            t = c.get("_core_text", c["text"]).strip()
            if not t: continue
            if t[0].islower() and t[0] not in "•–—-": bs += 1
            if t[-1] not in ".?!])\"';:,—–-\u2019\u201d0123456789": be += 1
        return bs, be
    n_bs, n_be = _boundary(new_chunks)
    n_punct = sum(1 for c in new_chunks if c.get("split_at_punctuation"))
    # Diagnose: what characters do bad-end chunks actually end with?
    from collections import Counter
    GOOD_ENDS = set(".?!])\"';:,—–-\u2019\u201d0123456789")
    bad_end_chars = Counter()
    for c in new_chunks:
        t = c.get("_core_text", c["text"]).strip()
        if t and t[-1] not in GOOD_ENDS:
            bad_end_chars[repr(t[-1])] += 1
    print(f"  Bad end boundary:        {n_be}/{len(new_chunks)} = {n_be/len(new_chunks)*100:.1f}%")
    print(f"  Bad start boundary:      {n_bs}/{len(new_chunks)} = {n_bs/len(new_chunks)*100:.1f}%")
    print(f"  Max token count:         {max(new_tc)}")
    print(f"  Chunks split at punct:   {n_punct}")
    print(f"\n  Bad-end character breakdown:")
    for ch, cnt in bad_end_chars.most_common(15):
        print(f"    {ch:>6s}: {cnt:>5d} ({cnt/n_be*100:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()
