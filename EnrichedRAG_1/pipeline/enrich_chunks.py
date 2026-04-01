#!/usr/bin/env python3
"""
Contextual enrichment for rechunked RAG chunks.
Calls MiniMax M2.7 via OpenRouter to prepend document-aware context.
Two-pass strategy: Pass 1 = batch enrichment, Pass 2 = repair failures.
Goal: 100% LLM-generated context on every chunk.
Traced with LangSmith.
"""

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

import tiktoken
import requests
from langsmith import traceable

# ── LangSmith ───────────────────────────────────────────────────────────────
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "guardrails-rag")

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "rechunked_output" / "chunks.json"
OUTPUT_PATH = BASE_DIR / "rechunked_output" / "enriched_chunks.json"
PROGRESS_PATH = BASE_DIR / "rechunked_output" / ".enrich_progress.json"

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "minimax/minimax-m2.7"

BATCH_SIZE = 5       # smaller batches = less pressure on API
BATCH_DELAY = 2.0    # more breathing room between batches
SAVE_EVERY = 200
LOG_EVERY = 50

ENC = tiktoken.get_encoding("cl100k_base")

def tok_count(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))


# ── Cost estimation ─────────────────────────────────────────────────────────
def estimate_cost(chunks):
    total_input_tokens = 0
    for c in chunks:
        prompt_overhead = 80
        chunk_tokens = tok_count(c["text"][:2000])
        total_input_tokens += prompt_overhead + chunk_tokens

    total_output_tokens = len(chunks) * 75
    input_cost = (total_input_tokens / 1_000_000) * 0.40
    output_cost = (total_output_tokens / 1_000_000) * 1.60
    total_cost = input_cost + output_cost

    print(f"\n  Chunks to process:       {len(chunks)}")
    print(f"  Est. input tokens:       {total_input_tokens:,}")
    print(f"  Est. output tokens:      {total_output_tokens:,}")
    print(f"  Est. TOTAL cost:         ${total_cost:.4f}")
    print(f"  Est. time:               ~{len(chunks) * BATCH_DELAY / BATCH_SIZE / 60:.0f} min")

    return total_cost


# ── Single API call (traced) ───────────────────────────────────────────────
@traceable(name="contextual_enrichment", run_type="llm")
def call_minimax(chunk_id: str, doc_name: str, chunk_text: str, timeout: int = 20) -> str | None:
    """Single attempt to call MiniMax. Returns context string or None."""

    # Cap input — send first 2000 chars, enough for context
    text_input = chunk_text[:2000]

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://rag-pipeline.local",
        "X-Title": "RAG Chunk Enrichment",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You summarize regulatory documents. Write exactly 1-2 sentences of context. Never refuse. Always respond."
            },
            {
                "role": "user",
                "content": (
                    f"Given this document excerpt, write 1-2 sentences "
                    f"(50-100 tokens) situating this chunk within the document. "
                    f"State the regulation name, article/section if present, "
                    f"and what obligation or concept it covers. "
                    f"Output only the context sentences, nothing else.\n\n"
                    f"Document name: {doc_name}\n"
                    f"Chunk: {text_input}"
                )
            }
        ],
        "max_tokens": 500,     # MiniMax M2.7 uses ~100 reasoning + ~100 content tokens
        "temperature": 0.4,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 429:
            return None
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]

        # Only use the content field — reasoning chain-of-thought is noisy
        raw = msg.get("content")
        if raw and raw.strip():
            return raw.strip()

        # content was null — return None so Pass 2 retries with longer timeout
        return None
    except Exception:
        return None


# ── Enrich with retries ────────────────────────────────────────────────────
def enrich_with_retries(chunk: dict, max_attempts: int = 3, base_delay: float = 2.0) -> tuple[str | None, int]:
    """Try up to max_attempts with exponential backoff. Returns (context, attempts_used)."""
    cid = chunk["chunk_id"]
    doc_name = chunk.get("doc_name", chunk.get("doc_id", "Unknown"))
    chunk_text = chunk["text"]

    for attempt in range(max_attempts):
        if attempt > 0:
            delay = base_delay * (2 ** (attempt - 1))
            time.sleep(delay)

        timeout = 20 + (attempt * 10)  # 20s, 30s, 40s
        context = call_minimax(cid, doc_name, chunk_text, timeout=timeout)
        if context:
            return context, attempt + 1

    return None, max_attempts


# ── Batch processor (traced) ──────────────────────────────────────────────
@traceable(name="enrichment_batch", run_type="chain")
def process_batch(batch: list[dict], completed: dict) -> tuple[list[dict], list[dict]]:
    """Process a batch. Returns (successful, failed) lists."""
    successful = []
    failed = []

    for chunk in batch:
        context, attempts = enrich_with_retries(chunk, max_attempts=3)

        if context:
            chunk["context_prefix"] = context
            chunk["enriched_text"] = context + "\n" + chunk["text"]
            chunk["context_source"] = "minimax_m2.7"
            completed[chunk["chunk_id"]] = context
            successful.append(chunk)
        else:
            failed.append(chunk)

    return successful, failed


# ── Progress ────────────────────────────────────────────────────────────────
import re as _re

FALLBACK_MARKERS = ["covering regulatory content.", "This excerpt is from"]
# Reasoning junk pattern: "(10) covering (11) the (12) planning"
REASONING_JUNK_RE = _re.compile(r"\(\d+\)\s+\w+\s+\(\d+\)")

def is_bad_context(val: str) -> bool:
    if not val:
        return True
    if any(marker in val for marker in FALLBACK_MARKERS):
        return True
    if REASONING_JUNK_RE.search(val):
        return True
    return False

def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r") as f:
            data = json.load(f)
        completed = data.get("completed", {})
        # Purge bad entries so they get re-tried
        purged = 0
        for cid in list(completed.keys()):
            if is_bad_context(completed.get(cid, "")):
                del completed[cid]
                purged += 1
        if purged:
            print(f"  Purged {purged} bad/fallback entries — will re-try via API")
        return completed
    return {}


def save_progress(completed: dict, enriched_chunks: list):
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"completed": completed}, f)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)


# ── Smart metadata fallback (last resort) ──────────────────────────────────
import re as _re

DOC_NAME_MAP = {
    "iso__27001": "ISO/IEC 27001:2022 (Information Security Management)",
    "nist__csf": "NIST Cybersecurity Framework (CSF) 2.0",
    "nist__ssdf": "NIST SSDF SP 800-218 (Secure Software Development)",
    "owasp__ai_agent": "OWASP AI Agent Security Cheatsheet",
    "owasp__llm_prompt": "OWASP LLM Prompt Injection Cheatsheet",
    "gdpr": "EU General Data Protection Regulation (GDPR)",
    "hipaa": "US HIPAA (Health Insurance Portability and Accountability Act)",
    "eu_ai_act": "EU Artificial Intelligence Act 2024",
    "ccpa": "California Consumer Privacy Act (CCPA)",
    "cpra": "California Privacy Rights Act (CPRA)",
    "nis2": "EU NIS2 Directive 2022/2555",
    "dsa_2022": "EU Digital Services Act (DSA) 2022/2065",
    "cra_2024": "EU Cyber Resilience Act 2024",
    "pipl": "China Personal Information Protection Law (PIPL)",
    "eo_14110": "US Executive Order 14110 on AI Safety",
    "appi": "Japan Act on Protection of Personal Information (APPI)",
    "saudi_pdpl": "Saudi Arabia Personal Data Protection Law (PDPL)",
    "online_safety": "UK Online Safety Act 2023",
    "coppa": "US Children's Online Privacy Protection Act (COPPA)",
    "convention_108": "Council of Europe Convention 108",
    "sg_pdpa": "Singapore Personal Data Protection Act (PDPA)",
    "edpb": "EDPB Guidelines",
    "edps": "EDPS Orientations",
    "ico__ai_audit": "UK ICO AI Auditing Framework",
    "ico__explaining": "UK ICO Explaining AI Decisions",
    "cisa__ai": "US CISA AI Roadmap",
    "cpsc": "US CPSC Generative AI Policy",
    "fed_occ": "US OCC/FDIC Model Risk Management Guidance",
    "occ__model": "US OCC Model Risk Management Handbook",
    "omb__ai": "US OMB AI Agency Governance (M-24-10)",
    "iso__42001": "ISO/IEC 42001 AI Management System (Annex A)",
    "ai_rmf__1": "NIST AI Risk Management Framework (AI RMF) 1.0",
    "ai_rmf_genai": "NIST AI RMF Generative AI Profile",
    "nistir_8312": "NIST IR 8312 Explainable AI",
    "sp_1270": "NIST SP 1270 Bias in AI",
    "pdpc__ai": "Singapore PDPC AI Governance Framework",
}

TOPIC_PATTERNS = [
    (r"risk\s+(management|assessment)", "risk management requirements"),
    (r"data\s+protect", "data protection obligations"),
    (r"personal\s+(data|information)", "personal data processing rules"),
    (r"secur(ity|e)", "security requirements"),
    (r"privacy", "privacy obligations"),
    (r"transparen", "transparency requirements"),
    (r"consent", "consent requirements"),
    (r"notif(y|ication)", "notification obligations"),
    (r"audit", "audit and compliance requirements"),
    (r"penalty|fine|sanction", "enforcement and penalties"),
    (r"rights?\s+of", "individual rights provisions"),
    (r"controller|processor", "controller and processor obligations"),
    (r"transfer", "data transfer provisions"),
    (r"govern(ance|ing)", "governance requirements"),
    (r"incident", "incident response requirements"),
    (r"vulnerabilit", "vulnerability management"),
    (r"supply\s+chain", "supply chain security"),
    (r"high.risk", "high-risk AI system obligations"),
    (r"prohibit", "prohibited practices"),
    (r"conform", "conformity assessment requirements"),
    (r"supervis", "supervisory authority provisions"),
    (r"enforcement", "enforcement mechanisms"),
    (r"complaint", "complaint handling procedures"),
    (r"breach", "breach notification requirements"),
    (r"impact\s+assess", "impact assessment requirements"),
    (r"lawful", "lawfulness of processing"),
]

def smart_fallback(chunk: dict) -> str:
    doc_id = chunk.get("doc_id", "")
    article = chunk.get("article_number")
    text = chunk.get("text", "")[:500]

    reg_name = chunk.get("doc_name", doc_id)
    for pattern, name in DOC_NAME_MAP.items():
        if pattern in doc_id.lower():
            reg_name = name
            break

    section_ref = ""
    if article:
        section_ref = f", Article {article}"
    else:
        m = _re.search(r"(?:Section|Chapter|Part|Clause)\s+(\d+[\.\d]*)", text)
        if m:
            section_ref = f", Section {m.group(1)}"

    topic = ""
    text_lower = text.lower()
    for pattern, desc in TOPIC_PATTERNS:
        if _re.search(pattern, text_lower):
            topic = f" This section addresses {desc}."
            break

    return f"This excerpt is from {reg_name}{section_ref}.{topic}"


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CONTEXTUAL ENRICHMENT — MiniMax M2.7 via OpenRouter       ║")
    print("║  Two-pass: enrich all → repair failures                    ║")
    print("║  LangSmith tracing → project: guardrails-rag               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"\n  Loaded {len(chunks)} chunks")

    completed = load_progress()
    if completed:
        print(f"  Resuming: {len(completed)} chunks with good LLM context cached")

    remaining = [c for c in chunks if c["chunk_id"] not in completed]
    if not remaining:
        print("  All chunks already enriched! Rebuilding output...")
    else:
        estimate_cost(remaining)
        print(f"\n  Proceed? (y/n): ", end="", flush=True)
        if input().strip().lower() != "y":
            print("  Aborted.")
            return

    # ── PASS 1: Batch enrichment ──────────────────────────────────────────
    print(f"\n  ═══ PASS 1: Batch enrichment ═══")
    print(f"  Starting at {datetime.now().strftime('%H:%M:%S')}...")

    enriched_chunks = []
    failed_chunks = []
    processed = 0

    # Add cached chunks
    for chunk in chunks:
        if chunk["chunk_id"] in completed:
            chunk["context_prefix"] = completed[chunk["chunk_id"]]
            chunk["enriched_text"] = completed[chunk["chunk_id"]] + "\n" + chunk["text"]
            chunk["context_source"] = "minimax_m2.7"
            enriched_chunks.append(chunk)

    # Process remaining
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        successful, failed = process_batch(batch, completed)

        enriched_chunks.extend(successful)
        failed_chunks.extend(failed)
        processed += len(batch)

        if batch_start + BATCH_SIZE < len(remaining):
            time.sleep(BATCH_DELAY)

        if processed % LOG_EVERY < BATCH_SIZE and processed >= LOG_EVERY:
            total = len(completed)
            fails = len(failed_chunks)
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Done {total}/{len(chunks)} ({total/len(chunks)*100:.1f}%) "
                  f"| Failed: {fails}")

        if processed % SAVE_EVERY < BATCH_SIZE and processed >= SAVE_EVERY:
            save_progress(completed, enriched_chunks)
            print(f"    Checkpoint saved ({len(completed)} done)")

    save_progress(completed, enriched_chunks)
    print(f"\n  Pass 1 complete: {len(completed)}/{len(chunks)} enriched, {len(failed_chunks)} failed")

    # ── PASS 2: Repair failures with aggressive retries ───────────────────
    if failed_chunks:
        print(f"\n  ═══ PASS 2: Repairing {len(failed_chunks)} failures ═══")
        print(f"  Using longer timeouts and exponential backoff...")

        still_failed = []
        for i, chunk in enumerate(failed_chunks):
            # Aggressive: 5 attempts, longer delays, longer timeouts
            context, attempts = enrich_with_retries(chunk, max_attempts=5, base_delay=3.0)

            if context:
                chunk["context_prefix"] = context
                chunk["enriched_text"] = context + "\n" + chunk["text"]
                chunk["context_source"] = "minimax_m2.7"
                completed[chunk["chunk_id"]] = context
                enriched_chunks.append(chunk)
                print(f"    ✓ Repaired {chunk['chunk_id']} (attempt {attempts})")
            else:
                still_failed.append(chunk)
                print(f"    ✗ Still failed: {chunk['chunk_id']}")

            if i < len(failed_chunks) - 1:
                time.sleep(3.0)  # generous delay between repair attempts

        save_progress(completed, enriched_chunks)
        print(f"\n  Pass 2 complete: repaired {len(failed_chunks) - len(still_failed)}/{len(failed_chunks)}")

        # ── PASS 3: Smart fallback for any remaining (should be rare) ─────
        if still_failed:
            print(f"\n  ═══ PASS 3: Smart metadata fallback for {len(still_failed)} stubborn chunks ═══")
            for chunk in still_failed:
                context = smart_fallback(chunk)
                chunk["context_prefix"] = context
                chunk["enriched_text"] = context + "\n" + chunk["text"]
                chunk["context_source"] = "metadata_fallback"
                completed[chunk["chunk_id"]] = context
                enriched_chunks.append(chunk)
                print(f"    → Fallback: {chunk['chunk_id']}")

    # ── Final save ────────────────────────────────────────────────────────
    chunk_order = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    enriched_chunks.sort(key=lambda c: chunk_order.get(c["chunk_id"], 0))

    # Deduplicate (in case of resume adding duplicates)
    seen = set()
    deduped = []
    for c in enriched_chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            deduped.append(c)
    enriched_chunks = deduped

    save_progress(completed, enriched_chunks)

    if len(completed) == len(chunks):
        PROGRESS_PATH.unlink(missing_ok=True)

    # ── Summary ───────────────────────────────────────────────────────────
    api_count = sum(1 for c in enriched_chunks if c.get("context_source") == "minimax_m2.7")
    fallback_count = sum(1 for c in enriched_chunks if c.get("context_source") == "metadata_fallback")
    missing = len(chunks) - len(enriched_chunks)

    print(f"\n  {'='*55}")
    print(f"  FINAL RESULTS")
    print(f"  {'='*55}")
    print(f"  Total chunks:             {len(enriched_chunks)}/{len(chunks)}")
    print(f"  ├── MiniMax API context:  {api_count} ({api_count/len(enriched_chunks)*100:.1f}%)")
    print(f"  ├── Metadata fallback:    {fallback_count} ({fallback_count/len(enriched_chunks)*100:.1f}%)")
    print(f"  └── Missing:              {missing}")
    print(f"  Saved to: {OUTPUT_PATH}")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  LangSmith: https://smith.langchain.com → guardrails-rag")
    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
