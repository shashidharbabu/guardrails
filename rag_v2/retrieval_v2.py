#!/usr/bin/env python3
"""
retrieval_v2.py — Hybrid retrieval for guardrails_rag_v2
──────────────────────────────────────────────────────────
Single combined retriever for the unified old-corpus + healthcare collection.
No domain router needed — one collection handles both domains naturally.

Pipeline (identical to old retrieval.py, adapted for combined collection):
  1. Dense retrieval     — Qwen3-Embedding-4B embeds raw query → Qdrant top-20
                           Filters is_summary=False (real chunks only)
  2. Sparse retrieval    — BM25 on raw query → top-20
                           Already excludes summary nodes (not in BM25 corpus)
  3. RRF fusion          — Merge dense + sparse → top-30 candidates (k=60)
  4. BGE reranker        — BAAI/bge-reranker-v2-m3 cross-encoder → rescores top-30
  5. Tier boost          — Old T0 chunks in positions 6-10 forced into top-5
                           HC chunks have tier="" → unaffected by boost
  6. Source diversity    — ≤3 chunks from same source_file in final top-5
  7. Weak evidence check — Flag insufficient_evidence conservatively
  8. Output              — Top-5 real chunks with citations and parent context

Key differences from old retrieval.py:
  - Collection: guardrails_rag_v2 (instead of guardrails_rag)
  - BM25: bm25_combined.pkl (9299 docs, both domains)
  - Chunks loaded from both prepared/ JSONL files (not enriched_chunks.json)
  - Citation uses source_file (unified schema) instead of doc_name
  - Output includes: domain, source_file, parent_text
  - Summary nodes always excluded from final results
  - Source diversity floor added (≤3 from same source_file)
  - Optional domain filter for targeted queries

No HyDE (disabled by default — ablation confirmed no-HyDE wins).
Designed for Lightning AI T4 / local GPU.
"""

import json
import pickle
import os
import time
import re
from pathlib import Path

import torch
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:
    pass

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(os.environ.get("RAG_V2_BASE_DIR", Path(__file__).resolve().parent))

QDRANT_PATH = Path(os.environ.get("RAG_V2_QDRANT_PATH", str(BASE_DIR / "indexes_qdrant_data")))
BM25_PATH = Path(os.environ.get("RAG_V2_BM25_PATH", str(BASE_DIR / "bm25_combined.pkl")))

# Chunk payloads for BM25 lookup. These are the assets in the NewRAG 2 package.
OLD_CHUNKS_PATH = Path(os.environ.get("RAG_V2_OLD_CHUNKS_PATH", str(BASE_DIR / "data" / "enriched_chunks.json")))
HC_CHUNKS_PATH = Path(os.environ.get("RAG_V2_HC_CHUNKS_PATH", str(BASE_DIR / "data" / "healthcare_enriched_chunks.jsonl")))

COLLECTION = "guardrails_rag_v2"

# ── Models ───────────────────────────────────────────────────────────────────

EMBED_MODEL_NAME   = "Qwen/Qwen3-Embedding-4B"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# ── Retrieval parameters (identical to old pipeline) ─────────────────────────

DENSE_TOP_K         = 20    # candidates from Qdrant
SPARSE_TOP_K        = 20    # candidates from BM25
RRF_K               = 60    # RRF constant (standard value)
RERANK_CANDIDATES   = 30    # feed to reranker after RRF
FINAL_TOP_K         = 5     # final output to MAD agents
MAX_FROM_ONE_SOURCE = 3     # source diversity floor per source_file

MIN_RERANK_SCORE    = float(os.environ.get("MIN_RERANK_SCORE",   "0.15"))
MIN_TOP_RRF_SCORE   = float(os.environ.get("MIN_TOP_RRF_SCORE",  "0.013"))
MIN_SOURCE_COVERAGE = int(os.environ.get("MIN_SOURCE_COVERAGE",   "1"))

# Qdrant mode — local embedded DB, localhost server, or Qdrant Cloud.
QDRANT_MODE = os.environ.get("QDRANT_MODE", "local")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_TIMEOUT = int(os.environ.get("QDRANT_TIMEOUT", "60"))
RAG_V2_DENSE_DEVICE = os.environ.get("RAG_V2_DENSE_DEVICE", "auto").lower()

# HyDE — available but disabled (ablation proved no-HyDE wins)
HYDE_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HYDE_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HYDE_MODEL   = "minimax/minimax-m2.7"
DEFAULT_USE_HYDE = os.environ.get("USE_HYDE", "false").lower() == "true"

# LangSmith tracing (optional)
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.environ.get("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_PROJECT", "guardrails-rag-v2")

try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(func): return func
        return decorator


# ── Tokenizer (shared between BM25 and dense) ────────────────────────────────

def tokenize_for_bm25(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer. Identical to indexing tokenizer."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


# ── Chunk loader (for BM25 payload lookup) ───────────────────────────────────

def load_all_chunks() -> list[dict]:
    """
    Load all real chunks (old + HC) from the packaged NewRAG 2 payload files.
    Same order as BM25 corpus: old first, then HC.
    Returns list and chunk_id → chunk dict.
    """
    chunks = []

    if not OLD_CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing regulatory chunk file: {OLD_CHUNKS_PATH}")
    with open(OLD_CHUNKS_PATH, "r", encoding="utf-8") as f:
        old_chunks = json.load(f)
    for chunk in old_chunks:
        chunk.setdefault("domain", "general_regulatory")
        if not chunk.get("source_file"):
            chunk["source_file"] = chunk.get("doc_name", "")
        chunks.append(chunk)

    if not HC_CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing healthcare chunk file: {HC_CHUNKS_PATH}")
    with open(HC_CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                chunk.setdefault("domain", "healthcare")
                if not chunk.get("source_file"):
                    chunk["source_file"] = chunk.get("doc_name", "")
                chunks.append(chunk)
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# STEP 0: Optional HyDE — disabled by default
# ═══════════════════════════════════════════════════════════════════════════
# Ablation on old corpus proved hybrid_no_hyde (0.9220) > hybrid_hyde (0.8727).
# Available for re-testing on new corpus if needed.
# ═══════════════════════════════════════════════════════════════════════════

import requests

@traceable(name="hyde_expansion", run_type="llm")
def generate_hyde_answer(query: str, timeout: int = 25) -> str:
    """Generate a hypothetical answer passage for dense retrieval (HyDE)."""
    if not HYDE_API_KEY:
        return query

    headers = {
        "Authorization": f"Bearer {HYDE_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://rag-pipeline.local",
        "X-Title": "RAG HyDE Generation",
    }
    payload = {
        "model": HYDE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a regulatory and clinical compliance expert. "
                    "Given a question, write a short passage (100-150 words) "
                    "that would appear in a regulation or clinical guideline "
                    "and directly answers the question. Write as if quoting "
                    "from the actual document. Do not reference the question."
                )
            },
            {"role": "user", "content": query}
        ],
        "max_tokens": 500,
        "temperature": 0.4,
    }

    try:
        resp = requests.post(HYDE_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "")
        if content and content.strip():
            return content.strip()
    except Exception as e:
        print(f"  HyDE failed ({e}), falling back to raw query")

    return query


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Dense Retrieval — Qwen3-Embedding-4B → Qdrant
# ═══════════════════════════════════════════════════════════════════════════
# Real chunks only (is_summary=False filter).
# Summary nodes live in the index but are never returned as retrieval results.
# ═══════════════════════════════════════════════════════════════════════════

class DenseRetriever:
    """Qwen3-Embedding-4B query embedding + Qdrant guardrails_rag_v2 search."""

    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        from qdrant_client import QdrantClient

        print("  Loading embedding model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            EMBED_MODEL_NAME, trust_remote_code=True)
        requested_device = RAG_V2_DENSE_DEVICE
        if requested_device == "auto":
            requested_device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        dtype = torch.float32 if requested_device == "cpu" else torch.float16

        self.model = AutoModel.from_pretrained(
            EMBED_MODEL_NAME,
            trust_remote_code=True,
            dtype=dtype,
        )
        self.device = requested_device
        self.model = self.model.to(self.device).eval()

        print("  Connecting to Qdrant...")
        if QDRANT_MODE == "cloud":
            if not QDRANT_URL or not QDRANT_API_KEY:
                raise RuntimeError("QDRANT_MODE=cloud requires QDRANT_URL and QDRANT_API_KEY")
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=QDRANT_TIMEOUT,
            )
        elif QDRANT_MODE == "server":
            self.client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                timeout=QDRANT_TIMEOUT,
            )
        else:
            self.client = QdrantClient(path=str(QDRANT_PATH))

        info = self.client.get_collection(COLLECTION)
        print(f"  Dense retriever ready: {info.points_count} vectors on {self.device}")

    @torch.no_grad()
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns (2560,) float32 L2-normalized vector."""
        encoded = self.tokenizer(
            [text], padding=True, truncation=True,
            max_length=8192, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().float().numpy()[0]

    @traceable(name="dense_retrieval", run_type="retriever")
    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = DENSE_TOP_K,
        domain_filter: str | None = None,
    ) -> list[dict]:
        """
        Search guardrails_rag_v2 for real chunks only.

        Args:
            query_embedding: (2560,) float32 L2-normalized query vector
            top_k: number of candidates to return
            domain_filter: optional 'healthcare' or 'general_regulatory'
                           to restrict results to one domain
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        # Always filter out summary nodes
        must_conditions = [
            FieldCondition(key="is_summary", match=MatchValue(value=False))
        ]

        # Optional domain filter
        if domain_filter:
            must_conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=domain_filter))
            )

        results = self.client.query_points(
            collection_name=COLLECTION,
            query=query_embedding.tolist(),
            query_filter=Filter(must=must_conditions),
            limit=top_k,
        ).points

        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "score": r.score,
                "source": "dense",
                "payload": r.payload,
            }
            for r in results
        ]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Sparse Retrieval — BM25 on raw query
# ═══════════════════════════════════════════════════════════════════════════
# BM25 corpus already excludes summary nodes (built in build_combined_index.py).
# Uses raw text (not enriched_text) — exact lexical matching works better
# without the synthetic context prefix noise.
# ═══════════════════════════════════════════════════════════════════════════

class SparseRetriever:
    """BM25 retriever on combined real-chunk corpus."""

    def __init__(self, chunk_map: dict[str, dict]):
        print("  Loading BM25 index...")
        if not BM25_PATH.exists():
            raise FileNotFoundError(f"Missing BM25 index: {BM25_PATH}")

        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)

        self.bm25      = bm25_data["bm25"]
        self.chunk_ids = bm25_data["chunk_ids"]
        self.chunk_map = chunk_map   # chunk_id → chunk dict (from prepared/ JSONL)
        print(f"  BM25 ready: {len(self.chunk_ids)} documents")

    @traceable(name="sparse_retrieval", run_type="retriever")
    def retrieve(
        self,
        raw_query: str,
        top_k: int = SPARSE_TOP_K,
        domain_filter: str | None = None,
    ) -> list[dict]:
        """
        BM25 search on raw query. Optional domain filter applied post-scoring.
        """
        query_tokens = tokenize_for_bm25(raw_query)
        scores = self.bm25.get_scores(query_tokens)
        top_idxs = np.argsort(scores)[-top_k * 3:][::-1]   # over-fetch for domain filter

        results = []
        for idx in top_idxs:
            if scores[idx] <= 0:
                continue
            cid   = self.chunk_ids[idx]
            chunk = self.chunk_map.get(cid, {})

            # Optional domain filter
            if domain_filter and chunk.get("domain", "") != domain_filter:
                continue

            results.append({
                "chunk_id": cid,
                "score":    float(scores[idx]),
                "source":   "sparse",
                "payload":  chunk,
            })

            if len(results) == top_k:
                break

        return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Reciprocal Rank Fusion
# ═══════════════════════════════════════════════════════════════════════════
# Rank-based fusion of dense and sparse results.
# k=60 prevents top-ranked items from dominating.
# Dense (cosine 0-1) and BM25 (0-30+) are on different scales —
# RRF handles this naturally by using rank not raw score.
# ═══════════════════════════════════════════════════════════════════════════

@traceable(name="rrf_fusion", run_type="chain")
def reciprocal_rank_fusion(
    dense_results:  list[dict],
    sparse_results: list[dict],
    k:     int = RRF_K,
    top_n: int = RERANK_CANDIDATES,
) -> list[dict]:
    """Merge dense and sparse results via RRF. Returns top-n candidates."""
    rrf_scores = {}
    payloads   = {}
    sources    = {}

    for rank, r in enumerate(dense_results, 1):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)
        payloads[cid]   = r["payload"]
        sources.setdefault(cid, set()).add("dense")

    for rank, r in enumerate(sparse_results, 1):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)
        if cid not in payloads:
            payloads[cid] = r["payload"]
        sources.setdefault(cid, set()).add("sparse")

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_n]

    return [
        {
            "chunk_id":  cid,
            "rrf_score": rrf_scores[cid],
            "sources":   sources[cid],
            "payload":   payloads[cid],
        }
        for cid in sorted_ids
    ]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: BGE Reranker — Cross-encoder rescoring
# ═══════════════════════════════════════════════════════════════════════════
# Cross-encoder reads (query, candidate) pairs jointly — catches subtle
# relevance signals that bi-encoder cosine similarity misses.
# Only runs on 30 RRF candidates (not 9299) so it's fast.
# Uses enriched_text for reranking (context prefix helps cross-encoder).
# ═══════════════════════════════════════════════════════════════════════════

class BGEReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder."""

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        print("  Loading BGE reranker...")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        print(f"  Reranker ready on {self.device}")

    @torch.no_grad()
    @traceable(name="bge_reranking", run_type="chain")
    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Score each candidate against the raw query. Returns sorted list."""
        if not candidates:
            return []

        pairs = []
        for c in candidates:
            text = c["payload"].get("enriched_text") or c["payload"].get("text", "")
            pairs.append((query, text[:2048]))

        encoded = self.tokenizer(
            pairs, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**encoded)
        scores  = outputs.logits.squeeze(-1).cpu().float().numpy()

        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Tier Boost
# ═══════════════════════════════════════════════════════════════════════════
# Protects T0 authoritative sources (ISO 27001, NIST CSF, NIST SSDF etc.)
# that may fall just outside top-5 due to length/vocabulary.
# HC chunks have tier="" → not affected by this boost. Correct behavior.
# ═══════════════════════════════════════════════════════════════════════════

@traceable(name="tier_boost", run_type="chain")
def apply_tier_boost(reranked: list[dict], final_k: int = FINAL_TOP_K) -> list[dict]:
    """If T0 chunks are in positions 6-10, force them into top-5.

    Keep the full candidate list so later source-diversity selection can
    backfill to final_k instead of shrinking the result set.
    """
    top_5 = reranked[:final_k]
    rest  = reranked[final_k : final_k + 5]   # positions 6-10

    t0_in_rest = [c for c in rest if c["payload"].get("tier") == "T0"]
    if not t0_in_rest:
        return reranked

    non_t0_in_top5 = [c for c in top_5 if c["payload"].get("tier") != "T0"]
    non_t0_in_top5.sort(key=lambda x: x["rerank_score"])

    for t0_chunk in t0_in_rest:
        if non_t0_in_top5:
            replaced = non_t0_in_top5.pop(0)
            top_5.remove(replaced)
            top_5.append(t0_chunk)

    top_5.sort(key=lambda x: x["rerank_score"], reverse=True)
    promoted_ids = {c["chunk_id"] for c in top_5}
    return top_5[:final_k] + [c for c in reranked if c["chunk_id"] not in promoted_ids]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Source Diversity Floor
# ═══════════════════════════════════════════════════════════════════════════
# Prevent top-5 from being monopolized by a single verbose document.
# Example: ADA 2024 is 17 chunks — dense retrieval might surface all 5
# from ADA when the query spans multiple docs.
# Floor: ≤3 results from the same source_file in final top-5.
# ═══════════════════════════════════════════════════════════════════════════

def apply_source_diversity(
    candidates: list[dict],
    final_k: int = FINAL_TOP_K,
    max_per_source: int = MAX_FROM_ONE_SOURCE,
) -> list[dict]:
    """
    From reranked candidates (30), greedily select top-5 with diversity.
    Enforces ≤ max_per_source chunks from the same source_file.
    """
    source_counts: dict[str, int] = {}
    result = []

    for c in candidates:
        source_file = c["payload"].get("source_file") or c["payload"].get("doc_name") or "unknown"
        count = source_counts.get(source_file, 0)
        if count < max_per_source:
            result.append(c)
            source_counts[source_file] = count + 1
        if len(result) == final_k:
            break

    return result


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Weak Evidence Check
# ═══════════════════════════════════════════════════════════════════════════
# Conservatively flags weak retrieval. Intentionally mild to avoid
# suppressing good answers on borderline cases.
# ═══════════════════════════════════════════════════════════════════════════

@traceable(name="weak_evidence_check", run_type="chain")
def assess_retrieval_strength(results: list[dict]) -> dict:
    """Return evidence quality metadata for the final top-k results."""
    if not results:
        return {
            "is_sufficient": False,
            "reason": "no_results",
            "top_rerank_score": 0.0,
            "top_rrf_score": 0.0,
            "unique_sources": 0,
        }

    unique_sources = len({
        c["payload"].get("source_file") or c["payload"].get("doc_name") or "?"
        for c in results
    })
    top_rerank = float(results[0].get("rerank_score", 0.0))
    top_rrf    = float(results[0].get("rrf_score",    0.0))

    is_sufficient = (
        top_rerank >= MIN_RERANK_SCORE
        and top_rrf >= MIN_TOP_RRF_SCORE
        and unique_sources >= MIN_SOURCE_COVERAGE
    )

    return {
        "is_sufficient": is_sufficient,
        "reason": "ok" if is_sufficient else "weak_retrieval",
        "top_rerank_score": top_rerank,
        "top_rrf_score":    top_rrf,
        "unique_sources":   unique_sources,
    }


# ── Citation builder ──────────────────────────────────────────────────────────

def build_citation(payload: dict) -> str:
    """Build a compact source citation string for downstream LLM answering."""
    # Unified schema: source_file is canonical. doc_name is fallback for old schema.
    source_file  = payload.get("source_file") or payload.get("doc_name") or "unknown_source"
    article      = payload.get("article_number")
    source_url   = payload.get("source_url")
    source_sec   = payload.get("source_section")

    parts = [source_file]
    if article not in (None, "", 0, "0"):
        parts.append(f"Article {article}")
    if source_sec:
        parts.append(source_sec)
    if source_url:
        parts.append(source_url)

    return " | ".join(str(p) for p in parts)


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class RetrievalPipeline:
    """
    End-to-end hybrid retrieval: query → 5 enriched chunks for MAD agents.

    Single pipeline for guardrails_rag_v2 (old regulatory + healthcare).
    No domain router — pass domain_filter="healthcare" or "general_regulatory"
    if you want targeted results. Default is unrestricted (both domains).
    """

    def __init__(
        self,
        qdrant_path: str | Path | None = None,
        bm25_path: str | Path | None = None,
        old_chunks_path: str | Path | None = None,
        healthcare_chunks_path: str | Path | None = None,
        collection_name: str | None = None,
        qdrant_mode: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        qdrant_timeout: int | None = None,
    ):
        global QDRANT_PATH, BM25_PATH, OLD_CHUNKS_PATH, HC_CHUNKS_PATH, COLLECTION
        global QDRANT_MODE, QDRANT_URL, QDRANT_API_KEY, QDRANT_TIMEOUT
        if qdrant_path is not None:
            QDRANT_PATH = Path(qdrant_path)
        if bm25_path is not None:
            BM25_PATH = Path(bm25_path)
        if old_chunks_path is not None:
            OLD_CHUNKS_PATH = Path(old_chunks_path)
        if healthcare_chunks_path is not None:
            HC_CHUNKS_PATH = Path(healthcare_chunks_path)
        if collection_name is not None:
            COLLECTION = collection_name
        if qdrant_mode is not None:
            QDRANT_MODE = qdrant_mode
        if qdrant_url is not None:
            QDRANT_URL = qdrant_url
        if qdrant_api_key is not None:
            QDRANT_API_KEY = qdrant_api_key
        if qdrant_timeout is not None:
            QDRANT_TIMEOUT = qdrant_timeout

        print("=" * 60)
        print("  RETRIEVAL PIPELINE V2 — guardrails_rag_v2")
        print("  Combined: 5903 old regulatory + 3396 healthcare")
        print("=" * 60)

        # Load all real chunks for BM25 payload lookup
        print("\n  Loading chunk payloads...")
        all_chunks  = load_all_chunks()
        chunk_map   = {c["chunk_id"]: c for c in all_chunks}
        print(f"  Loaded {len(all_chunks)} real chunks (old + HC)")

        self.dense   = DenseRetriever()
        self.sparse  = SparseRetriever(chunk_map)
        self.reranker = BGEReranker()

        print("\n  Pipeline ready.\n")

    @traceable(name="full_retrieval_v2", run_type="chain")
    def retrieve(
        self,
        query:         str,
        use_hyde:      bool = DEFAULT_USE_HYDE,
        domain_filter: str | None = None,
        verbose:       bool = False,
    ) -> dict:
        """
        Full retrieval pipeline.

        Args:
            query:         Raw user query
            use_hyde:      Whether to use HyDE expansion (default: False)
            domain_filter: Optional 'healthcare' or 'general_regulatory'
                           Restricts Qdrant + BM25 to one domain.
                           Default None = search both domains.
            verbose:       Print intermediate step results

        Returns:
            {
              "query": str,
              "domain_filter": str | None,
              "use_hyde": bool,
              "retrieval_time_sec": float,
              "evidence": dict,
              "insufficient_evidence": bool,
              "chunks": list[dict]   ← top-5 real chunks
            }
        """
        t0 = time.time()

        # Step 0: Optional HyDE (disabled by default)
        if use_hyde:
            query_for_dense = generate_hyde_answer(query)
            if verbose:
                print(f"  [HyDE] Generated {len(query_for_dense.split())} word hypothetical")
        else:
            query_for_dense = query

        # Step 1: Dense retrieval (summary nodes excluded via Qdrant filter)
        query_embedding = self.dense.embed_query(query_for_dense)
        dense_results   = self.dense.retrieve(
            query_embedding, top_k=DENSE_TOP_K, domain_filter=domain_filter)
        if verbose:
            top_score = dense_results[0]["score"] if dense_results else 0.0
            print(f"  [Dense] {len(dense_results)} candidates, top score: {top_score:.4f}")

        # Step 2: Sparse retrieval (BM25, real chunks only)
        sparse_results = self.sparse.retrieve(
            query, top_k=SPARSE_TOP_K, domain_filter=domain_filter)
        if verbose:
            top_score = sparse_results[0]["score"] if sparse_results else 0.0
            print(f"  [Sparse] {len(sparse_results)} candidates, top score: {top_score:.4f}")

        # Step 3: RRF fusion → top-30 candidates
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        if verbose:
            both = sum(1 for c in fused if len(c["sources"]) == 2)
            print(f"  [RRF] {len(fused)} candidates ({both} from both sources)")

        # Step 4: BGE reranker
        reranked = self.reranker.rerank(query, fused)
        if verbose:
            top_score = reranked[0]["rerank_score"] if reranked else 0.0
            print(f"  [Rerank] Top rerank score: {top_score:.4f}")

        # Step 5: Tier boost (protects old T0 chunks; HC chunks unaffected)
        after_boost = apply_tier_boost(reranked)

        # Step 6: Source diversity (≤3 from same source_file in top-5)
        final = apply_source_diversity(after_boost)

        # Step 7: Weak evidence check
        strength = assess_retrieval_strength(final)

        elapsed = time.time() - t0
        if verbose:
            print(f"  [Done] {len(final)} chunks in {elapsed:.2f}s")
            print(
                f"  [Evidence] sufficient={strength['is_sufficient']} "
                f"top_rerank={strength['top_rerank_score']:.4f} "
                f"top_rrf={strength['top_rrf_score']:.4f} "
                f"unique_sources={strength['unique_sources']}"
            )

        # Format output
        output = []
        for rank, c in enumerate(final, 1):
            p = c["payload"]
            output.append({
                "rank":           rank,
                "chunk_id":       c["chunk_id"],
                "rerank_score":   round(c["rerank_score"], 4),
                "rrf_score":      round(c["rrf_score"],    4),
                "sources":        list(c["sources"]),
                # Identity
                "domain":         p.get("domain", ""),
                "source_file":    p.get("source_file") or p.get("doc_name", ""),
                "source_section": p.get("source_section", ""),
                "tier":           p.get("tier", ""),
                "doc_type":       p.get("doc_type", ""),
                "article_number": p.get("article_number", ""),
                "source_url":     p.get("source_url", ""),
                "citation":       build_citation(p),
                # Content — what the LLM gets
                "text":           p.get("text", ""),
                "enriched_text":  p.get("enriched_text", p.get("text", "")),
                "context_prefix": p.get("context_prefix", ""),
                # Parent window — adjacent clauses for full context
                "parent_text":    p.get("parent_text", ""),
                "has_parent_window": p.get("has_parent_window", False),
            })

        return {
            "query":                query,
            "domain_filter":        domain_filter,
            "use_hyde":             use_hyde,
            "retrieval_time_sec":   round(elapsed, 3),
            "evidence":             strength,
            "insufficient_evidence": not strength["is_sufficient"],
            "chunks":               output,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI — Test the pipeline with representative queries from both domains
# ═══════════════════════════════════════════════════════════════════════════

def main():
    pipeline = RetrievalPipeline()

    test_queries = [
        # Healthcare clinical
        {
            "query":  "What is the recommended HbA1c target for non-pregnant adults with diabetes?",
            "domain": None,   # unrestricted — should naturally return HC
        },
        # Healthcare regulatory
        {
            "query":  "What are the HIPAA requirements for protecting electronic PHI?",
            "domain": None,
        },
        # Old corpus
        {
            "query":  "What rights do data subjects have under GDPR Article 17?",
            "domain": None,
        },
        # Cross-domain
        {
            "query":  "How do FDA regulations differ from GDPR for health data?",
            "domain": None,
        },
        # Explicit domain filter test
        {
            "query":  "What are the screening recommendations for colorectal cancer?",
            "domain": "healthcare",   # force healthcare only
        },
    ]

    print("=" * 60)
    print("  RETRIEVAL V2 TEST")
    print("=" * 60)

    for item in test_queries:
        q      = item["query"]
        domain = item["domain"]
        label  = f"[domain_filter={domain}]" if domain else "[no filter]"

        print(f"\n  Query: {q}")
        print(f"  {label}")
        print("-" * 60)

        result = pipeline.retrieve(q, domain_filter=domain, verbose=True)
        print()

        if result["insufficient_evidence"]:
            ev = result["evidence"]
            print(
                f"  ⚠  Insufficient evidence: "
                f"top_rerank={ev['top_rerank_score']:.4f} "
                f"top_rrf={ev['top_rrf_score']:.4f}"
            )

        for r in result["chunks"]:
            tier_tag   = f"[{r['tier']}]" if r["tier"] else ""
            domain_tag = f"[{r['domain']}]"
            sources    = "+".join(r["sources"])
            print(
                f"  #{r['rank']} {tier_tag}{domain_tag} "
                f"[{sources}] rerank={r['rerank_score']:.4f}"
            )
            print(f"     {r['source_file'][:60]}")
            if r["source_section"]:
                print(f"     section: {r['source_section'][:60]}")
            print(f"     {r['text'][:120]}...")
            print()

        print(f"  Time: {result['retrieval_time_sec']}s")
        print()


if __name__ == "__main__":
    main()
