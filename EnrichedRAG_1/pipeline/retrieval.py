#!/usr/bin/env python3
"""
retrieval.py — Hybrid retrieval pipeline for Enterprise Guardrails RAG.

Pipeline:
  1. Dense retrieval      — Qwen3-Embedding-4B embeds the raw query → Qdrant top-20 cosine
  2. Sparse retrieval     — BM25 on raw query (exact term matching) → top-20 by BM25 score
  3. Reciprocal Rank Fusion — Merges dense + sparse results → top-30 candidates
  4. BGE Reranker         — BAAI/bge-reranker-v2-m3 cross-encoder rescores all 30 against raw query
  5. Tier Boost           — If T0 chunk in top-10 but not top-5, force include it (drop lowest non-T0)
  6. Weak evidence check  — conservatively flags weak retrieval instead of pretending confidence
  7. Output               — Top-5 enriched chunks with source citations and metadata

HyDE support remains available but is disabled by default until evaluation proves it helps.
Designed to run on Colab Pro (A100). Qdrant local disk mode.
"""

import json
import pickle
import os
import time
import re
from pathlib import Path

import torch
import numpy as np
import requests

# ── Config ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(os.environ.get("RAG_BASE_DIR", "."))
BM25_PATH = BASE_DIR / "bm25_index.pkl"
ENRICHED_CHUNKS_PATH = BASE_DIR / "rechunked_output" / "enriched_chunks.json"

# Qdrant
QDRANT_MODE = os.environ.get("QDRANT_MODE", "local")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_data")
COLLECTION_NAME = "guardrails_rag"

# Models
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# HyDE — MiniMax M2.7 via OpenRouter
HYDE_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HYDE_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HYDE_MODEL = "minimax/minimax-m2.7"

# Retrieval parameters
DENSE_TOP_K = 20       # candidates from Qdrant
SPARSE_TOP_K = 20      # candidates from BM25
RRF_K = 60             # RRF constant (standard value)
RERANK_CANDIDATES = 30 # feed to reranker after RRF
FINAL_TOP_K = 5        # final output to MAD agents
DEFAULT_USE_HYDE = os.environ.get("USE_HYDE", "false").lower() == "true"
MIN_RERANK_SCORE = float(os.environ.get("MIN_RERANK_SCORE", "0.15"))
MIN_TOP_RRF_SCORE = float(os.environ.get("MIN_TOP_RRF_SCORE", "0.013"))
MIN_SOURCE_COVERAGE = int(os.environ.get("MIN_SOURCE_COVERAGE", "1"))

# LangSmith tracing (optional)
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.environ.get("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_PROJECT", "guardrails-rag")

try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(func):
            return func
        return decorator


# ═══════════════════════════════════════════════════════════════════════════
# STEP 0: Optional HyDE — kept available but disabled by default
# ═══════════════════════════════════════════════════════════════════════════
# Instead of embedding the raw user query (which is short and vague),
# we ask MiniMax M2.7 to generate a hypothetical answer. That answer
# looks like a real chunk, so its embedding lands closer to relevant
# chunks in vector space. Raw query is still used for BM25.
# ═══════════════════════════════════════════════════════════════════════════

@traceable(name="hyde_expansion", run_type="llm")
def generate_hyde_answer(query: str, timeout: int = 25) -> str:
    """Generate a hypothetical document passage that answers the query."""
    if not HYDE_API_KEY:
        # No API key → skip HyDE, return raw query
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
                    "You are a regulatory compliance expert. Given a question, "
                    "write a short passage (100-150 words) that would appear in "
                    "a regulation or guideline and directly answers the question. "
                    "Write as if quoting from the actual document. "
                    "Do not say 'according to' or reference the question."
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
# Embeds the query text and searches Qdrant by cosine
# similarity. Returns top-20 candidates with scores.
# ═══════════════════════════════════════════════════════════════════════════

class DenseRetriever:
    """Qwen3-Embedding-4B for query embedding + Qdrant for search."""

    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        from qdrant_client import QdrantClient

        print("  Loading embedding model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
        except Exception:
            # Older local transformers/tokenizers builds can fail on the fast
            # Qwen tokenizer JSON. Fall back to the slow tokenizer so local
            # hybrid retrieval still works.
            self.tokenizer = AutoTokenizer.from_pretrained(
                EMBED_MODEL_NAME,
                trust_remote_code=True,
                use_fast=False,
            )
        self.model = AutoModel.from_pretrained(
            EMBED_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

        print("  Connecting to Qdrant...")
        if QDRANT_MODE == "server":
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        else:
            self.client = QdrantClient(path=QDRANT_PATH)

        info = self.client.get_collection(COLLECTION_NAME)
        print(f"  Dense retriever ready: {info.points_count} vectors on {self.device}")

    @torch.no_grad()
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text. Returns (dim,) float32 array."""
        encoded = self.tokenizer(
            [text], padding=True, truncation=True, max_length=8192, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().float().numpy()[0]

    @traceable(name="dense_retrieval", run_type="retriever")
    def retrieve(self, query_embedding: np.ndarray, top_k: int = DENSE_TOP_K) -> list[dict]:
        """Search Qdrant and return top-k results with scores."""
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
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
# BM25 uses exact term matching — great for specific regulation names,
# article numbers, and technical terms that embedding models might miss.
# Uses the RAW query (not HyDE) because BM25 needs actual search terms,
# not a generated passage.
# ═══════════════════════════════════════════════════════════════════════════

class SparseRetriever:
    """BM25 retriever using prebuilt index."""

    def __init__(self, chunks: list[dict]):
        print("  Loading BM25 index...")
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)

        self.bm25 = bm25_data["bm25"]
        self.chunk_ids = bm25_data["chunk_ids"]
        # Build lookup from chunk_id to chunk data
        self.chunk_map = {c["chunk_id"]: c for c in chunks}
        print(f"  BM25 retriever ready: {len(self.chunk_ids)} documents")

    @traceable(name="sparse_retrieval", run_type="retriever")
    def retrieve(self, raw_query: str, top_k: int = SPARSE_TOP_K) -> list[dict]:
        """BM25 search on raw query. Returns top-k results with scores."""
        query_tokens = tokenize_for_bm25(raw_query)
        scores = self.bm25.get_scores(query_tokens)
        top_idxs = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_idxs:
            if scores[idx] <= 0:
                continue
            cid = self.chunk_ids[idx]
            chunk = self.chunk_map.get(cid, {})
            results.append({
                "chunk_id": cid,
                "score": float(scores[idx]),
                "source": "sparse",
                "payload": chunk,
            })

        return results


def tokenize_for_bm25(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Reciprocal Rank Fusion (RRF)
# ═══════════════════════════════════════════════════════════════════════════
# Merges dense and sparse results using rank-based scoring.
# Each result gets score = 1/(k + rank), then scores are summed across
# both lists. This is robust because it uses RANK not raw score —
# dense cosine (0-1) and BM25 scores (0-30+) are on different scales.
# k=60 is the standard constant that prevents top-ranked items from
# dominating too heavily.
# ═══════════════════════════════════════════════════════════════════════════

@traceable(name="rrf_fusion", run_type="chain")
def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = RRF_K,
    top_n: int = RERANK_CANDIDATES,
) -> list[dict]:
    """Merge dense and sparse results using RRF. Returns top-n candidates."""
    rrf_scores = {}   # chunk_id → cumulative RRF score
    payloads = {}     # chunk_id → payload (keep the richer one)
    sources = {}      # chunk_id → set of sources

    for rank, r in enumerate(dense_results, 1):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)
        payloads[cid] = r["payload"]
        sources.setdefault(cid, set()).add("dense")

    for rank, r in enumerate(sparse_results, 1):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)
        if cid not in payloads:
            payloads[cid] = r["payload"]
        sources.setdefault(cid, set()).add("sparse")

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_n]

    return [
        {
            "chunk_id": cid,
            "rrf_score": rrf_scores[cid],
            "sources": sources[cid],
            "payload": payloads[cid],
        }
        for cid in sorted_ids
    ]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: BGE Reranker — Cross-encoder rescoring
# ═══════════════════════════════════════════════════════════════════════════
# The reranker is a cross-encoder that reads (query, candidate) pairs
# together and outputs a relevance score. Unlike bi-encoder embeddings
# (which encode query and document separately), cross-encoders see both
# at once — so they catch subtle relevance signals that cosine similarity
# misses. We only run it on 30 candidates (not 5,903) because it's slow.
# ═══════════════════════════════════════════════════════════════════════════

class BGEReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder reranker."""

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
    def rerank(self, query: str, candidates: list[dict], top_k: int = RERANK_CANDIDATES) -> list[dict]:
        """Score each candidate against the raw query. Returns rescored + sorted list."""
        if not candidates:
            return []

        # Build (query, chunk_text) pairs for cross-encoder
        pairs = []
        for c in candidates:
            text = c["payload"].get("enriched_text", c["payload"].get("text", ""))
            pairs.append((query, text[:2048]))  # truncate to avoid OOM

        # Tokenize all pairs at once
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # Get relevance scores
        outputs = self.model(**encoded)
        scores = outputs.logits.squeeze(-1).cpu().float().numpy()

        # Attach scores and sort
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Tier Boost — Prioritize authoritative sources
# ═══════════════════════════════════════════════════════════════════════════
# T0 chunks (ISO 27001, NIST CSF, NIST SSDF) are foundational documents.
# If a T0 chunk scored well enough to be in top-10 after reranking but
# just missed the top-5 cutoff, we force it in and drop the lowest-scoring
# non-T0 chunk. This prevents verbose T1/T2/T3 docs from crowding out
# the most authoritative sources.
# ═══════════════════════════════════════════════════════════════════════════

@traceable(name="tier_boost", run_type="chain")
def apply_tier_boost(reranked: list[dict], final_k: int = FINAL_TOP_K) -> list[dict]:
    """If T0 chunks are in top-10 but not top-5, force them in."""
    top_5 = reranked[:final_k]
    rest = reranked[final_k:final_k + 5]  # positions 6-10

    # Find T0 chunks in positions 6-10 that should be boosted
    t0_in_rest = [c for c in rest if c["payload"].get("tier") == "T0"]

    if not t0_in_rest:
        return top_5

    # Find non-T0 chunks in top-5 that can be replaced (lowest rerank score)
    non_t0_in_top5 = [c for c in top_5 if c["payload"].get("tier") != "T0"]
    non_t0_in_top5.sort(key=lambda x: x["rerank_score"])

    for t0_chunk in t0_in_rest:
        if non_t0_in_top5:
            # Replace the lowest-scoring non-T0 with this T0
            replaced = non_t0_in_top5.pop(0)
            top_5.remove(replaced)
            top_5.append(t0_chunk)

    # Re-sort by rerank score
    top_5.sort(key=lambda x: x["rerank_score"], reverse=True)
    return top_5[:final_k]


def build_citation(payload: dict) -> str:
    """Build a compact source citation string for downstream answering."""
    doc_name = payload.get("doc_name") or "unknown_source"
    article = payload.get("article_number")
    source_url = payload.get("source_url")

    parts = [doc_name]
    if article not in (None, "", 0):
        parts.append(f"Article {article}")
    if source_url:
        parts.append(source_url)
    return " | ".join(str(part) for part in parts)


def extract_result_payload(item: dict) -> dict:
    """Return the metadata payload regardless of whether the result is raw or already formatted."""
    payload = item.get("payload")
    return payload if isinstance(payload, dict) else item


def extract_source_identity(item: dict) -> str:
    """
    Build a stable source identity for evidence checks.
    Works for both raw reranker candidates and formatted output rows.
    """
    payload = extract_result_payload(item)
    doc_name = payload.get("doc_name")
    if doc_name:
        return str(doc_name)

    source_url = payload.get("source_url")
    if source_url:
        return str(source_url)

    chunk_id = item.get("chunk_id") or payload.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"

    return ""


@traceable(name="weak_retrieval_assessment", run_type="chain")
def assess_retrieval_strength(results: list[dict]) -> dict:
    """
    Conservatively flag weak retrieval.
    This is intentionally mild so it does not suppress good answers unnecessarily.
    """
    if not results:
        return {
            "is_sufficient": False,
            "reason": "no_results",
            "top_rerank_score": 0.0,
            "top_rrf_score": 0.0,
            "unique_sources": 0,
        }

    top = results[0]
    unique_sources = len(
        {
            extract_source_identity(item)
            for item in results
            if extract_source_identity(item)
        }
    )
    top_rerank = float(top.get("rerank_score", 0.0))
    top_rrf = float(top.get("rrf_score", 0.0))

    is_sufficient = (
        top_rerank >= MIN_RERANK_SCORE
        and top_rrf >= MIN_TOP_RRF_SCORE
        and unique_sources >= MIN_SOURCE_COVERAGE
    )

    reason = "ok" if is_sufficient else "weak_retrieval"
    return {
        "is_sufficient": is_sufficient,
        "reason": reason,
        "top_rerank_score": top_rerank,
        "top_rrf_score": top_rrf,
        "unique_sources": unique_sources,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE — Ties all steps together
# ═══════════════════════════════════════════════════════════════════════════

class RetrievalPipeline:
    """End-to-end retrieval: query → 5 enriched chunks for MAD agents."""

    def __init__(self):
        print("=" * 60)
        print("  RETRIEVAL PIPELINE — Enterprise Guardrails RAG")
        print("=" * 60)

        # Load chunks for BM25 lookup
        print("\n  Loading enriched chunks...")
        if not ENRICHED_CHUNKS_PATH.exists():
            raise FileNotFoundError(f"Missing enriched chunks file: {ENRICHED_CHUNKS_PATH}")
        if not BM25_PATH.exists():
            raise FileNotFoundError(f"Missing BM25 index file: {BM25_PATH}")
        with open(ENRICHED_CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"  Loaded {len(self.chunks)} chunks")

        self.dense = DenseRetriever()
        self.sparse = SparseRetriever(self.chunks)
        self.reranker = BGEReranker()
        print("\n  Pipeline ready.\n")

    @traceable(name="full_retrieval", run_type="chain")
    def retrieve(self, query: str, use_hyde: bool = DEFAULT_USE_HYDE, verbose: bool = False) -> dict:
        """
        Full retrieval pipeline.

        Args:
            query: Raw user query
            use_hyde: Whether to use HyDE expansion for dense retrieval
            verbose: Print intermediate results

        Returns:
            Retrieval result bundle with chunks, citations, and evidence status
        """
        t0 = time.time()

        # Optional HyDE expansion. Disabled by default.
        if use_hyde:
            query_for_dense = generate_hyde_answer(query)
            if verbose:
                print(f"  [HyDE] Generated {len(query_for_dense.split())} word hypothetical answer")
        else:
            query_for_dense = query

        # Step 1: Dense retrieval
        query_embedding = self.dense.embed_query(query_for_dense)
        dense_results = self.dense.retrieve(query_embedding)
        if verbose:
            top_score = dense_results[0]["score"] if dense_results else 0.0
            print(f"  [Dense] {len(dense_results)} candidates, top score: {top_score:.4f}")

        # Step 2: Sparse retrieval
        sparse_results = self.sparse.retrieve(query)
        if verbose:
            top_score = sparse_results[0]["score"] if sparse_results else 0.0
            print(f"  [Sparse] {len(sparse_results)} candidates, top score: {top_score:.4f}")

        # Step 3: RRF fusion
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        if verbose:
            both = sum(1 for c in fused if len(c["sources"]) == 2)
            print(f"  [RRF] {len(fused)} candidates ({both} from both sources)")

        # Step 4: BGE reranker
        reranked = self.reranker.rerank(query, fused)
        if verbose:
            top_score = reranked[0]["rerank_score"] if reranked else 0.0
            print(f"  [Rerank] Top score: {top_score:.4f}")

        # Step 5: Tier boost
        final = apply_tier_boost(reranked)
        strength = assess_retrieval_strength(final)

        elapsed = time.time() - t0
        if verbose:
            print(f"  [Done] {len(final)} chunks in {elapsed:.2f}s")
            print(
                "  [Evidence] "
                f"sufficient={strength['is_sufficient']} "
                f"top_rerank={strength['top_rerank_score']:.4f} "
                f"top_rrf={strength['top_rrf_score']:.4f} "
                f"unique_sources={strength['unique_sources']}"
            )

        # Format output
        output = []
        for rank, c in enumerate(final, 1):
            output.append({
                "rank": rank,
                "chunk_id": c["chunk_id"],
                "rerank_score": c["rerank_score"],
                "rrf_score": c["rrf_score"],
                "sources": list(c["sources"]),
                "tier": c["payload"].get("tier", ""),
                "doc_name": c["payload"].get("doc_name", ""),
                "article_number": c["payload"].get("article_number", ""),
                "text": c["payload"].get("text", ""),
                "enriched_text": c["payload"].get("enriched_text", ""),
                "context_prefix": c["payload"].get("context_prefix", ""),
                "source_url": c["payload"].get("source_url", ""),
                "citation": build_citation(c["payload"]),
            })

        return {
            "query": query,
            "use_hyde": use_hyde,
            "retrieval_time_sec": round(elapsed, 3),
            "evidence": strength,
            "insufficient_evidence": not strength["is_sufficient"],
            "chunks": output,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI — Test the pipeline with sample queries
# ═══════════════════════════════════════════════════════════════════════════

def main():
    pipeline = RetrievalPipeline()

    test_queries = [
        "What are the requirements for information security management under ISO 27001?",
        "What rights do data subjects have under GDPR?",
        "How should AI systems be assessed for bias and fairness?",
        "What are the NIST cybersecurity framework core functions?",
        "HIPAA requirements for protecting patient health information",
    ]

    print("=" * 60)
    print("  RETRIEVAL TEST")
    print("=" * 60)

    for q in test_queries:
        print(f"\n  Query: {q}")
        print("-" * 60)
        results = pipeline.retrieve(q, use_hyde=DEFAULT_USE_HYDE, verbose=True)
        print()
        if results["insufficient_evidence"]:
            print("  Insufficient evidence for a confident answer.")
        for r in results["chunks"]:
            tier_tag = f"[{r['tier']}]" if r['tier'] else ""
            sources_tag = "+".join(r["sources"])
            print(f"  #{r['rank']} {tier_tag} [{sources_tag}] score={r['rerank_score']:.4f}")
            print(f"     {r['doc_name'][:50]}")
            print(f"     citation: {r['citation']}")
            print(f"     {r['text'][:100]}...")
            print()


if __name__ == "__main__":
    main()
