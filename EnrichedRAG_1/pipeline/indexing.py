#!/usr/bin/env python3
"""
indexing.py — Dual indexing pipeline for Enterprise Guardrails RAG.

Dense:  Qwen3-Embedding-4B → Qdrant collection (cosine similarity)
Sparse: BM25 (rank_bm25) → bm25_index.pkl

Embeds enriched_text (context_prefix + chunk text) for dense.
Embeds raw text for BM25 (no context prefix — BM25 works on exact terms).

Designed to run on Colab Pro (A100). Qdrant runs locally in Docker.
"""

import json
import pickle
import time
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────

# Paths — adjust if running on Colab
BASE_DIR = Path(os.environ.get("RAG_BASE_DIR", "."))
INPUT_PATH = BASE_DIR / "rechunked_output" / "enriched_chunks.json"
BM25_OUTPUT = BASE_DIR / "bm25_index.pkl"

# Qdrant — uses local disk storage by default (no Docker needed on Colab)
# Set QDRANT_MODE=server to connect to a running Qdrant instance instead
QDRANT_MODE = os.environ.get("QDRANT_MODE", "local")   # "local" = disk, "server" = Docker/remote
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_data")  # disk path for local mode
DRIVE_BACKUP = os.environ.get("DRIVE_BACKUP", "")  # e.g. /content/drive/MyDrive/RAG_indexes
COLLECTION_NAME = "guardrails_rag"

# Embedding model
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
EMBED_DIM = 2560          # Qwen3-Embedding-4B output dimension
EMBED_BATCH_SIZE = 4      # safer default for Kaggle T4-class GPUs
MAX_SEQ_LEN = 8192        # Qwen3-Embedding supports up to 8192 tokens

# BM25
BM25_K1 = 1.5
BM25_B = 0.75

# Test mode — set to True to index only 50 chunks for validation
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"
TEST_LIMIT = 50


# ── Load chunks ─────────────────────────────────────────────────────────────

def load_chunks(path: Path, test_mode: bool = False) -> list[dict]:
    """Load enriched chunks from JSON or JSONL."""
    if path.suffix == ".jsonl":
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

    if test_mode:
        # Sample across tiers for representative test
        chunks = chunks[:TEST_LIMIT]
        print(f"  TEST MODE: Using {len(chunks)} chunks")

    print(f"  Loaded {len(chunks)} chunks from {path.name}")
    return chunks


# ── Dense: Qwen3-Embedding-4B ──────────────────────────────────────────────

class DenseEmbedder:
    """Qwen3-Embedding-4B wrapper for batch embedding."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        from transformers import AutoTokenizer, AutoModel

        print(f"\n  Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = self.model.eval()
        print(f"  Model loaded on {self.device}")
        print(f"  Embedding dimension: {EMBED_DIM}")

    @torch.no_grad()
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (batch_size, EMBED_DIM) float32 array."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        # Use last_hidden_state with attention mask for mean pooling
        attention_mask = encoded["attention_mask"]
        hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_hidden / sum_mask

        # L2 normalize for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        result = embeddings.cpu().float().numpy()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def embed_all(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        """Embed all texts in batches with progress bar."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding"):
            batch = texts[i:i + batch_size]
            embs = self.embed_batch(batch)
            all_embeddings.append(embs)
        return np.vstack(all_embeddings)


# ── Sparse: BM25 ───────────────────────────────────────────────────────────

def tokenize_for_bm25(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    # Remove very short tokens
    return [t for t in tokens if len(t) > 1]


def build_bm25_index(chunks: list[dict]) -> dict:
    """Build BM25 index from raw chunk text (not enriched — BM25 uses exact terms)."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: rank_bm25. Install it in Colab with:\n"
            "  pip install rank-bm25"
        ) from exc

    print("\n  Building BM25 index...")
    corpus = [tokenize_for_bm25(c["text"]) for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
    print(f"  BM25 index built: {len(corpus)} documents, avg {np.mean([len(d) for d in corpus]):.0f} tokens/doc")

    return {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "corpus": corpus,
    }


# ── Qdrant ──────────────────────────────────────────────────────────────────

def create_qdrant_collection(client, collection_name: str, dim: int):
    """Create or recreate Qdrant collection with cosine similarity."""
    from qdrant_client.models import Distance, VectorParams

    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        print(f"  Collection '{collection_name}' exists — recreating...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
        ),
    )
    print(f"  Created collection '{collection_name}' (dim={dim}, cosine)")


def upsert_to_qdrant(client, collection_name: str, chunks: list[dict], embeddings: np.ndarray, batch_size: int = 100):
    """Upload embeddings + metadata to Qdrant."""
    from qdrant_client.models import PointStruct

    print(f"\n  Uploading {len(chunks)} vectors to Qdrant...")
    total = len(chunks)

    for i in tqdm(range(0, total, batch_size), desc="  Upserting"):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        points = []
        for j, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings)):
            # Build metadata payload
            payload = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk.get("doc_id", ""),
                "doc_name": chunk.get("doc_name", ""),
                "tier": chunk.get("tier", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "token_count": chunk.get("token_count", 0),
                "article_number": chunk.get("article_number", ""),
                "has_overlap": chunk.get("has_overlap", False),
                "starts_at_boundary": chunk.get("starts_at_boundary", False),
                "source_url": chunk.get("source_url", ""),
                "text": chunk["text"],
                "enriched_text": chunk.get("enriched_text", chunk["text"]),
                "context_prefix": chunk.get("context_prefix", ""),
            }

            points.append(PointStruct(
                id=i + j,
                vector=emb.tolist(),
                payload=payload,
            ))

        client.upsert(collection_name=collection_name, points=points)

    # Verify
    info = client.get_collection(collection_name)
    print(f"  Qdrant collection '{collection_name}': {info.points_count} points indexed")


# ── Validation ──────────────────────────────────────────────────────────────

def validate_indexing(client, collection_name: str, embedder: DenseEmbedder, bm25_data: dict, chunks: list[dict]):
    """Run quick sanity checks on both indexes."""
    print("\n" + "=" * 60)
    print("  VALIDATION")
    print("=" * 60)

    # Test 1: Dense retrieval — query should return relevant chunks
    test_queries = [
        "What are the requirements for information security management under ISO 27001?",
        "GDPR data subject rights and consent requirements",
        "AI risk management framework bias and fairness",
    ]

    print("\n  Dense retrieval test (Qdrant):")
    for q in test_queries:
        q_emb = embedder.embed_batch([q])[0].tolist()
        results = client.query_points(
            collection_name=collection_name,
            query=q_emb,
            limit=3,
        ).points
        print(f"\n  Query: {q[:60]}...")
        for r in results:
            print(f"    [{r.score:.4f}] {r.payload['doc_name'][:40]} | {r.payload['text'][:80]}...")

    # Test 2: BM25 retrieval
    print("\n\n  BM25 retrieval test:")
    bm25 = bm25_data["bm25"]
    chunk_ids = bm25_data["chunk_ids"]

    for q in test_queries:
        q_tokens = tokenize_for_bm25(q)
        scores = bm25.get_scores(q_tokens)
        top_idxs = np.argsort(scores)[-3:][::-1]

        print(f"\n  Query: {q[:60]}...")
        for idx in top_idxs:
            cid = chunk_ids[idx]
            chunk = next((c for c in chunks if c["chunk_id"] == cid), None)
            if chunk:
                print(f"    [{scores[idx]:.4f}] {chunk['doc_name'][:40]} | {chunk['text'][:80]}...")

    # Test 3: Tier distribution in Qdrant
    print("\n\n  Tier distribution in Qdrant:")
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    for tier in ["T0", "T1", "T2", "T3"]:
        try:
            count = client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="tier", match=MatchValue(value=tier))]
                ),
            )
            print(f"    {tier}: {count.count} chunks")
        except Exception:
            # Fallback: scroll and count manually
            results, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="tier", match=MatchValue(value=tier))]
                ),
                limit=10000,
            )
            print(f"    {tier}: {len(results)} chunks")

    print("\n  Validation complete.")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  INDEXING PIPELINE — Enterprise Guardrails RAG")
    print("  Dense: Qwen3-Embedding-4B → Qdrant")
    print("  Sparse: BM25 → bm25_index.pkl")
    print("=" * 60)

    # Check test mode
    if TEST_MODE:
        print("\n  *** TEST MODE — indexing only 50 chunks ***")

    # 1. Load chunks
    chunks = load_chunks(INPUT_PATH, test_mode=TEST_MODE)

    # Prepare texts for embedding
    # Dense: use enriched_text (context_prefix + original text)
    dense_texts = [c.get("enriched_text", c["text"]) for c in chunks]
    print(f"  Dense texts prepared: {len(dense_texts)} (using enriched_text)")

    # 2. Dense embedding
    embedder = DenseEmbedder()
    t0 = time.time()
    embeddings = embedder.embed_all(dense_texts)
    embed_time = time.time() - t0
    print(f"  Embedding complete: {embeddings.shape} in {embed_time:.1f}s")
    print(f"  Throughput: {len(dense_texts) / embed_time:.0f} chunks/sec")

    # 3. BM25 index
    bm25_data = build_bm25_index(chunks)
    with open(BM25_OUTPUT, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"  BM25 saved to {BM25_OUTPUT} ({BM25_OUTPUT.stat().st_size / 1024 / 1024:.1f} MB)")

    # 4. Qdrant upload
    from qdrant_client import QdrantClient
    if QDRANT_MODE == "server":
        print(f"\n  Connecting to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    else:
        print(f"\n  Using local Qdrant storage at {QDRANT_PATH}")
        client = QdrantClient(path=QDRANT_PATH)

    create_qdrant_collection(client, COLLECTION_NAME, EMBED_DIM)
    upsert_to_qdrant(client, COLLECTION_NAME, chunks, embeddings)

    # 5. Save embeddings to numpy (for backup/reuse without re-embedding)
    emb_path = BASE_DIR / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"  Embeddings saved to {emb_path} ({emb_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 6. Backup to Google Drive if mounted
    if DRIVE_BACKUP:
        import shutil
        backup_dir = Path(DRIVE_BACKUP)
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BM25_OUTPUT, backup_dir / "bm25_index.pkl")
        shutil.copy2(emb_path, backup_dir / "embeddings.npy")
        if Path(QDRANT_PATH).exists():
            shutil.copytree(QDRANT_PATH, backup_dir / "qdrant_data", dirs_exist_ok=True)
        print(f"  Backed up to Google Drive: {backup_dir}")

    # 7. Validate
    validate_indexing(client, COLLECTION_NAME, embedder, bm25_data, chunks)

    # 8. Summary
    info = client.get_collection(COLLECTION_NAME)
    print("\n" + "=" * 60)
    print("  INDEXING COMPLETE")
    print("=" * 60)
    print(f"  Chunks indexed:         {info.points_count}")
    print(f"  Qdrant collection:      {COLLECTION_NAME}")
    print(f"  Embedding model:        {EMBED_MODEL_NAME}")
    print(f"  Embedding dimension:    {EMBED_DIM}")
    print(f"  Embedding time:         {embed_time:.1f}s")
    print(f"  BM25 index:             {BM25_OUTPUT}")
    print(f"  Device used:            {embedder.device}")
    print("=" * 60)


if __name__ == "__main__":
    main()
