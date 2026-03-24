# rag/ — Real Qdrant Retriever for MAD Pipeline

Connects the MAD pipeline to your Qdrant cloud vector database.
Drop-in replacement for `multi_agent/rag_stub.py`.

---

## Files

```
rag/
├── __init__.py
├── config.py      — Qdrant URL, collection name, embedding model settings
├── embedder.py    — Loads Nemotron-8B, handles query prefix
├── retriever.py   — retrieve(query, top_k) → List[EvidenceChunk]
└── README.md
```

---

## Corpus facts (confirmed)

| Field | Value |
|-------|-------|
| Embedding model | `nvidia/llama-embed-nemotron-8b` |
| Vector size | 4096 |
| Distance metric | Cosine |
| Collection | `ai_governance_chunks_nemotron8b` |
| Total chunks | 4,664 |
| Query prefix | `Instruct: Retrieve relevant regulatory passage to answer the query\nQuery: ` |
| Chunk prefix | None (raw text only at ingest) |

---

## Setup

### 1. Install dependencies

```bash
pip install qdrant-client transformers torch accelerate
```

### 2. Set environment variables

```bash
export QDRANT_URL="https://e2e7b7d2-4927-4c61-a78d-61f9c4e024bb.us-east4-0.gcp.cloud.qdrant.io"
export QDRANT_API_KEY="your-api-key-here"
export HF_TOKEN="your-hf-token-here"   # if Nemotron-8B is gated on HuggingFace
```

The URL and API key default to values from the notebook — set them via env vars
rather than editing config.py so you never commit credentials to git.

### 3. Verify connection

```bash
python3 -c "
from rag.retriever import health_check
print(health_check())
"
```

Expected:
```
[RAG] Connecting to Qdrant ...
[RAG] Connected. Collection: ai_governance_chunks_nemotron8b
{'status': 'ok', 'collection': 'ai_governance_chunks_nemotron8b', 'vectors_count': 4664, 'points_count': 4664}
```

---

## Activate in MAD — one line change

In `multi_agent/rag_stub.py`, find the `retrieve()` function and replace
its **entire body** with this single import:

```python
def retrieve(
    query: str,
    top_k: int = TOP_K_CHUNKS,
) -> List[EvidenceChunk]:
    from rag.retriever import retrieve as _real_retrieve
    return _real_retrieve(query, top_k)
```

Or more simply, at the top of rag_stub.py after the existing imports, add:

```python
# Uncomment this line to use real Qdrant instead of TF-IDF stub:
# from rag.retriever import retrieve  # noqa: F401
```

Then comment out the TF-IDF retrieve() function body below it.

**Nothing else changes.** `agent_a.py`, `agent_b.py`, `judge.py`,
`debate_engine.py` all call `rag_stub.retrieve()` unchanged.

---

## Test end-to-end

```bash
# With Qdrant activated:
python -m multi_agent.run_test --example hipaa_encryption

# Verify storage:
python -m multi_agent.run_test --example hipaa_encryption --verify-storage
```

You should see real chunk_ids from your corpus
(e.g. `t0__iso__27001_2022_infosec__chunk_0042`) in the judge verdicts
instead of the built-in sample chunk IDs.

---

## How retrieval works

```
User query or claim text
        │
        ▼
embed_query(text)
  → prepends QUERY_PREFIX
  → tokenises (max 512 tokens)
  → runs Nemotron-8B last hidden state [:, -1]
  → L2 normalises
  → returns List[float] of length 4096
        │
        ▼
qdrant.query_points(collection, vector, limit=top_k)
  → cosine similarity search
  → returns top_k ScoredPoints
        │
        ▼
Map payload → EvidenceChunk(chunk_id, text, source, tier)
  → source = payload["source"]["document_name"]
  → tier   = 1 (neutral — T0/T1 corpus labels are batch IDs, not authority)
        │
        ▼
List[EvidenceChunk] returned to MAD agents
```

---

## Note on tier

The corpus stores tier as `"T0"` or `"T1"` strings. These are **batch/collection
labels** used when the corpus was organised, not authority rankings — OWASP
and ISO 27001 are both `T0`. MAD does not use tier for filtering in v0.1.
All chunks are returned as `tier=1` (neutral placeholder).

To add authority-based filtering in a future version, add an `authority_level`
field (int 1/2/3) to each Qdrant payload at ingest time and update
`rag/retriever.py` accordingly.

---

*Guardrails Gateway · SJSU CS298B 2025–26*
