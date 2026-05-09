# guardrails_rag_v2 — RAG Package for MAD / CoT Pipelines

Hybrid RAG endpoint for the Guardrails enterprise pipeline.
Built for healthcare + regulatory document retrieval.
Powers Multi-Agent Debate (MAD), Chain-of-Thought (CoT) verification, and GRPO fine-tuning.

---

## What's in This Package (Google Drive Folder)

```
NewRAG/
├── indexes_qdrant_data/          ← Qdrant vector store (304 MB)
│   ├── meta.json                 ← collection config (dim=2560, cosine, HNSW m=16)
│   └── collection/
│       └── guardrails_rag_v2/
│           └── storage.sqlite    ← 9355 vectors (9299 chunks + 56 summary nodes)
│
├── bm25_combined.pkl             ← BM25 sparse index (28.7 MB, rank_bm25)
│                                    Tokenised corpus of all 9299 real chunks
│
├── healthcare_enriched_chunks.jsonl  ← Raw healthcare corpus chunks (18.8 MB)
│                                        3396 chunks, domain=healthcare
│                                        Sources: ADA 2024, AHA/ACC 2023, CDC Opioid,
│                                                 FDA CFR, NICE, EU MDR, HITECH, etc.
│
├── enriched_chunks.jsonl             ← Raw regulatory corpus chunks (23.9 MB)
│                                        5903 chunks, domain=general_regulatory
│                                        Sources: GDPR, EU AI Act, HIPAA, NIST,
│                                                 NIS2, CCPA, UK OSA, DSA, etc.
│
├── retrieval_v2.py               ← Hybrid retrieval pipeline (upload this too)
├── rag_service.py                ← RAG endpoint — 3 task-oriented methods (upload this)
└── NEWRAG_PACKAGE_README.md      ← This file
```

> **To use the RAG endpoint**: you need `indexes_qdrant_data/`, `bm25_combined.pkl`,
> `retrieval_v2.py`, and `rag_service.py`. The `.jsonl` files are raw source data —
> not required to run retrieval, but useful if you want to re-index or inspect chunks.

---

## The Indexes

### 1. `indexes_qdrant_data/` — Dense Vector Store

**What it is**: A local Qdrant on-disk collection (`guardrails_rag_v2`) containing
dense vector embeddings of all 9299 chunks plus 56 parent-window summary nodes.

**Vector config**:
| Field | Value |
|-------|-------|
| Embedding model | `Qwen/Qwen3-Embedding-4B` |
| Embedding dimension | 2560 |
| Distance metric | Cosine |
| HNSW m | 16 |
| HNSW ef_construct | 200 |
| Storage format | SQLite (Qdrant on-disk) |
| Total vectors | 9355 (9299 real + 56 summary) |

**How it was built**:
```
For each chunk in both corpora:
  1. Text cleaned and split (512 tokens, 64 overlap)
  2. Parent-window summaries generated (56 nodes — excluded from retrieval results)
  3. Qwen3-Embedding-4B embeds each chunk → 2560-dim vector
  4. Stored in Qdrant with payload: {chunk_id, text, source_file, domain, tier, is_summary}
```

Summary nodes are stored with `is_summary=True` and are always filtered out
during retrieval — they exist only as parent context for chunk payloads.

**How to run Qdrant locally**:

Option A — Docker (recommended):
```bash
docker run -p 6333:6333 \
  -v /path/to/indexes_qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

Option B — Qdrant binary:
```yaml
# config.yaml
storage:
  storage_path: /path/to/indexes_qdrant_data
```
```bash
./qdrant
```

Verify it loaded correctly:
```bash
curl http://localhost:6333/collections/guardrails_rag_v2
# Expected: {"result": {"status": "green", "vectors_count": 9355}}
```

---

### 2. `bm25_combined.pkl` — Sparse BM25 Index

**What it is**: A serialised `rank_bm25.BM25Okapi` object pre-fit on the tokenised
text of all 9299 real chunks (both corpora combined). Used for the sparse leg of
hybrid retrieval.

**How it was built**:
```python
from rank_bm25 import BM25Okapi
tokenised = [text.lower().split() for text in all_chunk_texts]  # 9299 docs
bm25 = BM25Okapi(tokenised)
pickle.dump(bm25, open("bm25_combined.pkl", "wb"))
```

**How to load**:
```python
import pickle
with open("bm25_combined.pkl", "rb") as f:
    bm25 = pickle.load(f)
scores = bm25.get_scores(query.lower().split())  # returns np.array of 9299 scores
```

---

## The Chunk Data

### `enriched_chunks.jsonl` — Regulatory Corpus (5903 chunks)

Old regulatory corpus covering global privacy, AI, and cybersecurity law.

**Documents indexed**:
| Regulation | Region |
|-----------|--------|
| GDPR (General Data Protection Regulation) | EU |
| EU AI Act | EU |
| NIS2 Directive | EU |
| Digital Services Act (DSA) | EU |
| HIPAA Privacy + Security Rules | US |
| HITECH Act | US |
| CCPA / CPRA | US |
| NIST AI RMF 1.0 | US |
| NIST Cybersecurity Framework 2.0 | US |
| UK Online Safety Act (OSA) | UK |
| SOC 2 Type II guidance | International |

**Chunk schema**:
```json
{
  "chunk_id":    "gdpr_art5_001",
  "text":        "Article 5 GDPR requires that personal data shall be...",
  "source_file": "gdpr_full.pdf",
  "domain":      "general_regulatory",
  "tier":        1,
  "parent_text": "..."
}
```

**Tier system** (used for boosting in retrieval):
- `tier 1` — Primary law / official regulation text
- `tier 2` — Regulatory guidance / official FAQ
- `tier 3` — Framework / standard (NIST, ISO)
- `tier 4` — Commentary / interpretation

---

### `healthcare_enriched_chunks.jsonl` — Healthcare Corpus (3396 chunks)

New healthcare corpus added in the v2 index build.

**Documents indexed**:
| Source | Type |
|--------|------|
| ADA Standards of Medical Care 2024 | Clinical guideline |
| AHA/ACC Cardiovascular Guidelines 2023 | Clinical guideline |
| CDC Opioid Prescribing Guidelines | Clinical guideline |
| FDA CFR Title 21 (Food & Drug) | Regulation |
| NICE Clinical Guidelines | Clinical guideline (UK) |
| EU MDR (Medical Device Regulation) | Regulation |
| HITECH Act (health IT provisions) | Regulation |
| CMS Conditions of Participation | Regulation |

**Chunk schema** (same as regulatory):
```json
{
  "chunk_id":    "ada2024_s6_012",
  "text":        "For adults with type 2 diabetes, the ADA recommends...",
  "source_file": "ada_standards_2024.pdf",
  "domain":      "healthcare",
  "tier":        1,
  "parent_text": "..."
}
```

Healthcare chunks have `tier=""` (empty) in the index — the tier boost in
retrieval only affects old regulatory chunks and does not penalise HC chunks.

---

## How Indexing Was Done

The full pipeline (run on Lightning AI T4 GPU):

```
Step 1 — Corpus preparation
  prepare_old_corpus.py   → enriched_chunks.jsonl      (5903 chunks)
  prepare_hc_corpus.py    → healthcare_enriched_chunks.jsonl  (3396 chunks)

  Each script:
    - Loads raw PDF/text → splits at 512 tokens with 64-token overlap
    - Generates parent-window context (1024-token surrounding text)
    - Assigns chunk_id, source_file, domain, tier
    - Writes JSONL

Step 2 — Build combined BM25 index
  build_combined_index.py
    - Loads both JSONL files (9299 chunks total)
    - Tokenises: text.lower().split()
    - Fits BM25Okapi on all 9299 docs
    - Saves → bm25_combined.pkl

Step 3 — Build Qdrant vector index
  indexing_v2.py
    - Loads both JSONL files
    - For each chunk:
        embed = Qwen3-Embedding-4B(chunk.text)    # 2560-dim
        qdrant.upsert(collection="guardrails_rag_v2",
                      vector=embed,
                      payload={chunk_id, text, source_file, domain, tier, is_summary=False})
    - Also generates 56 parent-window summary nodes (is_summary=True)
    - HNSW m=16, ef_construct=200

Total index build time on T4: ~3.5 hours (embedding 9355 vectors at 2560-dim)
```

**Why Qwen3-Embedding-4B (2560-dim)?**
Ablation against other embedding models showed significantly better
faithfulness on regulatory domain text. The high dimensionality (2560 vs
standard 768/1024) preserves more regulatory-specific semantic nuance.

---

## Retrieval Pipeline (`retrieval_v2.py`)

**Hybrid: Dense + Sparse → RRF → Rerank → Diversity**

```
Query
  │
  ├── Dense (Qwen3-4B)    top-20  ─────────────────┐
  │     Qdrant cosine search                        │
  │     Filter: is_summary=False                    │
  │                                                 ▼
  └── BM25 (rank_bm25)   top-20           RRF fusion (k=60)
        get_scores on raw query                     │
        Already excludes summary nodes              ▼
                                     BGE Reranker (bge-reranker-v2-m3)
                                       Cross-encoder, rescores top-30
                                                    │
                                                    ▼
                                     Tier boost
                                       tier=1 chunks in pos 6-10
                                       promoted into top-5 if displaced
                                                    │
                                                    ▼
                                     Source diversity
                                       ≤3 chunks per source_file
                                                    │
                                                    ▼
                                            top-5 final chunks
```

**No HyDE** — ablation confirmed no-HyDE wins:
```
no_hyde overall: 0.9485
hyde overall:    0.9417
```

**Optional: domain filter** — pass `domain_filter="healthcare"` or
`domain_filter="general_regulatory"` to restrict retrieval to one corpus.
Used internally by cross-doc retrieval.

---

## RAG Service (`rag_service.py`) — 3 Task-Oriented Methods

```python
from rag_service import get_service
svc = get_service()   # loads models once (~15s on GPU), returns singleton
```

All methods return the same consistent dict:
```python
{
  "chunks":   list[dict],   # full chunk objects with text, source_file, tier, etc.
  "context":  str,          # formatted context string — paste directly into LLM prompt
  "evidence": dict,         # {is_sufficient, top_rerank_score, unique_sources, ...}
  "meta":     dict,         # call metadata (task, query, k, agent_id, round, etc.)
}
```

---

### Method 1 — `retrieve_for_llm` — Baseline LLM Answer Generation

```python
result = svc.retrieve_for_llm(query, question_type="")
# question_type="cross_doc" → dual-domain retrieval (healthcare + regulatory)
```

**When to use**: Before a plain LLM generates its initial answer.
Also used as CoT Step 1 — the initial retrieval is identical for both.

Returns `k=5` chunks. `result["context"]` is ready to paste into your LLM prompt.

**Cross-doc retrieval** (automatic when `question_type="cross_doc"`):
For questions comparing two documents (e.g. "How does HIPAA differ from GDPR?"),
standard retrieval only surfaces one document's chunks. Cross-doc mode fetches
3 chunks from each domain separately, merges by rerank score, and returns top-5.
This prevents hallucination on the missing document.

```python
# Standard
result = svc.retrieve_for_llm("What are HIPAA encryption requirements?")

# Cross-doc (HIPAA vs GDPR, healthcare vs regulatory, etc.)
result = svc.retrieve_for_llm(
    "Compare HIPAA and GDPR patient data portability rights",
    question_type="cross_doc"
)
```

---

### Method 2 — `retrieve_for_cot` — CoT Per-Claim Verification

```python
result = svc.retrieve_for_cot(claim_text, original_query=query)
```

**When to use**: After extracting atomic claims from the baseline LLM answer.
Called ONCE PER CLAIM extracted from the baseline answer.

```python
# CoT flow
baseline = svc.retrieve_for_llm(query)
llm_answer = your_llm(baseline["context"])

# Extract claims from llm_answer → ["Claim 1...", "Claim 2...", ...]
for claim in claims:
    result = svc.retrieve_for_cot(claim, original_query=query)
    # result["chunks"] → 3 focused chunks for this specific claim
    # CoT agent reasons: SUPPORTED / PARTIAL / NOT_SUPPORTED / IDK + confidence
```

Returns `k=3` targeted chunks. The combined query internally is
`"{claim}. Context: {original_query}"` to anchor retrieval to the claim
while keeping query context.

---

### Method 3 — `retrieve_for_cod` — Multi-Agent Debate (CoD)

Called multiple times per debate session with different `agent_id` and `round_num`.

```python
# Round 0: shared initial chunks for both agents
result = svc.retrieve_for_cod(
    query, session_id="s1", agent_id="shared", round_num=0,
    question_type=""   # or "cross_doc"
)

# Round 1+: per-agent claim retrieval
result = svc.retrieve_for_cod(
    claim_text, session_id="s1", agent_id="agent_a", round_num=1
)
result = svc.retrieve_for_cod(
    challenge_text, session_id="s1", agent_id="agent_b", round_num=1
)

# Judge grounding
result = svc.retrieve_for_cod(
    query, session_id="s1", agent_id="judge", round_num=99
)
```

**k defaults by agent_id**:
| agent_id | k |
|----------|---|
| `shared` | 5 |
| `agent_a` | 3 |
| `agent_b` | 3 |
| `judge` | 3 |

`result["meta"]` includes: `session_id`, `agent_id`, `round_num`, `question_type`
for GRPO training log alignment.

---

## Prompt Helpers (exported from `rag_service.py`)

All system prompts and prompt builders are exported. Import, don't rewrite:

```python
from rag_service import (
    # System prompts
    LLM_SYSTEM,
    COT_ROUND1_SYSTEM, COT_ROUND2_SYSTEM, COT_ROUND3_SYSTEM,
    COD_AGENT_A_SYSTEM, COD_AGENT_B_SYSTEM,
    COD_REBUTTAL_A_SYSTEM, COD_REBUTTAL_B_SYSTEM,
    COD_JUDGE_SYSTEM,

    # Prompt builders
    build_llm_prompt,          # (query, context_str) → str
    build_cot_round1,          # (query, context_str, claim) → str
    build_cot_round2,          # (query, claim_chunks: dict, claims: list) → str
    build_cot_round3,          # (query, claim_chunks: dict, claims: list) → str
    build_cod_initial,         # (query, context_str) → str
    build_cod_rebuttal,        # (query, context_str, opponent_position) → str
    build_cod_judge,           # (query, context_str, debate_transcript) → str
)
```

---

## Quick Setup

### 1. Install dependencies

```bash
pip install qdrant-client torch transformers sentence-transformers rank-bm25
```

### 2. Start Qdrant

```bash
docker run -p 6333:6333 \
  -v /absolute/path/to/indexes_qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

### 3. Set environment variables

```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export COLLECTION_NAME=guardrails_rag_v2
export BM25_PATH=/absolute/path/to/bm25_combined.pkl
```

### 4. Use it

```python
from rag_service import get_service

svc = get_service()   # first call: downloads Qwen3-4B + BGE (~9GB total), ~15s on GPU

result = svc.retrieve_for_llm("What encryption does HIPAA require for PHI at rest?")
print(result["context"])   # paste into your LLM prompt
```

**Models downloaded automatically on first `get_service()` call:**
- `Qwen/Qwen3-Embedding-4B` — ~8 GB
- `BAAI/bge-reranker-v2-m3` — ~1.1 GB

GPU recommended (T4 or better). CPU works but retrieval takes ~10s per query vs ~1s on GPU.

---

## Files to Upload (what your friend needs)

Minimum set to run the RAG endpoint:
```
indexes_qdrant_data/      ← Qdrant collection (already on Drive)
bm25_combined.pkl         ← BM25 index (already on Drive)
retrieval_v2.py           ← retrieval logic (upload this)
rag_service.py            ← the endpoint (upload this)
```

Optional (for inspection / re-indexing):
```
healthcare_enriched_chunks.jsonl   ← already on Drive
enriched_chunks.jsonl              ← already on Drive
```

---

## Eval Results (guardrails_rag_v2, no_hyde, 75 questions)

Evaluated on 75 questions (30 regulatory + 45 new questions including cross-doc).
Judge: Gemini 2.5 Pro. Answer model: MiniMax M2.5.

| Metric | Score |
|--------|-------|
| faithfulness | ~0.942 |
| answer_relevance | ~0.970 |
| context_precision | ~0.880 |
| hallucination_score | ~0.040 |
| MRR@5 | ~0.785 |
| overall | ~0.934 |

Cross-doc retrieval enabled for comparison questions (Q056, Q065 type — HIPAA vs GDPR).
Without cross-doc mode, faithfulness on those questions was ~0.62. With it: ~0.91.
