# RAG Pipeline

Retrieval-Augmented Generation pipeline for the Guardrails Gateway. Retrieves relevant regulatory document chunks to ground the MAD pipeline's debate.

**Important:** The RAG corpus is NOT used to answer user queries — it is used to verify the LLM's answer after the fact. The MAD pipeline agents retrieve from this corpus to check whether the LLM's claims are accurate.

## Status
🚧 In Development

## Stack

| Component | Technology |
|-----------|-----------|
| Vector store | Qdrant (GCP-hosted) |
| Embedding model | Qwen3-4B (fine-tuned with LoRA — r=16, α=32, all layers, 3 epochs) |
| Retrieval strategy | Dense vector search → BM25 hybrid → cross-encoder reranker |
| Reranker | `ms-marco-MiniLM-L-6-v2` |
| Metrics | Recall@1: 94.9%, nDCG@10: 0.891 |
| Training data | 13,572 query-chunk pairs |

## Corpus

72+ regulatory documents across 4 authority tiers:

| Tier | Type | Examples |
|------|------|---------|
| 1 | Primary law / statute | GDPR, HIPAA, ADA, CCPA, HITECH |
| 2 | Official guidance | EDPB guidelines, HHS OCR, FTC guidance |
| 3 | Framework / best practice | NIST frameworks, OWASP Top 10, ISO standards |
| 4 | Commentary / unofficial | Industry papers |

**Primary evaluation domain:** Healthcare / Hospital.

## Chunk format (JSONL)

```json
{"chunk_id": "hipaa_164_312_p1", "text": "...", "source": "HIPAA_Security_Rule_45CFR164", "tier": 1}
```

Supported field aliases: `text/content/chunk_text`, `source/doc_id/filename`, `tier/authority_tier`, `chunk_id/id`.

## Integration with MAD pipeline

Set the `CHUNKS_JSONL_PATH` environment variable to point at your local JSONL export from Qdrant:

```bash
export CHUNKS_JSONL_PATH="/path/to/rag/chunks.jsonl"
```

If not set, `multi_agent/rag_stub.py` falls back to built-in Healthcare + GDPR/HIPAA sample chunks for testing.

To swap from the JSONL stub to live Qdrant in the MAD pipeline, replace the body of `multi_agent/rag_stub.retrieve()` with the Qdrant client call (see `multi_agent/README.md` for the exact snippet).

## Retrieval pipeline

```
User query / claim text
       ↓
Dense vector search (Qwen3-4B embeddings → Qdrant)
       ↓
BM25 hybrid re-score
       ↓
Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
       ↓
Top-K chunks returned to MAD agent
```

## Planned Components
- Qdrant collection initialisation and ingestion scripts
- Embedding generation pipeline (Qwen3-4B LoRA fine-tuned)
- BM25 hybrid retrieval integration
- Cross-encoder reranking layer
- Retrieval evaluation harness (Recall@K, nDCG@10)
- JSONL export utility for MAD pipeline integration
