# NewRAG v2

Hybrid retrieval package for V4MAD.

This package includes:
- `retrieval_v2.py` - dense + BM25 + reranker retrieval pipeline
- `rag_service.py` - task-oriented service wrapper for LLM, CoT, and MAD/CoD retrieval
- `bm25_combined.pkl` - sparse BM25 index
- `data/enriched_chunks.json` - regulatory chunk payloads
- `data/healthcare_enriched_chunks.jsonl` - healthcare chunk payloads
- `indexes_qdrant_data/` - expected local Qdrant storage path

The Qdrant `storage.sqlite` file is not committed because it is about 304 MB and this
repository does not have Git LFS configured. Place the local artifact here:

```text
rag_v2/indexes_qdrant_data/collection/guardrails_rag_v2/storage.sqlite
```

Defaults can be overridden with:

```bash
QDRANT_MODE=cloud
QDRANT_URL=https://your-cluster.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_TIMEOUT=120
RAG_V2_BASE_DIR=/path/to/rag_v2
RAG_V2_QDRANT_PATH=/path/to/indexes_qdrant_data
RAG_V2_BM25_PATH=/path/to/bm25_combined.pkl
RAG_V2_OLD_CHUNKS_PATH=/path/to/enriched_chunks.json
RAG_V2_HC_CHUNKS_PATH=/path/to/healthcare_enriched_chunks.jsonl
RAG_V2_COLLECTION_NAME=guardrails_rag_v2
```

Migrate local Qdrant vectors to Cloud:

```bash
PYTHONPATH=. python rag_v2/scripts/migrate_qdrant_to_cloud.py \
  --local-path "/path/to/indexes_qdrant_data"
```

At runtime, dense retrieval comes from Qdrant Cloud while BM25 remains local:

```text
claim + original_query -> Qdrant Cloud dense + local BM25 -> RRF -> BGE rerank -> top 5 chunks
```

Smoke import:

```bash
PYTHONPATH=. python -c "from rag_v2 import get_service; print(get_service)"
```
