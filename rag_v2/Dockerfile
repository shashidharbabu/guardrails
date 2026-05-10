# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_V2_BASE_DIR=/app/rag_v2 \
    RAG_V2_BM25_PATH=/app/rag_v2/bm25_combined.pkl \
    RAG_V2_OLD_CHUNKS_PATH=/app/rag_v2/data/enriched_chunks.json \
    RAG_V2_HC_CHUNKS_PATH=/app/rag_v2/data/healthcare_enriched_chunks.jsonl \
    RAG_V2_QDRANT_PATH=/app/rag_v2/indexes_qdrant_data \
    RAG_V2_DENSE_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-rag.txt ./
RUN pip install --no-cache-dir -r requirements-rag.txt

COPY rag_v2/ ./rag_v2/

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8010/health || exit 1

CMD ["uvicorn", "rag_v2.rag_server:app", "--host", "0.0.0.0", "--port", "8010"]
