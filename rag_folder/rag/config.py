"""
rag/config.py — configuration for the real Qdrant RAG retriever.

All values are overridable via environment variables.
Qdrant URL and API key must be set — either here or as env vars.
Do NOT commit real API keys to git. Use environment variables in production.
"""
import os

# ── Qdrant connection ──────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv(
    "QDRANT_URL",
    "https://e2e7b7d2-4927-4c61-a78d-61f9c4e024bb.us-east4-0.gcp.cloud.qdrant.io"
)
QDRANT_API_KEY: str = os.getenv(
    "QDRANT_API_KEY",
    ""   # set via env var — never hardcode in production
)
QDRANT_TIMEOUT: int = int(os.getenv("QDRANT_TIMEOUT", "60"))

# ── Collection ─────────────────────────────────────────────────────────────────
COLLECTION_NAME: str = os.getenv(
    "QDRANT_COLLECTION",
    "ai_governance_chunks_nemotron8b"
)
VECTOR_SIZE: int = 4096   # nvidia/llama-embed-nemotron-8b output dimension

# ── Embedding model ────────────────────────────────────────────────────────────
EMBED_MODEL: str = os.getenv(
    "EMBED_MODEL",
    "nvidia/llama-embed-nemotron-8b"
)
EMBED_MAX_LENGTH: int = 512   # max tokens per chunk/query — matches ingest

# Query prefix required by the Nemotron instruction-following embedding model.
# This MUST be prepended to every query at retrieval time.
# Chunks were ingested WITHOUT this prefix (only queries use it).
QUERY_PREFIX: str = (
    "Instruct: Retrieve relevant regulatory passage to answer the query\nQuery: "
)

# HuggingFace token — required if model is gated
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
