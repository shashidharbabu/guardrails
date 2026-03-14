"""
config.py — central config for the MAD pipeline.
All values are overridable via environment variables.
"""
import os
from pathlib import Path

# ── Ollama / LLM ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY: str  = "ollama"                      # Ollama ignores this but openai client requires it
AGENT_MODEL: str     = os.getenv("AGENT_MODEL", "qwen2.5:7b")   # Agent A + Agent B
JUDGE_MODEL: str     = os.getenv("JUDGE_MODEL", "qwen2.5:7b")   # Judge Agent

# ── Tools ─────────────────────────────────────────────────────────────────────
TAVILY_API_KEY: str  = os.getenv("TAVILY_API_KEY", "")

# ── RAG / Chunks ──────────────────────────────────────────────────────────────
# Path to your local JSONL chunk file.
# Each line: {"chunk_id": "...", "text": "...", "source": "...", "tier": 1}
# Adjust CHUNKS_JSONL_PATH to point at your actual file.
_repo_root = Path(__file__).resolve().parent.parent
CHUNKS_JSONL_PATH: str = os.getenv(
    "CHUNKS_JSONL_PATH",
    str(_repo_root / "rag" / "chunks.jsonl")
)

# ── Debate settings ───────────────────────────────────────────────────────────
MAX_CYCLES: int             = int(os.getenv("MAX_CYCLES", "2"))
TOP_K_CHUNKS: int           = int(os.getenv("TOP_K_CHUNKS", "5"))
TOP_K_CHALLENGE_CHUNKS: int = int(os.getenv("TOP_K_CHALLENGE_CHUNKS", "3"))

# ── Routing thresholds ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD_HIGH: float = float(os.getenv("CONFIDENCE_THRESHOLD_HIGH", "0.8"))
CONFIDENCE_THRESHOLD_LOW: float  = float(os.getenv("CONFIDENCE_THRESHOLD_LOW", "0.4"))

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("MAD_API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("MAD_API_PORT", "8001"))
