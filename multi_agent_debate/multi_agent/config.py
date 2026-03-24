"""
config.py — central configuration for the MAD pipeline.
All values are overridable via environment variables.
"""
import os
from pathlib import Path

# ── Ollama / LLM (Agent A + Agent B) ───────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY:  str = "ollama"
AGENT_MODEL:     str = os.getenv("AGENT_MODEL", "qwen2.5:7b")

# ── Judge model ─────────────────────────────────────────────────────────────────
# JUDGE_PROVIDER=anthropic → uses Claude via Anthropic SDK (recommended)
# JUDGE_PROVIDER=ollama    → uses Ollama model (fallback / offline)
JUDGE_PROVIDER:      str = os.getenv("JUDGE_PROVIDER", "ollama")
JUDGE_MODEL:         str = os.getenv("JUDGE_MODEL", "qwen2.5:7b")
ANTHROPIC_API_KEY:   str = os.getenv("ANTHROPIC_API_KEY", "")

# ── RAG / Chunks ───────────────────────────────────────────────────────────────
# Path to your local JSONL chunk file.
# Each line: {"chunk_id": "...", "text": "...", "source": "...", "tier": 1}
# If not found → falls back to built-in Healthcare + GDPR/HIPAA sample chunks.
_repo_root = Path(__file__).resolve().parent.parent
CHUNKS_JSONL_PATH: str = os.getenv(
    "CHUNKS_JSONL_PATH",
    str(_repo_root / "rag" / "chunks.jsonl")
)

# ── Debate settings ────────────────────────────────────────────────────────────
MAX_CYCLES:              int = int(os.getenv("MAX_CYCLES", "2"))
TOP_K_CHUNKS:            int = int(os.getenv("TOP_K_CHUNKS", "5"))
TOP_K_CHALLENGE_CHUNKS:  int = int(os.getenv("TOP_K_CHALLENGE_CHUNKS", "3"))

# ── Routing thresholds ─────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD_HIGH: float = float(os.getenv("CONFIDENCE_THRESHOLD_HIGH", "0.8"))
CONFIDENCE_THRESHOLD_LOW:  float = float(os.getenv("CONFIDENCE_THRESHOLD_LOW", "0.4"))

# ── Storage (SQLite — 4 tables per friend's GRPO spec) ────────────────────────
# All debate data is stored here for the feedback loop to consume.
# Tables: queries, claims, attacks, judge_verdicts
# The feedback loop (separate codebase) writes: b_reward, brier_reward, rewards table
DB_PATH: str = os.getenv("MAD_DB_PATH", str(_repo_root / "mad_store.db"))

# ── FastAPI ────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("MAD_API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("MAD_API_PORT", "8001"))
