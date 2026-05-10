import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent

# ── Models ─────────────────────────────────────────────────────────────────────
BASELINE_MODEL   = os.getenv("BASELINE_MODEL", "Qwen/Qwen2.5-3B-Instruct")    # want hallucinations — use smaller model
DECOMPOSER_MODEL = os.getenv("DECOMPOSER_MODEL", "Qwen/Qwen2.5-7B-Instruct")  # reliable JSON extraction
AGENTS_MODEL     = os.getenv("AGENTS_MODEL", "Qwen/Qwen2.5-14B-Instruct")     # both agents share this model + server
USE_GRPO_LORA_AGENTS = os.getenv("USE_GRPO_LORA_AGENTS", "0").lower() in {"1", "true", "yes"}
AGENT_A_MODEL    = os.getenv("AGENT_A_MODEL", "agent_a" if USE_GRPO_LORA_AGENTS else AGENTS_MODEL)
AGENT_B_MODEL    = os.getenv("AGENT_B_MODEL", "agent_b" if USE_GRPO_LORA_AGENTS else AGENTS_MODEL)
AGENT_A_LORA_PATH = Path(os.getenv("AGENT_A_LORA_PATH", str(ROOT / "adapters" / "grpo_agent_a" / "final")))
AGENT_B_LORA_PATH = Path(os.getenv("AGENT_B_LORA_PATH", str(ROOT / "adapters" / "grpo_agent_b" / "final")))
JUDGE_BACKEND = os.getenv("JUDGE_BACKEND", "vllm").lower()  # vllm | claude
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
CLAUDE_JUDGE_MODEL = os.getenv("CLAUDE_JUDGE_MODEL", "claude-sonnet-4-6")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── vLLM ports (one server runs at a time) ─────────────────────────────────────
BASELINE_PORT    = 8001
DECOMPOSER_PORT  = 8002
AGENTS_PORT      = 8003
JUDGE_PORT       = 8004
BASELINE_BASE_URL = os.getenv("BASELINE_BASE_URL", f"http://localhost:{BASELINE_PORT}/v1")
BASELINE_API_KEY = os.getenv("BASELINE_API_KEY", "EMPTY")
DECOMPOSER_BASE_URL = os.getenv("DECOMPOSER_BASE_URL", f"http://127.0.0.1:{DECOMPOSER_PORT}/v1")
DECOMPOSER_API_KEY = os.getenv("DECOMPOSER_API_KEY", "EMPTY")
AGENTS_BASE_URL  = os.getenv("AGENTS_BASE_URL", f"http://localhost:{AGENTS_PORT}/v1")
AGENTS_API_KEY   = os.getenv("AGENTS_API_KEY", "EMPTY")
JUDGE_BASE_URL   = os.getenv("JUDGE_BASE_URL", f"http://localhost:{JUDGE_PORT}/v1")
JUDGE_API_KEY    = os.getenv("JUDGE_API_KEY", "EMPTY")

# ── Temperatures ───────────────────────────────────────────────────────────────
BASELINE_TEMP    = 0.3    # some creativity so it generates plausible hallucinations
DECOMPOSER_TEMP  = 0.0    # deterministic JSON extraction
AGENT_A_TEMP     = float(os.getenv("AGENT_A_TEMP", "0.4"))    # strict verifier — precise
AGENT_B_TEMP     = float(os.getenv("AGENT_B_TEMP", "0.85"))   # skeptical challenger — creative, finds edge cases
JUDGE_TEMP       = 0.0    # deterministic verdict

# ── Token budgets ──────────────────────────────────────────────────────────────
BASELINE_MAX_TOKENS   = 512
DECOMPOSER_MAX_TOKENS = 1024
AGENT_MAX_TOKENS      = int(os.getenv("AGENT_MAX_TOKENS", "384"))
JUDGE_MAX_TOKENS      = 512

# ── Chunk assignment (by ranked position after per-claim reranking) ────────────
# Agent A: top-3 (highest relevance support)
# Agent B: rank-0 anchor + rank-2, rank-3 (different-perspective evidence)
# Judge  : all 5 (complete evidence pool)
AGENT_A_CHUNK_INDICES = [0, 1, 2]
AGENT_B_CHUNK_INDICES = [0, 2, 3]
JUDGE_CHUNK_INDICES   = [0, 1, 2, 3, 4]
QUERY_TOP_K           = 5    # how many chunks stored per query in source data

# ── RAG v2 / Qdrant Cloud (live per-claim retrieval) ───────────────────────────
MAD_USE_LIVE_RAG = os.getenv("MAD_USE_LIVE_RAG", "1").lower() in {"1", "true", "yes"}
RAG_V2_REMOTE_URL = os.getenv("RAG_V2_REMOTE_URL", "").rstrip("/")
RAG_V2_REMOTE_TIMEOUT = float(os.getenv("RAG_V2_REMOTE_TIMEOUT", "300"))
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "guardrails_rag_v2"

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH            = Path(os.getenv("MAD_DB_PATH", str(ROOT / "mad_v4.db")))
CHECKPOINT_DB      = Path(os.getenv("MAD_CHECKPOINT_DB", str(ROOT / "checkpoints.db")))
DATA_DIR           = ROOT / "data"
QUERIES_FILE       = DATA_DIR / "queries_50.json"
QUERY_CHUNKS_FILE  = DATA_DIR / "query_chunks_50.json"

# ── Concurrency ────────────────────────────────────────────────────────────────
CLAIM_CONCURRENCY = 2    # parallel claims within debate nodes
DECOMPOSER_MIN_CLAIMS = int(os.getenv("DECOMPOSER_MIN_CLAIMS", "3"))
DECOMPOSER_MAX_CLAIMS = int(os.getenv("DECOMPOSER_MAX_CLAIMS", "5"))
V4MAD_MAX_CLAIMS = int(os.getenv("V4MAD_MAX_CLAIMS", str(DECOMPOSER_MAX_CLAIMS)))  # 0 = no cap

# ── Langfuse ───────────────────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY = "pk-lf-2efefb74-7c5a-415e-9bb7-062ae2edc2d5"
LANGFUSE_SECRET_KEY = "sk-lf-9a48abf7-0969-4974-8dfd-b5d8108b5022"
LANGFUSE_HOST = "https://us.cloud.langfuse.com"
