import os
from pathlib import Path

try:
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache
except ModuleNotFoundError:
    from langchain.cache import SQLiteCache
    from langchain.globals import set_llm_cache

# Default Ollama endpoint — overridden by vLLM env vars when running on GPU
_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

BASELINE_MODEL_NAME = os.getenv("BASELINE_MODEL_NAME", "qwen2.5:7b")
DECOMPOSER_MODEL_NAME = os.getenv("DECOMPOSER_MODEL_NAME", "qwen2.5:7b")
AGENTS_MODEL_NAME = os.getenv("AGENTS_MODEL_NAME", "qwen2.5:7b")
# Per-role model names — used when vLLM serves both GRPO adapters via --lora-modules.
# If unset, both fall back to AGENTS_MODEL_NAME (Ollama base model).
AGENT_A_MODEL_NAME = os.getenv("AGENT_A_MODEL_NAME", AGENTS_MODEL_NAME)
AGENT_B_MODEL_NAME = os.getenv("AGENT_B_MODEL_NAME", AGENTS_MODEL_NAME)
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "qwen2.5:7b")

VLLM_BASELINE_URL = os.getenv("VLLM_BASELINE_URL", _OLLAMA_URL)
VLLM_DECOMPOSER_URL = os.getenv("VLLM_DECOMPOSER_URL", _OLLAMA_URL)
VLLM_AGENTS_URL = os.getenv("VLLM_AGENTS_URL", _OLLAMA_URL)
VLLM_JUDGE_URL = os.getenv("VLLM_JUDGE_URL", _OLLAMA_URL)

# DB stored next to app_sessions.db (repo_root/app/)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DB = str(_REPO_ROOT / "app" / "mad_v4.db")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", _DEFAULT_DB)
LANGCHAIN_CACHE_DB = os.getenv("LANGCHAIN_CACHE_DB", str(_REPO_ROOT / "app" / ".langchain_cache.db"))
CLAIM_CONCURRENCY = int(os.getenv("CLAIM_CONCURRENCY", "2"))

set_llm_cache(SQLiteCache(database_path=LANGCHAIN_CACHE_DB))
