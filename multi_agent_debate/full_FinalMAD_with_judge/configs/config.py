import os

try:
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache
except ModuleNotFoundError:
    from langchain.cache import SQLiteCache
    from langchain.globals import set_llm_cache

BASELINE_MODEL_NAME = os.getenv("BASELINE_MODEL_NAME", "baseline")
DECOMPOSER_MODEL_NAME = os.getenv("DECOMPOSER_MODEL_NAME", "decomposer")
AGENTS_MODEL_NAME = os.getenv("AGENTS_MODEL_NAME", "agents")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "Qwen2.5-32B-Instruct")

VLLM_BASELINE_URL = os.getenv("VLLM_BASELINE_URL", "http://localhost:8003/v1")
VLLM_DECOMPOSER_URL = os.getenv("VLLM_DECOMPOSER_URL", "http://localhost:8002/v1")
VLLM_AGENTS_URL = os.getenv("VLLM_AGENTS_URL", "http://localhost:8001/v1")
VLLM_JUDGE_URL = os.getenv("VLLM_JUDGE_URL", "http://localhost:8004/v1")

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "mad.db")
LANGCHAIN_CACHE_DB = os.getenv("LANGCHAIN_CACHE_DB", ".langchain_cache.db")
CLAIM_CONCURRENCY = int(os.getenv("CLAIM_CONCURRENCY", "2"))

set_llm_cache(SQLiteCache(database_path=LANGCHAIN_CACHE_DB))
