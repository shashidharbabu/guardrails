"""
LangGraph node: Baseline LLM generation (NO RAG).

The baseline model (3B) answers from parametric memory.
Smaller model = more hallucinations = better test cases for the debate agents.
Results are cached in SQLite so re-runs skip already-processed queries.
Every generation is traced to Langfuse under session_id=query_id.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI

from config import (BASELINE_API_KEY, BASELINE_BASE_URL, BASELINE_MAX_TOKENS,
                    BASELINE_MODEL, BASELINE_TEMP,
                    LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
from db import get_db_conn, get_llm_cache, insert_llm_cache, insert_query
from prompts import BASELINE_SYSTEM, baseline_user
from schemas import MADState

try:
    from langfuse import Langfuse
    _lf = Langfuse(
        secret_key=LANGFUSE_SECRET_KEY,
        public_key=LANGFUSE_PUBLIC_KEY,
        host=LANGFUSE_HOST,
    )
except Exception:
    _lf = None


async def baseline_node(state: MADState) -> dict:
    if state.get("baseline_answer"):
        return {}

    query_id   = state["query_id"]
    user_query = state["user_query"]
    conn       = get_db_conn()

    messages = [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user",   "content": baseline_user(user_query)},
    ]

    cache_key = hashlib.md5(
        f"{BASELINE_MODEL}|{BASELINE_TEMP}|{json.dumps(messages)}".encode()
    ).hexdigest()

    cached = get_llm_cache(conn, cache_key)
    if cached:
        content    = cached
        ti = to = latency_ms = 0
        from_cache = True
    else:
        client = AsyncOpenAI(base_url=BASELINE_BASE_URL, api_key=BASELINE_API_KEY)
        t0   = time.time()
        resp = await client.chat.completions.create(
            model=BASELINE_MODEL,
            messages=messages,
            temperature=BASELINE_TEMP,
            max_tokens=BASELINE_MAX_TOKENS,
        )
        latency_ms = int((time.time() - t0) * 1000)
        content    = resp.choices[0].message.content
        ti         = resp.usage.prompt_tokens
        to         = resp.usage.completion_tokens
        from_cache = False
        insert_llm_cache(conn, cache_key, content, BASELINE_MODEL)

    if _lf:
        try:
            _lf.generation(
                name="baseline-llm",
                session_id=query_id,
                model=BASELINE_MODEL,
                model_parameters={"temperature": BASELINE_TEMP, "max_tokens": BASELINE_MAX_TOKENS},
                input=messages,
                output=content,
                metadata={"from_cache": from_cache, "latency_ms": latency_ms},
                usage={"input": ti, "output": to, "total": ti + to},
            )
        except Exception:
            pass

    insert_query(conn, query_id, state["run_id"], user_query,
                 content, BASELINE_MODEL, ti, to, latency_ms)

    print(f"  [baseline] {query_id}: {content[:80].replace(chr(10),' ')}...")
    return {"baseline_answer": content}
