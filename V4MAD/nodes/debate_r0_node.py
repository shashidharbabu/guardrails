"""
LangGraph node: Debate Round 0.

Both agents receive the claim and their respective (different) chunk sets.
Agent A and B are called IN PARALLEL per claim (asyncio.gather).
CLAIM_CONCURRENCY claims processed simultaneously.

Key asymmetries enforced here:
  Agent A: system=AGENT_A_SYSTEM, temp=0.4, chunks [0,1,2]
  Agent B: system=AGENT_B_SYSTEM, temp=0.85, chunks [0,2,3]

Both point to the SAME vLLM server. With USE_GRPO_LORA_AGENTS=1, Agent A
calls model="agent_a" and Agent B calls model="agent_b"; otherwise both
call the shared base model.

Langfuse: every LLM call is traced as a generation with session_id=query_id.
LLM cache: prompt hash checked before each call — skips repeat work.
"""

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI

from config import (AGENT_A_TEMP, AGENT_B_TEMP, AGENT_MAX_TOKENS,
                    AGENT_A_MODEL, AGENT_B_MODEL, AGENTS_API_KEY,
                    AGENTS_BASE_URL, CLAIM_CONCURRENCY,
                    LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
from db import get_db_conn, get_llm_cache, insert_agent_output, insert_llm_cache
from prompts import AGENT_A_SYSTEM, AGENT_B_SYSTEM, round0_user
from schemas import MADState, parse_agent_json, validate_agent_output

try:
    from langfuse import Langfuse
    _lf = Langfuse(
        secret_key=LANGFUSE_SECRET_KEY,
        public_key=LANGFUSE_PUBLIC_KEY,
        host=LANGFUSE_HOST,
    )
except Exception:
    _lf = None

_client_a = AsyncOpenAI(base_url=AGENTS_BASE_URL, api_key=AGENTS_API_KEY)
_client_b = AsyncOpenAI(base_url=AGENTS_BASE_URL, api_key=AGENTS_API_KEY)


async def _call_agent(
    client, system: str, user: str, model: str, temp: float,
    agent_role: str, round_num: int, claim_id: str, query_id: str, conn
) -> tuple[dict, str, int, int, int, bool]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    cache_key = hashlib.md5(
        f"{model}|{temp}|{json.dumps(messages)}".encode()
    ).hexdigest()

    cached = get_llm_cache(conn, cache_key)
    if cached:
        raw  = cached
        ti = to = latency_ms = 0
        from_cache = True
    else:
        t0   = time.time()
        resp = await client.chat.completions.create(
            model=model, messages=messages,
            temperature=temp, max_tokens=AGENT_MAX_TOKENS,
        )
        latency_ms = int((time.time() - t0) * 1000)
        raw        = resp.choices[0].message.content
        ti         = resp.usage.prompt_tokens
        to         = resp.usage.completion_tokens
        from_cache = False
        insert_llm_cache(conn, cache_key, raw, model)

    parsed = parse_agent_json(raw) or {}
    ok, _ = validate_agent_output(parsed)
    if not ok:
        parsed = {"verdict": "IDK", "reasoning": "parse failed", "evidence_cited": [], "confidence_internal": 0.5}

    parsed["round_num"] = round_num

    if _lf:
        try:
            _lf.generation(
                name=f"{agent_role}-r{round_num}",
                session_id=query_id,
                model=model,
                model_parameters={"temperature": temp, "max_tokens": AGENT_MAX_TOKENS},
                input=messages,
                output=raw,
                metadata={
                    "claim_id": claim_id, "agent": agent_role, "round": round_num,
                    "from_cache": from_cache, "latency_ms": latency_ms,
                    "verdict": parsed.get("verdict"), "confidence": parsed.get("confidence_internal"),
                },
                usage={"input": ti, "output": to, "total": ti + to},
            )
        except Exception:
            pass

    insert_agent_output(conn, claim_id, agent_role, round_num,
                        parsed, raw, ti, to, latency_ms, from_cache)
    return parsed, raw, ti, to, latency_ms, from_cache


async def debate_r0_node(state: MADState) -> dict:
    claims       = state.get("claims", [])
    claim_chunks = state.get("claim_chunks", {})
    user_query   = state["user_query"]
    query_id     = state["query_id"]
    conn         = get_db_conn()

    semaphore    = asyncio.Semaphore(CLAIM_CONCURRENCY)
    agent_outputs = dict(state.get("agent_outputs", {}))

    async def process_claim(claim: dict):
        async with semaphore:
            claim_id   = claim["claim_id"]
            chunks     = claim_chunks.get(claim_id, {})
            a_chunks   = chunks.get("agent_a", [])
            b_chunks   = chunks.get("agent_b", [])

            a_user = round0_user(claim, a_chunks, user_query)
            b_user = round0_user(claim, b_chunks, user_query)

            (a_out, *_), (b_out, *_) = await asyncio.gather(
                _call_agent(_client_a, AGENT_A_SYSTEM, a_user,
                            AGENT_A_MODEL, AGENT_A_TEMP, "agent_a", 0, claim_id, query_id, conn),
                _call_agent(_client_b, AGENT_B_SYSTEM, b_user,
                            AGENT_B_MODEL, AGENT_B_TEMP, "agent_b", 0, claim_id, query_id, conn),
            )

            agent_outputs[claim_id] = {
                "agent_a": {0: a_out},
                "agent_b": {0: b_out},
            }
            print(f"    [r0] {claim_id[:8]} A={a_out['verdict']}({a_out['confidence_internal']:.2f}) "
                  f"B={b_out['verdict']}({b_out['confidence_internal']:.2f})")

    await asyncio.gather(*(process_claim(c) for c in claims))
    print(f"  [debate_r0] {query_id}: {len(claims)} claims done")
    return {"agent_outputs": agent_outputs}
