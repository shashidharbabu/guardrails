"""
LangGraph node: Debate Round 1.

Each agent sees the OTHER agent's Round 0 output (stripped — no confidence_internal).
Agent A sees B's R0 stripped output. Agent B sees A's R0 stripped output.

IMPORTANT: Agent A does NOT see Agent B's full chunks. It only sees the
relevant_quote (≤160 chars) that Agent B chose to cite. If Agent B cited
a chunk that's in position [3], Agent A cannot look it up in its
own evidence pool — creating genuine information asymmetry in the debate.

R1 prompt redesign: agents should HOLD their position unless the other agent
cited specific evidence they missed, not just because the other argument sounds
plausible. Both agents must start reasoning with "HOLDING:" or "UPDATING:".
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
                    AGENT_A_MODEL, AGENT_B_MODEL, AGENTS_PORT, CLAIM_CONCURRENCY,
                    LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
from db import get_db_conn, get_llm_cache, insert_agent_output, insert_llm_cache
from prompts import AGENT_A_SYSTEM, AGENT_B_SYSTEM, round1_user
from schemas import MADState, parse_agent_json, strip_for_peer, validate_agent_output

try:
    from langfuse import Langfuse
    _lf = Langfuse(
        secret_key=LANGFUSE_SECRET_KEY,
        public_key=LANGFUSE_PUBLIC_KEY,
        host=LANGFUSE_HOST,
    )
except Exception:
    _lf = None

_client_a = AsyncOpenAI(base_url=f"http://localhost:{AGENTS_PORT}/v1", api_key="EMPTY")
_client_b = AsyncOpenAI(base_url=f"http://localhost:{AGENTS_PORT}/v1", api_key="EMPTY")


async def _call_agent_r1(
    client, system: str, user: str, model: str, temp: float,
    agent_role: str, claim_id: str, query_id: str, conn
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

    parsed["round_num"] = 1

    if _lf:
        try:
            # Detect if agent held or updated its R0 verdict
            holding = parsed.get("reasoning", "").upper().startswith("HOLDING")
            _lf.generation(
                name=f"{agent_role}-r1",
                session_id=query_id,
                model=model,
                model_parameters={"temperature": temp, "max_tokens": AGENT_MAX_TOKENS},
                input=messages,
                output=raw,
                metadata={
                    "claim_id": claim_id, "agent": agent_role, "round": 1,
                    "from_cache": from_cache, "latency_ms": latency_ms,
                    "verdict": parsed.get("verdict"),
                    "confidence": parsed.get("confidence_internal"),
                    "holding_position": holding,
                },
                usage={"input": ti, "output": to, "total": ti + to},
            )
        except Exception:
            pass

    insert_agent_output(conn, claim_id, agent_role, 1,
                        parsed, raw, ti, to, latency_ms, from_cache)
    return parsed, raw, ti, to, latency_ms, from_cache


async def debate_r1_node(state: MADState) -> dict:
    claims        = state.get("claims", [])
    claim_chunks  = state.get("claim_chunks", {})
    agent_outputs = dict(state.get("agent_outputs", {}))
    user_query    = state["user_query"]
    query_id      = state["query_id"]
    conn          = get_db_conn()

    semaphore = asyncio.Semaphore(CLAIM_CONCURRENCY)

    async def process_claim(claim: dict):
        async with semaphore:
            claim_id = claim["claim_id"]

            r0_outputs = agent_outputs.get(claim_id, {})
            a_r0 = r0_outputs.get("agent_a", {}).get(0)
            b_r0 = r0_outputs.get("agent_b", {}).get(0)
            if not a_r0 or not b_r0:
                return  # R0 missing — skip R1 for this claim

            # Strip for peer: hide confidence_internal, truncate reasoning
            # Agent A sees B's stripped R0 (with B's chunk quotes — possibly from [2],[3])
            # Agent B sees A's stripped R0 (with A's chunk quotes from [0],[1],[2])
            b_stripped = strip_for_peer(b_r0, "Debater 2")
            a_stripped = strip_for_peer(a_r0, "Debater 1")

            chunks   = claim_chunks.get(claim_id, {})
            a_chunks = chunks.get("agent_a", [])
            b_chunks = chunks.get("agent_b", [])

            # Agent A's R1: sees its own [0,1,2] chunks + B's stripped R0
            a_r1_user = round1_user(claim, a_chunks, user_query, b_stripped)
            # Agent B's R1: sees its own [0,2,3] chunks + A's stripped R0
            b_r1_user = round1_user(claim, b_chunks, user_query, a_stripped)

            (a_r1, *_), (b_r1, *_) = await asyncio.gather(
                _call_agent_r1(_client_a, AGENT_A_SYSTEM, a_r1_user,
                               AGENT_A_MODEL, AGENT_A_TEMP, "agent_a", claim_id, query_id, conn),
                _call_agent_r1(_client_b, AGENT_B_SYSTEM, b_r1_user,
                               AGENT_B_MODEL, AGENT_B_TEMP, "agent_b", claim_id, query_id, conn),
            )

            a_r1["round_num"] = 1
            b_r1["round_num"] = 1

            agent_outputs[claim_id]["agent_a"][1] = a_r1
            agent_outputs[claim_id]["agent_b"][1] = b_r1

            # Log verdict delta
            a_changed = a_r1.get("verdict") != a_r0.get("verdict")
            b_changed = b_r1.get("verdict") != b_r0.get("verdict")
            print(f"    [r1] {claim_id[:8]} "
                  f"A:{a_r0['verdict']}→{a_r1['verdict']}{'↑' if a_changed else '='} "
                  f"B:{b_r0['verdict']}→{b_r1['verdict']}{'↑' if b_changed else '='}")

    await asyncio.gather(*(process_claim(c) for c in claims))
    print(f"  [debate_r1] {query_id}: {len(claims)} claims done")
    return {"agent_outputs": agent_outputs}
