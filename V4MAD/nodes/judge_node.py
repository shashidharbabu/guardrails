"""
LangGraph node: Judge.

Blind 14B/32B judge sees:
  - Both agents' R0 and R1 outputs (stripped — no confidence, anonymized as Debater 1/2)
  - ALL 5 chunks (complete evidence pool)

Judge renders a final verdict: v_label ∈ {0.0, 0.5, 1.0}
  0.0 = NOT_SUPPORTED (hallucination)
  0.5 = PARTIAL
  1.0 = SUPPORTED

Sequential per claim (judge is expensive — no parallel claims in judge step).
Every call traced to Langfuse.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI

from config import (JUDGE_MAX_TOKENS, JUDGE_MODEL, JUDGE_PORT, JUDGE_TEMP,
                    LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
from db import get_db_conn, get_llm_cache, insert_judge_verdict, insert_llm_cache
from prompts import JUDGE_SYSTEM, judge_user
from schemas import MADState, parse_agent_json, strip_for_peer

try:
    from langfuse import Langfuse
    _lf = Langfuse(
        secret_key=LANGFUSE_SECRET_KEY,
        public_key=LANGFUSE_PUBLIC_KEY,
        host=LANGFUSE_HOST,
    )
except Exception:
    _lf = None


def _parse_judge(raw: str) -> dict:
    parsed = parse_agent_json(raw)
    if not parsed:
        return {"v_label": 0.5, "judge_confidence": 0.0,
                "judge_reasoning": "parse failed", "evidence_chunk_ids": []}
    try:
        v = float(parsed["v_label"])
        if v not in (0.0, 0.5, 1.0):
            # Round to nearest valid value
            v = min([0.0, 0.5, 1.0], key=lambda x: abs(x - v))
        parsed["v_label"] = v
    except (KeyError, TypeError, ValueError):
        parsed["v_label"] = 0.5
    return parsed


async def judge_node(state: MADState) -> dict:
    claims        = state.get("claims", [])
    claim_chunks  = state.get("claim_chunks", {})
    agent_outputs = state.get("agent_outputs", {})
    user_query    = state["user_query"]
    query_id      = state["query_id"]
    conn          = get_db_conn()

    client         = AsyncOpenAI(base_url=f"http://localhost:{JUDGE_PORT}/v1", api_key="EMPTY")
    judge_verdicts = dict(state.get("judge_verdicts", {}))

    for claim in claims:
        claim_id   = claim["claim_id"]
        r0_outputs = agent_outputs.get(claim_id, {})

        a_r0 = r0_outputs.get("agent_a", {}).get(0)
        a_r1 = r0_outputs.get("agent_a", {}).get(1)
        b_r0 = r0_outputs.get("agent_b", {}).get(0)
        b_r1 = r0_outputs.get("agent_b", {}).get(1)

        if not all([a_r0, a_r1, b_r0, b_r1]):
            print(f"    [judge] {claim_id[:8]}: missing debate outputs — skipping")
            continue

        # Strip identity + confidence from all outputs before showing to judge
        d1_r0 = strip_for_peer(a_r0, "Debater 1")
        d1_r1 = strip_for_peer(a_r1, "Debater 1")
        d2_r0 = strip_for_peer(b_r0, "Debater 2")
        d2_r1 = strip_for_peer(b_r1, "Debater 2")

        judge_chunks = claim_chunks.get(claim_id, {}).get("judge", [])
        prompt       = judge_user(claim, judge_chunks, user_query,
                                  d1_r0, d1_r1, d2_r0, d2_r1)

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": prompt},
        ]

        cache_key = hashlib.md5(
            f"{JUDGE_MODEL}|{JUDGE_TEMP}|{json.dumps(messages)}".encode()
        ).hexdigest()

        cached = get_llm_cache(conn, cache_key)
        if cached:
            raw  = cached
            ti = to = latency_ms = 0
            from_cache = True
        else:
            t0   = time.time()
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL, messages=messages,
                temperature=JUDGE_TEMP, max_tokens=JUDGE_MAX_TOKENS,
            )
            latency_ms = int((time.time() - t0) * 1000)
            raw        = resp.choices[0].message.content
            ti         = resp.usage.prompt_tokens
            to         = resp.usage.completion_tokens
            from_cache = False
            insert_llm_cache(conn, cache_key, raw, JUDGE_MODEL)

        verdict = _parse_judge(raw)

        if _lf:
            try:
                _lf.generation(
                    name="judge",
                    session_id=query_id,
                    model=JUDGE_MODEL,
                    model_parameters={"temperature": JUDGE_TEMP, "max_tokens": JUDGE_MAX_TOKENS},
                    input=messages,
                    output=raw,
                    metadata={
                        "claim_id": claim_id, "from_cache": from_cache,
                        "latency_ms": latency_ms, "v_label": verdict["v_label"],
                        "judge_confidence": verdict.get("judge_confidence"),
                        "a_r1_verdict": a_r1.get("verdict"),
                        "b_r1_verdict": b_r1.get("verdict"),
                    },
                    usage={"input": ti, "output": to, "total": ti + to},
                )
            except Exception:
                pass

        insert_judge_verdict(conn, claim_id, verdict, JUDGE_MODEL,
                             raw, ti, to, latency_ms, from_cache)
        judge_verdicts[claim_id] = verdict

        label_str = {0.0: "NOT_SUPPORTED", 0.5: "PARTIAL", 1.0: "SUPPORTED"}.get(
            verdict["v_label"], "?"
        )
        print(f"    [judge] {claim_id[:8]}: {label_str} "
              f"(conf={verdict.get('judge_confidence', 0):.2f})")

    print(f"  [judge] {query_id}: {len(judge_verdicts)} verdicts rendered")
    return {"judge_verdicts": judge_verdicts}
