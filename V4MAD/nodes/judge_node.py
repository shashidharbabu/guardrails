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
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI

from config import (ANTHROPIC_API_KEY, CLAUDE_JUDGE_MODEL, JUDGE_API_KEY,
                    JUDGE_BACKEND, JUDGE_BASE_URL, JUDGE_MAX_TOKENS,
                    JUDGE_MODEL, JUDGE_TEMP, LANGFUSE_HOST,
                    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
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
        return _parse_judge_text_fallback(raw)
    try:
        v = float(parsed["v_label"])
        if v not in (0.0, 0.5, 1.0):
            # Round to nearest valid value
            v = min([0.0, 0.5, 1.0], key=lambda x: abs(x - v))
        parsed["v_label"] = v
    except (KeyError, TypeError, ValueError):
        parsed["v_label"] = 0.5
    return parsed


def _parse_judge_text_fallback(raw: str) -> dict:
    text = raw.strip()
    lower = text.lower()

    if any(marker in lower for marker in ("not supported", "contradicted", "hallucinated")):
        v_label = 0.0
    elif any(marker in lower for marker in ("partially supported", "partial", "not fully accurate", "mixed")):
        v_label = 0.5
    elif any(marker in lower for marker in ("fully supported", "directly supports", "clearly supports")):
        v_label = 1.0
    else:
        v_label = 0.5

    chunk_ids = sorted(set(re.findall(r"\b(?:t1|t2|EMA|hitech)[A-Za-z0-9_:.\-]+(?:rechunk_\d+|[a-f0-9]{12})?\b", raw)))
    return {
        "v_label": v_label,
        "judge_confidence": 0.5,
        "judge_reasoning": text[:2400] or "Judge returned non-JSON output; fallback text parser used.",
        "evidence_chunk_ids": chunk_ids[:8],
    }


async def judge_node(state: MADState) -> dict:
    claims        = state.get("claims", [])
    claim_chunks  = state.get("claim_chunks", {})
    agent_outputs = state.get("agent_outputs", {})
    user_query    = state["user_query"]
    query_id      = state["query_id"]
    conn          = get_db_conn()

    client = None
    anthropic_client = None
    if JUDGE_BACKEND == "claude":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("JUDGE_BACKEND=claude requires ANTHROPIC_API_KEY")
        from anthropic import AsyncAnthropic
        anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    else:
        client = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY)
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

        judge_model_name = CLAUDE_JUDGE_MODEL if JUDGE_BACKEND == "claude" else JUDGE_MODEL
        cache_key = hashlib.md5(
            f"{judge_model_name}|{JUDGE_TEMP}|{json.dumps(messages)}".encode()
        ).hexdigest()

        cached = get_llm_cache(conn, cache_key)
        if cached:
            raw  = cached
            ti = to = latency_ms = 0
            from_cache = True
        else:
            t0   = time.time()
            if JUDGE_BACKEND == "claude":
                resp = await anthropic_client.messages.create(
                    model=CLAUDE_JUDGE_MODEL,
                    max_tokens=JUDGE_MAX_TOKENS,
                    temperature=JUDGE_TEMP,
                    system=JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = "".join(
                    block.text for block in resp.content
                    if getattr(block, "type", "") == "text"
                )
                ti = getattr(resp.usage, "input_tokens", 0)
                to = getattr(resp.usage, "output_tokens", 0)
                model_for_cache = CLAUDE_JUDGE_MODEL
            else:
                resp = await client.chat.completions.create(
                    model=JUDGE_MODEL, messages=messages,
                    temperature=JUDGE_TEMP, max_tokens=JUDGE_MAX_TOKENS,
                )
                raw = resp.choices[0].message.content
                ti = resp.usage.prompt_tokens
                to = resp.usage.completion_tokens
                model_for_cache = JUDGE_MODEL
            latency_ms = int((time.time() - t0) * 1000)
            from_cache = False
            insert_llm_cache(conn, cache_key, raw, model_for_cache)

        verdict = _parse_judge(raw)

        if _lf:
            try:
                _lf.generation(
                    name="judge",
                    session_id=query_id,
                    model=judge_model_name,
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

        insert_judge_verdict(conn, claim_id, verdict, judge_model_name,
                             raw, ti, to, latency_ms, from_cache)
        judge_verdicts[claim_id] = verdict

        label_str = {0.0: "NOT_SUPPORTED", 0.5: "PARTIAL", 1.0: "SUPPORTED"}.get(
            verdict["v_label"], "?"
        )
        print(f"    [judge] {claim_id[:8]}: {label_str} "
              f"(conf={verdict.get('judge_confidence', 0):.2f})")

    print(f"  [judge] {query_id}: {len(judge_verdicts)} verdicts rendered")
    return {"judge_verdicts": judge_verdicts}
