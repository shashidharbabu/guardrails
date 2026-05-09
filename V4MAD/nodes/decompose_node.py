"""
LangGraph node: Decomposer — extract atomic claims from baseline answer.

Uses 7B model (deterministic, temp=0.0) for reliable JSON extraction.
Includes coverage check: do extracted claims cover ≥60% of baseline answer words?
"""

import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI

from config import (DECOMPOSER_API_KEY, DECOMPOSER_BASE_URL,
                    DECOMPOSER_MAX_TOKENS, DECOMPOSER_MODEL, DECOMPOSER_TEMP,
                    LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
from db import get_db_conn, get_llm_cache, insert_claim, insert_llm_cache
from prompts import DECOMPOSER_SYSTEM, decomposer_user
from schemas import MADState, parse_agent_json

try:
    from langfuse import Langfuse
    _lf = Langfuse(
        secret_key=LANGFUSE_SECRET_KEY,
        public_key=LANGFUSE_PUBLIC_KEY,
        host=LANGFUSE_HOST,
    )
except Exception:
    _lf = None



def _extract_json_object(raw: str) -> dict | None:
    """Best-effort JSON extraction/repair for decomposer output."""
    if not raw:
        return None

    text = raw.strip()

    # Remove markdown fences
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    # Keep only outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    text = text[start:end + 1]

    # Common model mistake: { { "claim_text": ... } }
    text = re.sub(r"\{\s*\{", "{", text)
    text = re.sub(r"\}\s*\}", "}", text)

    # Remove trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    # Last resort: extract claim_text strings manually
    matches = re.findall(r'"claim_text"\s*:\s*"([^"]+)"', text)
    if matches:
        return {
            "claims": [
                {
                    "claim_text": m,
                    "claim_index": i,
                    "is_material": True,
                    "is_critical": False,
                    "confidence_prior": 0.75,
                }
                for i, m in enumerate(matches)
            ]
        }

    return None


async def _call_decomposer(messages):
    client = AsyncOpenAI(
        base_url=DECOMPOSER_BASE_URL,
        api_key=DECOMPOSER_API_KEY,
    )
    resp = await client.chat.completions.create(
        model=DECOMPOSER_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=DECOMPOSER_MAX_TOKENS,
    )
    return resp


def _coverage_check(baseline: str, claims: list[dict], threshold: float = 0.6) -> tuple[bool, float]:
    baseline_words = set(baseline.lower().split())
    claim_words    = set(" ".join(c["claim_text"] for c in claims).lower().split())
    if not baseline_words:
        return True, 1.0
    ratio = len(baseline_words & claim_words) / len(baseline_words)
    return ratio >= threshold, ratio


async def decompose_node(state: MADState) -> dict:
    if state.get("claims"):
        return {}

    query_id        = state["query_id"]
    baseline_answer = state.get("baseline_answer", "")
    user_query      = state["user_query"]
    conn            = get_db_conn()

    messages = [
        {"role": "system", "content": DECOMPOSER_SYSTEM},
        {"role": "user",   "content": decomposer_user(user_query, baseline_answer)},
    ]

    cache_key = hashlib.md5(
        f"{DECOMPOSER_MODEL}|{DECOMPOSER_TEMP}|{json.dumps(messages)}".encode()
    ).hexdigest()

    cached = get_llm_cache(conn, cache_key)
    if cached:
        raw        = cached
        ti = to = latency_ms = 0
        from_cache = True
    else:
        t0   = time.time()
        resp = await _call_decomposer(messages)
        latency_ms = int((time.time() - t0) * 1000)
        raw        = resp.choices[0].message.content
        ti         = resp.usage.prompt_tokens
        to         = resp.usage.completion_tokens
        from_cache = False
        insert_llm_cache(conn, cache_key, raw, DECOMPOSER_MODEL)

    parsed = parse_agent_json(raw) or _extract_json_object(raw)

    if not parsed or "claims" not in parsed:
        repair_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous output was invalid JSON. Rewrite it as ONLY valid JSON. "
                    "Use exactly this schema: "
                    "{\"claims\":[{\"claim_text\":\"...\","
                    "\"claim_index\":0,\"is_material\":true,"
                    "\"is_critical\":false,\"confidence_prior\":0.75}]}. "
                    "No markdown. No explanation. No nested extra braces."
                ),
            }
        ]
        try:
            resp2 = await _call_decomposer(repair_messages)
            raw = resp2.choices[0].message.content
            parsed = parse_agent_json(raw) or _extract_json_object(raw)
        except Exception:
            parsed = None

    if not parsed or "claims" not in parsed:
        print(f"  [decompose] {query_id}: parse failed — {raw[:120]}")
        return {"errors": state.get("errors", []) + [f"{query_id}: decompose parse failed"]}

    raw_claims = parsed["claims"]
    if not isinstance(raw_claims, list):
        raw_claims = []

    claims = []
    for i, c in enumerate(raw_claims):
        if not isinstance(c, dict):
            continue
        if not str(c.get("claim_text", "")).strip():
            continue
        claim_id = str(uuid.uuid4())
        claim    = {
            "claim_id":        claim_id,
            "claim_text":      c.get("claim_text", ""),
            "claim_index":     c.get("claim_index", i),
            "is_material":     bool(c.get("is_material", True)),
            "is_critical":     bool(c.get("is_critical", False)),
            "confidence_prior": float(c.get("confidence_prior", 0.75)),
        }
        claims.append(claim)

    ok, ratio = _coverage_check(baseline_answer, claims)
    for claim in claims:
        insert_claim(conn, claim, query_id, coverage_check=ok, coverage_ratio=ratio)

    # Reconstructed text: claims joined in order — used offline to evaluate
    # decomposer quality via cosine similarity vs baseline_answer (see utils/eval_decomposer.py)
    reconstructed_text = " ".join(c["claim_text"] for c in claims)

    if _lf:
        try:
            _lf.generation(
                name="decomposer",
                session_id=query_id,
                model=DECOMPOSER_MODEL,
                model_parameters={"temperature": DECOMPOSER_TEMP},
                input=messages,
                output=raw,
                metadata={
                    "from_cache":         from_cache,
                    "latency_ms":         latency_ms,
                    "num_claims":         len(claims),
                    "coverage_ratio":     round(ratio, 4),
                    "coverage_ok":        ok,
                    "reconstructed_text": reconstructed_text,  # for offline similarity eval
                    "baseline_text":      baseline_answer,
                },
                usage={"input": ti, "output": to, "total": ti + to},
            )
        except Exception:
            pass

    print(f"  [decompose] {query_id}: {len(claims)} claims extracted "
          f"(coverage={ratio:.0%}{'✓' if ok else ' ⚠ below threshold'})")
    return {"claims": claims}
