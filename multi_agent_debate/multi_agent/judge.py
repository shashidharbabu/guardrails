"""
judge.py — Judge Agent (Partially Blind)

SEES:
  - User query
  - Agent A's final revised verdicts (verdict text + reasoning)
  - Evidence pool (all chunks from both agents — unlabelled, no agent attribution)

DOES NOT SEE:
  - Confidence scores (intentionally stripped — avoids anchoring)
  - Agent B's challenge text or argumentative framing
  - Which agent retrieved which chunk

Scores each material claim independently: v = 1.0 / 0.5 / 0.0
Generates a correction_signal for the retry loop when claims are wrong.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from openai import OpenAI

from . import mad_tracing as lf
from .config import (
    OLLAMA_BASE_URL, OLLAMA_API_KEY,
    JUDGE_MODEL, JUDGE_PROVIDER, ANTHROPIC_API_KEY,
)
from .models import Claim, EvidenceChunk, JudgeVerdict

# ── Client setup ───────────────────────────────────────────────────────────────
_ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

_anthropic_client = None
if JUDGE_PROVIDER == "anthropic":
    try:
        import anthropic as _anthropic_sdk
        _anthropic_client = _anthropic_sdk.Anthropic(api_key=ANTHROPIC_API_KEY)
        print(f"[Judge] Using Claude via Anthropic API — model: {JUDGE_MODEL}")
    except ImportError:
        print("[Judge] Warning: anthropic SDK not installed. Falling back to Ollama.")

_SYSTEM = (
    "You are an impartial regulatory compliance judge. "
    "Evaluate claims strictly based on the provided evidence. "
    "Be precise and cite specific evidence chunks. "
    "Do not speculate beyond what the evidence supports."
)

_JUDGE_PROMPT = """\
You are an impartial Judge evaluating whether an AI-generated regulatory compliance answer is accurate.

Your task: independently score each claim against the provided evidence pool.

SCORING:
  1.0 — claim is fully accurate and directly supported by the evidence
  0.5 — claim is partially correct but missing important nuance, caveats, or scope qualifications
  0.0 — claim is inaccurate, unsupported, or appears to be hallucinated

INSTRUCTIONS:
- Score ALL claims. For non-material claims (is_material=false): assign score based on factual accuracy.
- Cite specific chunk_ids in your reasoning.
- After scoring, produce a correction_signal:
    - A clear, actionable explanation of what the original answer got wrong and what the correct regulatory position is.
    - This signal will be sent to the LLM on retry so it must be self-contained and precise.
    - If ALL material claims score 1.0: set correction_signal to null.

User query:
{query}

Claims to evaluate (from verification agent — confidence scores REDACTED):
{claims_json}

Evidence pool (regulatory chunks from all sources — evaluate against these):
{evidence_pool_json}

Return ONLY valid JSON. No markdown fences, no explanation outside the JSON.
Output schema (replace all placeholder values with your actual evaluation):

{{
  "verdicts": [
    {{
      "claim_id": <integer matching a claim_id above>,
      "score": <1.0 | 0.5 | 0.0>,
      "reasoning": "<cite the specific chunk_id(s) from the evidence pool that support your score and explain why>"
    }}
  ],
  "correction_signal": "<concise actionable description of what the original answer got wrong and what the correct regulatory position is, grounded in the evidence pool — or null if all material claims score 1.0>"
}}
"""


def judge_claims(
    query:         str,
    final_claims:  List[Claim],
    evidence_pool: List[EvidenceChunk],
) -> Tuple[List[JudgeVerdict], Optional[str]]:
    """
    Judge evaluates all claims against the evidence pool.

    Intentionally STRIPS confidence scores from the claims JSON
    so the Judge scores independently without anchoring.

    Returns: (judge_verdicts, correction_signal)
    """
    # Build claims JSON with confidence STRIPPED
    claims_json = json.dumps(
        [{
            "claim_id":      c.claim_id,
            "claim_text":    c.claim_text,
            "is_material":   c.is_material,
            "agent_verdict": c.verdict.value if c.verdict else "IDK",
            "agent_reasoning": c.reasoning,
            # confidence intentionally omitted
        } for c in final_claims],
        indent=2,
    )

    evidence_json = json.dumps(
        [{"chunk_id": c.chunk_id, "source": c.source, "tier": c.tier, "text": c.text}
         for c in evidence_pool],
        indent=2,
    )

    prompt = _JUDGE_PROMPT.format(
        query=query.strip(),
        claims_json=claims_json,
        evidence_pool_json=evidence_json,
    )

    with lf.observe(
        "generation",
        "mad.judge.llm",
        model=JUDGE_MODEL,
        input_payload={
            "provider": JUDGE_PROVIDER,
            "claim_count": len(final_claims),
            "evidence_chunk_count": len(evidence_pool),
            "prompt_chars": len(prompt),
        },
    ):
        raw = _call_judge(prompt)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"[judge] Failed to parse Judge response.\nError: {e}\nRaw:\n{raw[:500]}"
        )

    correction_signal: Optional[str] = data.get("correction_signal") or None

    verdict_map = {v["claim_id"]: v for v in data.get("verdicts", [])}
    judge_verdicts: List[JudgeVerdict] = []

    for claim in final_claims:
        v = verdict_map.get(claim.claim_id, {})
        score = float(v.get("score", 1.0 if not claim.is_material else 0.5))
        # Clamp to valid values
        if score > 0.75:
            score = 1.0
        elif score > 0.25:
            score = 0.5
        else:
            score = 0.0

        judge_verdicts.append(JudgeVerdict(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            is_material=claim.is_material,
            score=score,
            reasoning=v.get("reasoning", ""),
        ))

    return judge_verdicts, correction_signal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_judge(prompt: str) -> str:
    """
    Call the judge LLM and return raw text response.
    Routes to Anthropic Claude if JUDGE_PROVIDER=anthropic, else Ollama.
    """
    if JUDGE_PROVIDER == "anthropic" and _anthropic_client is not None:
        message = _anthropic_client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=1024,
            temperature=0.0,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return _clean_json_object(message.content[0].text)

    # Fallback: Ollama
    response = _ollama_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
    )
    return _clean_json_object(response.choices[0].message.content)


def _clean_json_object(raw: str) -> str:
    """Strip markdown fences and isolate the JSON object."""
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    raw = raw.strip()
    # Prefer outermost { ... }
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return match.group() if match else raw
