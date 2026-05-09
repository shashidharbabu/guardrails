"""
claim_extractor.py — extracts atomic verifiable claims from the LLM's answer.

Each claim gets:
  claim_id       — sequential integer
  claim_text     — exact atomic statement (one fact, one regulation, one number)
  is_material    — True if the claim involves regulatory obligations, penalties, thresholds
  confidence     — prior belief before debate (0.5–0.9, never 0 or 1)
"""
from __future__ import annotations

import json
import re
from typing import List

from openai import OpenAI

from mad.config import OLLAMA_BASE_URL, OLLAMA_API_KEY, AGENT_MODEL
from mad.models import Claim

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

_SYSTEM = (
    "You are a precise regulatory compliance analyst. "
    "You extract atomic, verifiable claims from AI-generated answers for fact-checking."
)

_PROMPT = """\
You are given a user query and an AI-generated answer about regulatory compliance.

Your task: extract every distinct, atomic, verifiable claim from the answer.

RULES:
- Each claim must express exactly ONE verifiable fact (one regulation, one number, one obligation).
- Do NOT merge two facts into one claim. If the answer says "X is required and Y is prohibited", those are TWO claims.
- Do NOT paraphrase — extract the claim as close to the original wording as possible.
- is_material = true  → claim involves: a specific regulatory article, legal obligation, penalty, threshold, named standard, or compliance requirement.
- is_material = false → claim is background context, a general statement, or a definition with no direct compliance implication.
- confidence = your prior belief this claim is accurate (float 0.50–0.90). Never set 0 or 1 — these are priors, not verdicts.

Return ONLY a valid JSON array. No markdown fences, no explanation, no preamble.

[
  {{
    "claim_id": 1,
    "claim_text": "exact atomic claim text from the answer",
    "is_material": true,
    "confidence": 0.75
  }}
]

User query:
{query}

AI answer to analyze:
{llm_answer}
"""


def extract_claims(query: str, llm_answer: str) -> List[Claim]:
    """
    Call the LLM to extract atomic claims from llm_answer.
    Returns a list of Claim objects ready for the debate pipeline.
    """
    prompt = _PROMPT.format(query=query.strip(), llm_answer=llm_answer.strip())

    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    raw = _clean_json_array(raw)

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"[claim_extractor] Failed to parse LLM response as JSON.\n"
            f"Error: {e}\nRaw response:\n{raw[:500]}"
        )

    claims: List[Claim] = []
    for item in items:
        claims.append(
            Claim(
                claim_id=int(item["claim_id"]),
                claim_text=str(item["claim_text"]).strip(),
                is_material=bool(item.get("is_material", True)),
                confidence=float(item.get("confidence", 0.70)),
                verdict=None,
                evidence_chunks=[],
                reasoning="",
            )
        )

    return claims


def _clean_json_array(raw: str) -> str:
    """Strip markdown fences and isolate the JSON array."""
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    raw = raw.strip()
    # Find outermost [ ... ]
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        return match.group()
    return raw
