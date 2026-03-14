"""
agent_a.py — Agent A: Ground Truth Verifier

Responsibilities:
  verify_claims()   — Initial RAG-based verification of all extracted claims.
                       Assigns SUPPORTED / PARTIAL / NOT_SUPPORTED / IDK + confidence.
  revise_verdicts() — Updates verdicts/confidence after Agent B's challenges.
                       CANNOT change the original LLM claim text.
                       Can only update: verdict, confidence, reasoning, evidence_chunks.
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple

from openai import OpenAI

from multi_agent.config import OLLAMA_BASE_URL, OLLAMA_API_KEY, AGENT_MODEL, TOP_K_CHUNKS
from multi_agent.models import Claim, Challenge, EvidenceChunk, Verdict
from multi_agent import rag_stub

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are Agent A — a regulatory compliance expert and Ground Truth Verifier. "
    "Your role is to verify claims against retrieved regulatory evidence. "
    "Be precise, calibrated, and cite specific evidence chunks."
)

# ── Prompts ───────────────────────────────────────────────────────────────────
_VERIFY_PROMPT = """\
You are verifying claims made by an AI about regulatory compliance.

For each claim, evaluate it against the retrieved evidence chunks and assign:

VERDICT options:
  SUPPORTED     — evidence directly and fully supports the claim as stated
  PARTIAL       — evidence partially supports it but with important caveats, missing nuance, or scope limitations
  NOT_SUPPORTED — evidence contradicts the claim OR the claim is not found in any retrieved evidence
  IDK           — no relevant evidence retrieved; cannot make a determination

CONFIDENCE (float 0.0–1.0):
  Your calibrated belief that your verdict assignment is correct.
  0.9+ = very strong evidence. 0.7 = moderate. 0.5 = uncertain. 0.3 = weak.

Retrieved evidence chunks:
{evidence_json}

Claims to verify:
{claims_json}

Return ONLY a valid JSON array. No markdown, no explanation.

[
  {{
    "claim_id": 1,
    "verdict": "SUPPORTED",
    "confidence": 0.88,
    "reasoning": "GDPR Art 32 chunk directly states encryption as one appropriate measure...",
    "evidence_chunk_ids": ["gdpr_art32_p1"]
  }}
]
"""

_REVISE_PROMPT = """\
You are Agent A — Ground Truth Verifier. You have completed initial verification.

Agent B has raised challenges against some of your verdicts. Review each challenge carefully.

REVISION RULES:
- If Agent B provides new valid regulatory evidence that genuinely changes the picture: UPDATE your verdict and confidence.
- If Agent B is restating the same point without new evidence: MAINTAIN your position.
- If Agent B is attacking a well-supported claim without evidence (gaslighting): MAINTAIN your position and lower confidence only slightly.
- You CANNOT change the claim text — only verdict, confidence, reasoning, and evidence_chunk_ids.
- Include ALL claims in your response (even unchanged ones).

Your current verdicts:
{current_verdicts_json}

Agent B's challenges:
{challenges_json}

Additional evidence retrieved by Agent B:
{agent_b_evidence_json}

Return ONLY a valid JSON array with your REVISED verdicts for ALL claims. No markdown.

[
  {{
    "claim_id": 1,
    "verdict": "PARTIAL",
    "confidence": 0.52,
    "reasoning": "Reconsidering after B's challenge — EDPB 2024 chunk confirms Art 32 does not mandate AES-256...",
    "evidence_chunk_ids": ["gdpr_art32_p1", "edpb_guidelines_2024_encryption"]
  }}
]
"""


# ── Public functions ──────────────────────────────────────────────────────────

def verify_claims(
    query: str,
    claims: List[Claim],
) -> Tuple[List[Claim], List[EvidenceChunk]]:
    """
    Step A of each cycle: Initial verification via RAG.
    Returns updated claims + all evidence chunks retrieved.
    """
    # Retrieve evidence — one RAG call per claim + one for the overall query
    evidence_map: dict[str, EvidenceChunk] = {}

    for claim in claims:
        for chunk in rag_stub.retrieve(claim.claim_text, top_k=TOP_K_CHUNKS):
            evidence_map[chunk.chunk_id] = chunk

    for chunk in rag_stub.retrieve(query, top_k=3):
        evidence_map[chunk.chunk_id] = chunk

    all_evidence = list(evidence_map.values())

    # Build prompt payloads
    evidence_json = json.dumps(
        [{"chunk_id": c.chunk_id, "source": c.source, "tier": c.tier, "text": c.text}
         for c in all_evidence],
        indent=2,
    )
    claims_json = json.dumps(
        [{"claim_id": c.claim_id, "claim_text": c.claim_text, "is_material": c.is_material}
         for c in claims],
        indent=2,
    )

    prompt = _VERIFY_PROMPT.format(
        evidence_json=evidence_json,
        claims_json=claims_json,
    )

    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
    )

    raw = _clean_json_array(response.choices[0].message.content)
    verdict_items = json.loads(raw)
    verdict_map = {v["claim_id"]: v for v in verdict_items}

    updated: List[Claim] = []
    for claim in claims:
        v = verdict_map.get(claim.claim_id, {})
        updated.append(claim.model_copy(update={
            "verdict":        Verdict(v.get("verdict", "IDK")),
            "confidence":     float(v.get("confidence", 0.5)),
            "reasoning":      v.get("reasoning", ""),
            "evidence_chunks": v.get("evidence_chunk_ids", []),
        }))

    return updated, all_evidence


def revise_verdicts(
    claims:          List[Claim],
    challenges:      List[Challenge],
    agent_b_evidence: List[EvidenceChunk],
) -> List[Claim]:
    """
    Step C of each cycle: Revise verdicts after Agent B's challenges.
    Cannot change claim text — only verdict, confidence, reasoning, evidence.
    """
    current_json = json.dumps(
        [{
            "claim_id":          c.claim_id,
            "claim_text":        c.claim_text,
            "verdict":           c.verdict.value if c.verdict else "IDK",
            "confidence":        c.confidence,
            "reasoning":         c.reasoning,
            "evidence_chunk_ids": c.evidence_chunks,
            "is_material":       c.is_material,
        } for c in claims],
        indent=2,
    )

    challenges_json = json.dumps(
        [{
            "claim_id":       ch.claim_id,
            "challenge_type": ch.challenge_type.value,
            "challenge_text": ch.challenge_text,
            "suggested_verdict": ch.suggested_verdict.value if ch.suggested_verdict else None,
        } for ch in challenges],
        indent=2,
    )

    b_evidence_json = json.dumps(
        [{"chunk_id": c.chunk_id, "source": c.source, "tier": c.tier, "text": c.text}
         for c in agent_b_evidence],
        indent=2,
    )

    prompt = _REVISE_PROMPT.format(
        current_verdicts_json=current_json,
        challenges_json=challenges_json,
        agent_b_evidence_json=b_evidence_json,
    )

    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
    )

    raw = _clean_json_array(response.choices[0].message.content)
    revised_items = json.loads(raw)
    revised_map = {v["claim_id"]: v for v in revised_items}

    updated: List[Claim] = []
    for claim in claims:
        v = revised_map.get(claim.claim_id, {})
        # Merge both agents' evidence chunk IDs (deduped)
        merged_chunks = list(set(
            claim.evidence_chunks + v.get("evidence_chunk_ids", [])
        ))
        updated.append(claim.model_copy(update={
            "verdict":        Verdict(v.get("verdict", claim.verdict.value if claim.verdict else "IDK")),
            "confidence":     float(v.get("confidence", claim.confidence)),
            "reasoning":      v.get("reasoning", claim.reasoning),
            "evidence_chunks": merged_chunks,
        }))

    return updated


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_json_array(raw: str) -> str:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    return match.group() if match else raw
