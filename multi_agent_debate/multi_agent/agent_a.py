"""
agent_a.py — Agent A: Ground Truth Verifier
============================================

ROLE
----
Agent A is the primary verifier. It reads the enterprise LLM's answer,
extracts atomic claims (via claim_extractor.py), and verifies each claim
against retrieved regulatory evidence.

WHAT IT DOES AT EACH STAGE
---------------------------
Step A — Initial Verification (cycle 1 only):
  - Calls RAG with the claim text as query (confirmatory retrieval)
  - Assigns: SUPPORTED / PARTIAL / NOT_SUPPORTED / IDK
  - Assigns calibrated confidence p (0.0–1.0)
  - Returns per-claim prompts for storage (GRPO training input)

Step C — Revision (after Agent B challenges, every cycle):
  - Reads Agent B's challenges
  - Updates verdict and confidence if B provided valid new evidence
  - CANNOT change the original LLM claim text — only verdict/confidence/reasoning
  - Returns per-claim prompts for storage (base prompt + B's challenge appended)

WHY WE RETURN PER-CLAIM PROMPTS
--------------------------------
The GRPO trainer (TRL) needs exactly what Agent A received as input
at each checkpoint to train on. At post_step_A, this is:
  system + query + RAG chunks + claim text

At post_cycle1/2, this is:
  above + Agent B's challenge for this specific claim

Without these prompts, TRL has no training input — it cannot reproduce
the context and learn from the Brier reward signal.

The LLM calls themselves are batched (all claims at once) for efficiency.
The per-claim prompts are built separately for storage only.

CONFIDENCE CALIBRATION
----------------------
Agent A's confidence p represents its calibrated belief that its verdict
is correct. This feeds the Brier reward:
  brier_reward = 2 * p * v_label - p^2

Where v_label is the Judge's final score (1.0/0.5/0.0).
Good calibration means: when p=0.9, the claim really is verified ~90% of the time.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

from openai import OpenAI

from .config import OLLAMA_BASE_URL, OLLAMA_API_KEY, AGENT_MODEL, TOP_K_CHUNKS
from .models import Claim, Challenge, EvidenceChunk, Verdict
from . import rag_stub

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# ── System prompt ──────────────────────────────────────────────────────────────
# This is ALSO stored as part of agent_a_prompt in the claims table.
AGENT_A_SYSTEM = (
    "You are Agent A — a regulatory compliance expert and Ground Truth Verifier. "
    "Your job: verify whether each claim made by an enterprise AI is accurate "
    "according to the retrieved regulatory evidence. "
    "Be precise and calibrated. Never over-claim certainty. "
    "Cite the specific chunk_ids that support your verdict."
)

# ── Verification prompt ────────────────────────────────────────────────────────
_VERIFY_PROMPT = """\
You are verifying claims made by an enterprise AI assistant about regulatory compliance.

For each claim below, evaluate it against the retrieved evidence chunks and assign:

VERDICT:
  SUPPORTED     — evidence directly and fully supports the claim as stated
  PARTIAL       — evidence partially supports it but with important caveats or limits
  NOT_SUPPORTED — evidence contradicts the claim, OR claim is not found in any evidence
  IDK           — no relevant evidence retrieved; cannot make a determination

CONFIDENCE (float 0.0–1.0):
  Your calibrated belief that your verdict is correct.
  0.90+ = very strong evidence directly on point
  0.70  = moderate evidence, some interpretation needed
  0.50  = uncertain, evidence is ambiguous
  0.30  = weak, mostly inferring from context
  Never use exactly 0.0 or 1.0 — these are priors, not certainties.

Retrieved regulatory evidence:
{evidence_json}

Claims to verify:
{claims_json}

Return ONLY a valid JSON array. No markdown fences, no explanation.

[
  {{
    "claim_id": 1,
    "verdict": "SUPPORTED",
    "confidence": 0.82,
    "reasoning": "Chunk hipaa_164_502 directly states minimum necessary applies...",
    "evidence_chunk_ids": ["hipaa_164_502_uses_disclosures"]
  }}
]
"""

# ── Revision prompt ────────────────────────────────────────────────────────────
_REVISE_PROMPT = """\
You are Agent A — Ground Truth Verifier. You have completed initial verification.
Agent B has now challenged some of your verdicts.

REVISION RULES:
- If Agent B cites new regulatory evidence that genuinely changes the picture: UPDATE verdict and confidence.
- If Agent B restates the same point without new evidence: MAINTAIN your position.
- If Agent B attacks a well-supported claim with no new evidence (gaslighting): MAINTAIN position, confidence drops at most 0.05.
- You CANNOT change the claim text — only verdict, confidence, reasoning, evidence_chunk_ids.
- Include ALL claims in your response, even unchanged ones.
- Acknowledge good challenges explicitly in your reasoning.

Your current verdicts:
{current_json}

Agent B's challenges:
{challenges_json}

Additional evidence Agent B retrieved:
{b_evidence_json}

Return ONLY a valid JSON array. No markdown.

[
  {{
    "claim_id": 1,
    "verdict": "PARTIAL",
    "confidence": 0.54,
    "reasoning": "Revised after B's CHUNK_CURRENCY challenge — 2023 HHS OCR guidance confirms encryption is addressable, not mandatory. B's evidence is valid.",
    "evidence_chunk_ids": ["hipaa_164_312_technical_safeguards", "hhs_ocr_encryption_guidance_2023"]
  }}
]
"""


# ── Public functions ───────────────────────────────────────────────────────────

def verify_claims(
    query:  str,
    claims: List[Claim],
) -> Tuple[List[Claim], List[EvidenceChunk], Dict[int, str]]:
    """
    Step A: Initial RAG-based verification of all claims.

    Returns:
        updated_claims    — claims with verdicts + confidence assigned
        all_evidence      — all RAG chunks retrieved (for evidence pool)
        per_claim_prompts — dict[claim_id → prompt_str] for GRPO storage
                            Each prompt is what Agent A "saw" for that specific claim.
    """
    # ── Retrieve evidence for all claims ──────────────────────────────────────
    evidence_map: Dict[str, EvidenceChunk] = {}

    # Per-claim retrieval (confirmatory — query = claim text itself)
    for claim in claims:
        for chunk in rag_stub.retrieve(claim.claim_text, top_k=TOP_K_CHUNKS):
            evidence_map[chunk.chunk_id] = chunk

    # Overall query retrieval (catches context chunks A might need)
    for chunk in rag_stub.retrieve(query, top_k=3):
        evidence_map[chunk.chunk_id] = chunk

    all_evidence = list(evidence_map.values())

    # ── Call LLM (batched — all claims in one call for efficiency) ────────────
    evidence_json = json.dumps(
        [{"chunk_id": c.chunk_id, "source": c.source,
          "tier": c.tier, "text": c.text} for c in all_evidence],
        indent=2,
    )
    claims_json = json.dumps(
        [{"claim_id": c.claim_id, "claim_text": c.claim_text,
          "is_material": c.is_material} for c in claims],
        indent=2,
    )

    prompt = _VERIFY_PROMPT.format(
        evidence_json=evidence_json,
        claims_json=claims_json,
    )

    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": AGENT_A_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        # 0.1 = near-deterministic. Verification needs to be consistent
        # across multiple rollouts for Brier reward comparison to be meaningful.
    )

    raw          = _clean_json(response.choices[0].message.content)
    verdict_map  = {v["claim_id"]: v for v in json.loads(raw)}
    updated      = _apply_verdicts(claims, verdict_map)

    # ── Build per-claim prompts for GRPO storage ───────────────────────────────
    # Each claim gets its own prompt string representing what Agent A "saw"
    # before outputting its confidence at this checkpoint.
    per_claim_prompts: Dict[int, str] = {}
    for claim in updated:
        # Only include the evidence chunks relevant to this specific claim
        claim_chunks = [
            c for c in all_evidence
            if c.chunk_id in (verdict_map.get(claim.claim_id, {}).get("evidence_chunk_ids", [])
                              or [c.chunk_id for c in all_evidence])
        ][:TOP_K_CHUNKS]

        per_claim_prompts[claim.claim_id] = _build_step_a_prompt(
            query=query,
            claim_text=claim.claim_text,
            evidence_chunks=claim_chunks,
        )

    return updated, all_evidence, per_claim_prompts


def revise_verdicts(
    claims:           List[Claim],
    challenges:       List[Challenge],
    agent_b_evidence: List[EvidenceChunk],
    base_prompts:     Dict[int, str],
    cycle:            int,
) -> Tuple[List[Claim], Dict[int, str]]:
    """
    Step C: Revise verdicts after Agent B's challenges.

    Cannot change claim text — only verdict, confidence, reasoning, evidence.

    Args:
        claims           — current claims (before revision)
        challenges       — Agent B's challenges this cycle
        agent_b_evidence — evidence B retrieved
        base_prompts     — per-claim prompts from previous checkpoint
                           (used to build the new checkpoint prompts)
        cycle            — which cycle number (1 or 2)

    Returns:
        revised_claims    — updated claims
        per_claim_prompts — dict[claim_id → prompt_str] for GRPO storage
                            Each prompt = base_prompt + B's challenge for this claim
    """
    current_json = json.dumps(
        [{
            "claim_id":           c.claim_id,
            "claim_text":         c.claim_text,
            "verdict":            c.verdict.value if c.verdict else "IDK",
            "confidence":         c.confidence,
            "reasoning":          c.reasoning,
            "is_material":        c.is_material,
            "evidence_chunk_ids": c.evidence_chunks,
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
        [{"chunk_id": c.chunk_id, "source": c.source,
          "tier": c.tier, "text": c.text} for c in agent_b_evidence],
        indent=2,
    )

    prompt = _REVISE_PROMPT.format(
        current_json=current_json,
        challenges_json=challenges_json,
        b_evidence_json=b_evidence_json,
    )

    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": AGENT_A_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
    )

    raw         = _clean_json(response.choices[0].message.content)
    revised_map = {v["claim_id"]: v for v in json.loads(raw)}

    # Merge both agents' evidence chunks
    revised = []
    for claim in claims:
        v = revised_map.get(claim.claim_id, {})
        merged = list(set(claim.evidence_chunks + v.get("evidence_chunk_ids", [])))
        revised.append(claim.model_copy(update={
            "verdict":        _safe_verdict(v.get("verdict", claim.verdict.value if claim.verdict else "IDK")),
            "confidence":     float(v.get("confidence", claim.confidence)),
            "reasoning":      v.get("reasoning", claim.reasoning),
            "evidence_chunks": merged,
        }))

    # ── Build per-claim prompts for GRPO storage ───────────────────────────────
    # Build a lookup of challenges per claim
    challenge_map: Dict[int, str] = {}
    for ch in challenges:
        # Concatenate multiple challenges for same claim
        existing = challenge_map.get(ch.claim_id, "")
        separator = "\n" if existing else ""
        challenge_map[ch.claim_id] = (
            f"{existing}{separator}"
            f"[{ch.challenge_type.value}]: {ch.challenge_text}"
        )

    checkpoint = f"post_cycle{cycle}"
    per_claim_prompts: Dict[int, str] = {}
    for claim in revised:
        base    = base_prompts.get(claim.claim_id, "")
        b_text  = challenge_map.get(claim.claim_id, "(No challenge raised for this claim)")
        per_claim_prompts[claim.claim_id] = _build_revision_prompt(
            base_prompt=base,
            cycle=cycle,
            b_challenge_text=b_text,
        )

    return revised, per_claim_prompts


# ── Prompt builders (used both for LLM calls and for GRPO storage) ─────────────

def _build_step_a_prompt(
    query:          str,
    claim_text:     str,
    evidence_chunks: List[EvidenceChunk],
) -> str:
    """
    Builds the stored agent_a_prompt for checkpoint post_step_A.

    This is the GRPO training input for Agent A's initial verification.
    Contains: system + query + RAG chunks + claim text + instruction.
    """
    chunks_text = "\n".join(
        f"[{c.chunk_id}] (tier {c.tier}, source: {c.source}):\n{c.text[:350]}"
        for c in evidence_chunks
    )
    return (
        f"System: {AGENT_A_SYSTEM}\n\n"
        f"User query: {query}\n\n"
        f"Retrieved regulatory evidence:\n{chunks_text}\n\n"
        f"Claim to verify: {claim_text}\n\n"
        f"Output your verdict (SUPPORTED/PARTIAL/NOT_SUPPORTED/IDK) "
        f"and confidence (0.0 to 1.0):"
    )


def _build_revision_prompt(
    base_prompt:      str,
    cycle:            int,
    b_challenge_text: str,
) -> str:
    """
    Builds the stored agent_a_prompt for checkpoint post_cycle1 or post_cycle2.

    Takes the previous checkpoint's prompt and appends Agent B's challenge.
    The prompt grows at each checkpoint — Agent A sees the full challenge history.
    """
    return (
        f"{base_prompt}\n\n"
        f"--- Agent B Cycle {cycle} Challenge ---\n"
        f"{b_challenge_text}\n\n"
        f"Review B's challenge against your evidence. "
        f"If B identified a genuine regulatory gap or exception, lower your confidence. "
        f"If B is attacking a well-supported claim without new evidence, maintain your position. "
        f"Output your revised verdict and confidence:"
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _normalise_verdict(raw: str) -> str:
    """Normalise LLM verdict strings to enum values (e.g. 'NOT SUPPORTED' → 'NOT_SUPPORTED')."""
    return raw.strip().upper().replace(" ", "_")


def _safe_verdict(raw: str) -> Verdict:
    try:
        return Verdict(_normalise_verdict(raw))
    except ValueError:
        return Verdict.IDK


def _apply_verdicts(claims: List[Claim], verdict_map: Dict) -> List[Claim]:
    updated = []
    for c in claims:
        v = verdict_map.get(c.claim_id, {})
        updated.append(c.model_copy(update={
            "verdict":        _safe_verdict(v.get("verdict", "IDK")),
            "confidence":     float(v.get("confidence", 0.5)),
            "reasoning":      v.get("reasoning", ""),
            "evidence_chunks": v.get("evidence_chunk_ids", []),
        }))
    return updated


def _clean_json(raw: str) -> str:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    return match.group() if match else raw
