"""
agent_b.py — Agent B: Adversarial Auditor

True→Skeptic Rule:
  When Agent A says SUPPORTED, Agent B MUST escalate skepticism and attempt ALL 4 challenge types.

4 mandatory challenge types (per SUPPORTED claim):
  1. CHUNK_CURRENCY      — Is the cited chunk current? Has the regulation been amended?
  2. JURISDICTION_SCOPE  — Does this regulation actually apply to this context/jurisdiction?
  3. EXCEPTION_EXISTENCE — Does a carve-out, exception, or safe harbour apply?
  4. TIER_OVERRIDE       — Does a higher-authority source contradict or qualify this?

Gap-finding obligation:
  Find regulatory angles, related obligations, or important qualifications
  that Agent A's answer completely missed.

Tools: RAG (Qdrant stub) + Tavily web search
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple, Optional

from openai import OpenAI

from multi_agent.config import (
    OLLAMA_BASE_URL, OLLAMA_API_KEY, AGENT_MODEL,
    TAVILY_API_KEY, TOP_K_CHALLENGE_CHUNKS,
)
from multi_agent.models import Claim, Challenge, ChallengeType, EvidenceChunk, Verdict
from multi_agent import rag_stub

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# Tavily setup — graceful fallback if key not provided
_tavily = None
if TAVILY_API_KEY:
    try:
        from tavily import TavilyClient
        _tavily = TavilyClient(api_key=TAVILY_API_KEY)
    except ImportError:
        print("[Agent B] Warning: tavily-python not installed. Web search disabled.")

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are Agent B — a rigorous regulatory auditor and Adversarial Auditor. "
    "Your job: find flaws, exceptions, outdated references, jurisdiction limits, "
    "and missing regulatory context in compliance claims. Be precise. Use evidence. "
    "Never make up evidence — only use what is provided to you."
)

# ── Challenge prompt ──────────────────────────────────────────────────────────
_CHALLENGE_PROMPT = """\
You are Agent B — Adversarial Auditor. You follow the True→Skeptic rule:
When Agent A marks a claim as SUPPORTED, you MUST challenge it across all 4 types.

4 MANDATORY CHALLENGE TYPES (attempt ALL for every SUPPORTED claim):
1. CHUNK_CURRENCY      — Is Agent A's cited evidence current? Has the regulation been updated/amended since?
2. JURISDICTION_SCOPE  — Does this regulation apply to the specific context, jurisdiction, or entity type in the query?
3. EXCEPTION_EXISTENCE — Does an exception, safe harbour, carve-out, or alternative compliance path apply?
4. TIER_OVERRIDE       — Does a higher-authority source (primary law > guidance > framework) contradict or qualify Agent A's finding?

GAP_FINDING (always run this):
  Find regulatory requirements, related obligations, or important qualifications entirely absent from Agent A's claims.
  These are things the LLM answer should have mentioned but didn't.

For PARTIAL claims: challenge the weakest unsupported aspect.
For NOT_SUPPORTED claims: confirm with additional evidence if available.

Agent A's current verdicts:
{verdicts_json}

Evidence you retrieved (RAG):
{evidence_json}

Web search results (if available):
{web_results}

IMPORTANT:
- Only reference chunk_ids that actually appear in the evidence JSON above.
- If no evidence supports a challenge, say so honestly — do not fabricate.
- suggested_verdict must be one of: SUPPORTED, PARTIAL, NOT_SUPPORTED, IDK

Return ONLY a valid JSON array of challenges. No markdown, no explanation.

[
  {{
    "claim_id": 1,
    "challenge_type": "CHUNK_CURRENCY",
    "challenge_text": "The GDPR Art 32 chunk does not reflect 2024 EDPB guidance which clarifies AES-256 is not mandated...",
    "suggested_verdict": "PARTIAL",
    "evidence_chunk_ids": ["edpb_guidelines_2024_encryption"]
  }},
  {{
    "claim_id": 0,
    "challenge_type": "GAP_FINDING",
    "challenge_text": "The answer omits that Art 32(3) allows alternative compliance via approved codes of conduct...",
    "suggested_verdict": null,
    "evidence_chunk_ids": ["gdpr_art32_p3_exceptions"]
  }}
]
"""


# ── Public function ───────────────────────────────────────────────────────────

def challenge_claims(
    query:             str,
    agent_a_claims:    List[Claim],
    existing_evidence: List[EvidenceChunk],
    cycle:             int = 1,
) -> Tuple[List[Challenge], List[EvidenceChunk]]:
    """
    Generate challenges against Agent A's verdicts.
    Returns: (challenges, additional evidence chunks retrieved by B)
    """
    evidence_map: dict[str, EvidenceChunk] = {c.chunk_id: c for c in existing_evidence}

    # ── B retrieves additional evidence focused on exceptions / amendments ──
    supported = [c for c in agent_a_claims if c.verdict == Verdict.SUPPORTED]
    partial   = [c for c in agent_a_claims if c.verdict == Verdict.PARTIAL]

    # For each SUPPORTED claim, retrieve from 3 adversarial angles
    for claim in supported:
        _retrieve_into(f"exception to: {claim.claim_text}",           evidence_map, top_k=TOP_K_CHALLENGE_CHUNKS)
        _retrieve_into(f"amendment update limitation: {claim.claim_text}", evidence_map, top_k=2)
        _retrieve_into(f"jurisdiction scope limit: {claim.claim_text}", evidence_map, top_k=2)

    # For PARTIAL claims, dig into the unresolved part
    for claim in partial:
        _retrieve_into(claim.claim_text, evidence_map, top_k=TOP_K_CHALLENGE_CHUNKS)

    # Gap-finding: look for related obligations the answer didn't mention
    _retrieve_into(f"related regulatory requirements {query}", evidence_map, top_k=3)
    _retrieve_into(f"alternatives to compliance {query}",      evidence_map, top_k=2)

    b_new_evidence = list(evidence_map.values())

    # ── Tavily web search for SUPPORTED claims ──
    web_results = _web_search(query, supported)

    # ── Build prompt ──
    verdicts_json = json.dumps(
        [{
            "claim_id":   c.claim_id,
            "claim_text": c.claim_text,
            "verdict":    c.verdict.value if c.verdict else "IDK",
            "confidence": c.confidence,
            "reasoning":  c.reasoning,
            "is_material": c.is_material,
        } for c in agent_a_claims],
        indent=2,
    )

    evidence_json = json.dumps(
        [{"chunk_id": c.chunk_id, "source": c.source, "tier": c.tier, "text": c.text}
         for c in b_new_evidence],
        indent=2,
    )

    prompt = _CHALLENGE_PROMPT.format(
        verdicts_json=verdicts_json,
        evidence_json=evidence_json,
        web_results=web_results or "No web search results available.",
    )

    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,    # slightly higher than A — B is meant to explore
    )

    raw = _clean_json_array(response.choices[0].message.content)

    try:
        challenge_items = json.loads(raw)
    except json.JSONDecodeError:
        # Non-fatal — debate continues without challenges this cycle
        print(f"[Agent B] Warning: could not parse challenge JSON. Raw:\n{raw[:300]}")
        return [], b_new_evidence

    challenges: List[Challenge] = []
    for item in challenge_items:
        try:
            ctype_raw = item.get("challenge_type", "GAP_FINDING").upper()
            # Map to valid enum value
            if ctype_raw not in ChallengeType._value2member_map_:
                ctype_raw = "GAP_FINDING"

            sv = item.get("suggested_verdict")
            suggested = Verdict(sv) if sv and sv in Verdict._value2member_map_ else None

            challenges.append(Challenge(
                claim_id=int(item["claim_id"]),
                challenge_type=ChallengeType(ctype_raw),
                challenge_text=str(item["challenge_text"]).strip(),
                evidence_chunks=item.get("evidence_chunk_ids", []),
                suggested_verdict=suggested,
            ))
        except Exception as e:
            print(f"[Agent B] Skipping malformed challenge item: {e} — {item}")
            continue

    return challenges, b_new_evidence


# ── Helpers ───────────────────────────────────────────────────────────────────

def _retrieve_into(query: str, target: dict, top_k: int = 3) -> None:
    """Retrieve chunks and merge into target dict (deduped by chunk_id)."""
    for chunk in rag_stub.retrieve(query, top_k=top_k):
        target[chunk.chunk_id] = chunk


def _web_search(query: str, supported_claims: List[Claim]) -> Optional[str]:
    """Run Tavily web search for SUPPORTED claims. Returns formatted string or None."""
    if not _tavily or not supported_claims:
        return None

    # Build a targeted web query
    claim_texts = " ".join(c.claim_text[:60] for c in supported_claims[:2])
    web_query = f"regulatory compliance exceptions limitations {query} {claim_texts} 2024"

    try:
        results = _tavily.search(web_query, max_results=4, search_depth="basic")
        snippets: List[str] = []
        for r in results.get("results", []):
            url     = r.get("url", "")
            content = r.get("content", "")[:400]
            snippets.append(f"[{url}]\n{content}")
        return "\n\n".join(snippets) if snippets else None
    except Exception as e:
        print(f"[Agent B] Tavily web search failed: {e}")
        return None


def _clean_json_array(raw: str) -> str:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    return match.group() if match else raw
