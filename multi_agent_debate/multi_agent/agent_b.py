"""
agent_b.py — Agent B: Adversarial Auditor
==========================================

ROLE
----
Agent B is the adversarial challenger. Its job is NOT to verify claims —
that is Agent A's job. Agent B finds what Agent A missed, stated too broadly,
got wrong jurisdiction on, or supported with insufficiently authoritative evidence.

HOW IT WORKS — ASYMMETRIC RAG (the core design decision)
---------------------------------------------------------
Both agents query the SAME regulatory corpus (your JSONL / Qdrant), but
with fundamentally different query strategies:

  Agent A queries:
    "HIPAA requires encryption of ePHI"          ← confirmatory
    → RAG returns: chunks that SUPPORT this claim

  Agent B queries:
    "exception to HIPAA encryption addressable"  ← adversarial
    "amendment update HIPAA encryption 2023"     ← currency check
    "jurisdiction scope HIPAA encryption limit"  ← scope challenge
    → RAG returns: chunks that QUALIFY, LIMIT, or CONTRADICT

Same corpus. Opposite intent. This creates genuine debate tension.
Without asymmetric RAG, both agents would see the same chunks and agree —
which is self-consistency checking, not debate.

THE TRUE→SKEPTIC RULE
---------------------
When Agent A marks any claim SUPPORTED, Agent B is REQUIRED to attempt
all 4 challenge types before accepting it. This prevents early collapse
into agreement.

3 CHALLENGE TYPES
-----------------
1. CHUNK_CURRENCY      — Has this regulation been updated or superseded?
2. JURISDICTION_SCOPE  — Does it apply to this specific sector/country/entity?
3. EXCEPTION_EXISTENCE — Does a carve-out, safe harbour, or alternative path apply?

NOTE: TIER_OVERRIDE removed — corpus tier metadata (T0/T1/T2/T3) does not
reflect document authority (OWASP and ISO 27001 are both T0). Will be
re-added once corpus has reliable authority_level field per document.

GAP FINDING
-----------
Independent of specific claims — Agent B searches for obligations the
LLM answer missed entirely. These get claim_id=0.

NO WEB SEARCH
-------------
Web search was removed. Uncontrolled external sources introduce noise
and make results non-reproducible. Asymmetric RAG over your curated
corpus is the correct approach for a compliance system.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

from openai import OpenAI

from mad.config import (
    OLLAMA_BASE_URL, OLLAMA_API_KEY, AGENT_MODEL,
    TOP_K_CHALLENGE_CHUNKS,
)
from mad.models import Claim, Challenge, ChallengeType, EvidenceChunk, Verdict
from mad import rag_stub

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are Agent B — a rigorous regulatory compliance auditor and adversarial challenger. "
    "Your sole job is to find flaws in compliance claims: exceptions, outdated references, "
    "wrong jurisdictions, missing caveats, and higher-authority contradictions. "
    "You never accept SUPPORTED without attempting all 3 challenge types. "
    "Only cite chunk_ids that appear in the evidence provided to you. "
    "Never fabricate regulatory text or invent chunk IDs."
)

# ── Challenge prompt ───────────────────────────────────────────────────────────
_CHALLENGE_PROMPT = """\
You are Agent B — Adversarial Auditor.

TRUE→SKEPTIC RULE: For every SUPPORTED claim, attempt ALL 3 challenge types.

━━━ 3 MANDATORY CHALLENGE TYPES (for every SUPPORTED claim) ━━━

1. CHUNK_CURRENCY
   Has the cited regulation been amended, updated, or clarified since the
   chunk was written? Look for newer guidance that changes the picture.

2. JURISDICTION_SCOPE
   Does this regulation apply to THIS specific context?
   Right country/sector/entity size/data type?

3. EXCEPTION_EXISTENCE
   Does an exception, safe harbour, alternative compliance path,
   or carve-out exist that qualifies or changes the verdict?

━━━ GAP_FINDING (always run) ━━━
Find regulatory obligations or required caveats the LLM answer missed
entirely. Set claim_id=0, suggested_verdict=null.

━━━ FOR OTHER VERDICTS ━━━
PARTIAL     → challenge the weakest unsupported aspect
NOT_SUPPORTED → add corroborating evidence if available
IDK         → try a different retrieval angle

━━━ STRICT RULES ━━━
- Only cite chunk_ids present in the evidence JSON below
- If no evidence supports a challenge, state that honestly — do not fabricate
- suggested_verdict: SUPPORTED / PARTIAL / NOT_SUPPORTED / IDK / null

━━━ Agent A's current verdicts ━━━
{verdicts_json}

━━━ Evidence you retrieved (asymmetric adversarial RAG) ━━━
{evidence_json}

Return ONLY a valid JSON array. No markdown, no text outside the JSON.
Output schema (replace all placeholder values with your actual analysis):

[
  {{
    "claim_id": <integer claim_id from Agent A's verdicts above>,
    "challenge_type": "<CHUNK_CURRENCY | JURISDICTION_SCOPE | EXCEPTION_EXISTENCE | GAP_FINDING>",
    "challenge_text": "<your specific challenge based on the evidence above — quote the relevant chunk_id and explain the discrepancy>",
    "suggested_verdict": "<SUPPORTED | PARTIAL | NOT_SUPPORTED | IDK | null>",
    "evidence_chunk_ids": ["<chunk_id from the evidence above>"]
  }}
]
"""


# ── Public function ────────────────────────────────────────────────────────────

def challenge_claims(
    query:             str,
    agent_a_claims:    List[Claim],
    existing_evidence: List[EvidenceChunk],
    cycle:             int = 1,
) -> Tuple[List[Challenge], List[EvidenceChunk]]:
    """
    Agent B's main entry point — called once per cycle after Agent A's step A.

    What happens:
      1. Seed evidence map with everything Agent A already retrieved
      2. Run 3 adversarial RAG queries per SUPPORTED claim (asymmetric access)
      3. Run tier-1-only retrieval for TIER_OVERRIDE when A cited tier-2/3 evidence
      4. Run targeted retrieval for PARTIAL and IDK claims
      5. Run gap-finding queries for the overall topic
      6. Call LLM with Agent A's verdicts + B's adversarial evidence
      7. Parse and return structured Challenge objects

    Returns:
        challenges   — Challenge objects for Agent A to respond to in step C
        b_evidence   — merged evidence pool (A's + B's new chunks, deduped)
    """
    # ── 1. SEED WITH AGENT A'S EVIDENCE ───────────────────────────────────────
    # B sees everything A already retrieved — needed to understand what A cited
    evidence_map: Dict[str, EvidenceChunk] = {
        c.chunk_id: c for c in existing_evidence
    }

    supported  = [c for c in agent_a_claims if c.verdict == Verdict.SUPPORTED]
    partial    = [c for c in agent_a_claims if c.verdict == Verdict.PARTIAL]
    idk_claims = [c for c in agent_a_claims if c.verdict == Verdict.IDK]

    # ── 2. ASYMMETRIC RAG FOR SUPPORTED CLAIMS ────────────────────────────────
    # Three query angles that are the OPPOSITE of what Agent A used.
    # Agent A found confirming evidence. Agent B now finds disconfirming evidence.
    for claim in supported:
        _retrieve_into(
            f"amendment update superseded revision {claim.claim_text}",
            evidence_map, top_k=TOP_K_CHALLENGE_CHUNKS
        )
        _retrieve_into(
            f"jurisdiction scope does not apply limitation {claim.claim_text}",
            evidence_map, top_k=2
        )
        _retrieve_into(
            f"exception carve-out safe harbour alternative {claim.claim_text}",
            evidence_map, top_k=TOP_K_CHALLENGE_CHUNKS
        )

    # ── 3. PARTIAL AND IDK CLAIMS ─────────────────────────────────────────────
    for claim in partial:
        _retrieve_into(
            f"incomplete missing condition requirement {claim.claim_text}",
            evidence_map, top_k=TOP_K_CHALLENGE_CHUNKS
        )
    for claim in idk_claims:
        _retrieve_into(claim.claim_text, evidence_map, top_k=2)

    # ── 5. GAP-FINDING RETRIEVAL ──────────────────────────────────────────────
    # Anchor gap-finding queries with the detected regulation name so that
    # Qdrant returns regulation-specific chunks rather than off-topic EU law.
    reg_anchor = _detect_regulation(query)
    _retrieve_into(
        f"{reg_anchor} also required additionally must {query}",
        evidence_map, top_k=3
    )
    _retrieve_into(
        f"{reg_anchor} prerequisite condition exception {query}",
        evidence_map, top_k=2
    )
    # Removed: "international equivalent cross-border" query — it pulls EU AI
    # Act / DSA / NIS2 chunks into every GDPR/HIPAA query indiscriminately.

    b_evidence = list(evidence_map.values())

    # ── 6. BUILD PROMPT ────────────────────────────────────────────────────────
    # Agent B sees Agent A's confidence scores (unlike the Judge).
    # This helps B prioritise — focus harder on claims A is uncertain about.
    verdicts_json = json.dumps(
        [{
            "claim_id":    c.claim_id,
            "claim_text":  c.claim_text,
            "verdict":     c.verdict.value if c.verdict else "IDK",
            "confidence":  c.confidence,
            "reasoning":   c.reasoning,
            "is_material": c.is_material,
        } for c in agent_a_claims],
        indent=2,
    )

    evidence_json = json.dumps(
        [{
            "chunk_id": c.chunk_id,
            "source":   c.source,
            "text":     c.text,
        } for c in b_evidence],
        indent=2,
    )

    prompt = _CHALLENGE_PROMPT.format(
        verdicts_json=verdicts_json,
        evidence_json=evidence_json,
    )

    # ── 7. CALL LLM ────────────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        # 0.2 = slightly exploratory. Agent B needs to consider adversarial angles.
        # Agent A uses 0.1. Judge uses 0.0.
    )

    raw = _clean_json(response.choices[0].message.content)

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Agent B] Warning: could not parse challenge JSON (cycle {cycle}).")
        print(f"          First 300 chars: {raw[:300]}")
        return [], b_evidence

    # ── 8. PARSE CHALLENGES ────────────────────────────────────────────────────
    challenges: List[Challenge] = []
    for item in items:
        try:
            ctype = item.get("challenge_type", "GAP_FINDING").upper().strip()
            if ctype not in ChallengeType._value2member_map_:
                ctype = "GAP_FINDING"
            sv_raw    = item.get("suggested_verdict")
            suggested = (
                Verdict(sv_raw)
                if sv_raw and sv_raw in Verdict._value2member_map_
                else None
            )
            challenges.append(Challenge(
                claim_id=int(item.get("claim_id", 0)),
                challenge_type=ChallengeType(ctype),
                challenge_text=str(item.get("challenge_text", "")).strip(),
                evidence_chunks=item.get("evidence_chunk_ids", []),
                suggested_verdict=suggested,
            ))
        except Exception as e:
            print(f"[Agent B] Skipping malformed challenge item: {e}")
            continue

    print(f"  [Agent B] {len(challenges)} challenges in cycle {cycle}")
    return challenges, b_evidence


# ── Helpers ────────────────────────────────────────────────────────────────────

_REGULATION_KEYWORDS = [
    # Ordered by specificity — first match wins
    ("GDPR",         ["gdpr", "general data protection", "regulation 2016/679"]),
    ("HIPAA",        ["hipaa", "health insurance portability", "protected health information", "phi", "ephi"]),
    ("EU AI Act",    ["eu ai act", "ai act", "artificial intelligence act"]),
    ("NIS2",         ["nis2", "nis 2", "network and information security directive"]),
    ("CCPA",         ["ccpa", "ccpa/cpra", "cpra", "california consumer privacy"]),
    ("HITECH",       ["hitech", "health information technology"]),
    ("ISO 27001",    ["iso 27001", "iso27001"]),
    ("NIST",         ["nist csf", "nist sp", "nist cybersecurity"]),
    ("DSA",          ["digital services act", " dsa "]),
    ("CRA",          ["cyber resilience act", " cra "]),
    ("NIS2",         ["nis2"]),
    ("PIPL",         ["pipl", "china personal information"]),
    ("PDPA",         ["pdpa", "personal data protection act"]),
    ("PDPL",         ["pdpl", "saudi personal data"]),
    ("APPI",         ["japan appi", "act on the protection of personal information"]),
]


def _detect_regulation(query: str) -> str:
    """
    Return the primary regulation name mentioned in the query.
    Used to anchor retrieval queries so gap-finding stays within the
    relevant regulatory domain and doesn't pull in off-topic EU law.
    Falls back to empty string (no anchor) if nothing matches.
    """
    q = query.lower()
    for name, keywords in _REGULATION_KEYWORDS:
        if any(kw in q for kw in keywords):
            return name
    return ""


def _retrieve_into(
    query:  str,
    target: Dict[str, EvidenceChunk],
    top_k:  int = 3,
) -> None:
    """Retrieve chunks and merge into target dict (deduped by chunk_id)."""
    for chunk in rag_stub.retrieve(query, top_k=top_k):
        target[chunk.chunk_id] = chunk


def _clean_json(raw: str) -> str:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    raw = raw.strip()
    m   = re.search(r"\[.*\]", raw, re.DOTALL)
    return m.group() if m else raw
