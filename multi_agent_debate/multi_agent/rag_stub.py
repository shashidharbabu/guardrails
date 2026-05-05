"""
rag_stub.py — Stub RAG retriever backed by your local JSONL chunk file.
=======================================================================

INTERFACE CONTRACT (identical to the real Qdrant retriever)
------------------------------------------------------------
    retrieve(query, top_k) -> List[EvidenceChunk]

To swap in real Qdrant later, replace ONLY the body of retrieve().
Everything else (agent_a, agent_b, judge, debate_engine) calls
retrieve() unchanged.

TIER SYSTEM
-----------
Tier 1 — Primary law / statute (GDPR, HIPAA, CCPA, ADA)
Tier 2 — Official guidance / regulatory body interpretation (EDPB, HHS OCR)
Tier 3 — Framework / best practice (NIST, OWASP, ISO)
Tier 4 — Commentary / unofficial guidance

NOTE: Tier filtering removed — corpus tier metadata does not reliably
reflect document authority. Will be re-added once corpus has a validated
authority_level field. Retrieval is currently relevance-only.

HEALTHCARE DOMAIN
-----------------
Sample chunks cover the Healthcare domain (your paper's primary eval domain)
plus GDPR/HIPAA for cross-regulation testing. This mirrors the realistic
corpus your Qdrant DB contains.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List

from multi_agent.config import CHUNKS_JSONL_PATH, TOP_K_CHUNKS
from multi_agent.models import EvidenceChunk

# ── Module-level cache — loaded once per process ───────────────────────────────
_chunk_cache: List[Dict] = []
_loaded: bool = False

# ── Retrieval result cache — (query, top_k) → List[EvidenceChunk] ──────────────
# Eliminates duplicate Qdrant round-trips when Agent B re-runs the same
# gap-finding queries in cycle 2 that it already ran in cycle 1.
_retrieve_cache: dict = {}


def _load_chunks() -> List[Dict]:
    global _chunk_cache, _loaded
    if _loaded:
        return _chunk_cache

    path = Path(CHUNKS_JSONL_PATH)
    if path.exists():
        print(f"[RAG] Loading chunks from {path}")
        with open(path, encoding="utf-8") as f:
            _chunk_cache = [json.loads(line) for line in f if line.strip()]
        print(f"[RAG] Loaded {len(_chunk_cache)} chunks")
    else:
        print(f"[RAG] JSONL not found at {path} — using built-in sample chunks")
        _chunk_cache = _sample_chunks()

    _loaded = True
    return _chunk_cache


def retrieve(
    query: str,
    top_k: int = TOP_K_CHUNKS,
) -> List[EvidenceChunk]:
    """
    Full RAG pipeline: Qdrant retrieval → SLM verification → EvidenceChunks.

    Delegates to rag.pipeline.retrieve_verified() which:
      1. Embeds the query with Nemotron-8B + instruction prefix
      2. Runs cosine similarity search against ai_governance_chunks_nemotron8b (4,662 chunks)
      3. Passes candidates to Qwen SLM verifier (via Ollama) for selection + ranking
      4. Returns verified EvidenceChunk objects

    Falls back to TF-IDF stub if rag.pipeline is not importable
    (e.g. Qdrant env vars not set, qdrant-client not installed, or Ollama down).
    """
    cache_key = (query, top_k)
    if cache_key in _retrieve_cache:
        return _retrieve_cache[cache_key]

    try:
        from rag.pipeline import retrieve_verified
        result = retrieve_verified(query, top_k)
        _retrieve_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"[RAG] Full pipeline failed — falling back to TF-IDF stub. Reason: {e}")

    # ── Fallback: TF-IDF stub ──────────────────────────────────────────────────
    chunks = _load_chunks()
    if not chunks:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return [_to_evidence(c, i) for i, c in enumerate(chunks[:top_k])]

    # TF-IDF-lite scoring
    idf = _compute_idf(chunks, query_terms)
    scored: List[tuple[float, EvidenceChunk]] = []

    for i, chunk in enumerate(chunks):
        text        = _get_text(chunk)
        chunk_terms = _tokenize(text)
        if not chunk_terms:
            continue
        score = sum(
            (chunk_terms.count(t) / len(chunk_terms)) * idf.get(t, 0.0)
            for t in query_terms
        )
        if score > 0:
            scored.append((score, _to_evidence(chunk, i)))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [ec for _, ec in scored[:top_k]]

    if not results:
        results = [_to_evidence(c, i) for i, c in enumerate(chunks[:top_k])]

    return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _get_text(chunk: Dict) -> str:
    for key in ("text", "content", "chunk_text", "body"):
        if key in chunk:
            return chunk[key]
    return ""


def _to_evidence(chunk: Dict, idx: int) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=str(chunk.get("chunk_id", chunk.get("id", f"chunk_{idx}"))),
        text=_get_text(chunk),
        source=str(chunk.get("source", chunk.get("doc_id",
                   chunk.get("filename", f"doc_{idx}")))),
        tier=int(chunk.get("tier", chunk.get("authority_tier", 1))),
    )


def _compute_idf(chunks: List[Dict], terms: List[str]) -> Dict[str, float]:
    N = len(chunks)
    return {
        t: math.log((N + 1) / (sum(1 for c in chunks if t in _tokenize(_get_text(c))) + 1)) + 1.0
        for t in set(terms)
    }


# ── Built-in sample chunks (Healthcare domain + GDPR/HIPAA) ──────────────────

def _sample_chunks() -> List[Dict]:
    """
    Realistic sample chunks for the Healthcare domain.
    Covers HIPAA, HITECH, ADA, and GDPR for cross-regulation testing.
    Tier assignments reflect actual authority levels.
    """
    return [
        # ── HIPAA PRIMARY LAW (Tier 1) ──────────────────────────────────────
        {
            "chunk_id": "hipaa_164_502_uses_disclosures",
            "text": (
                "45 CFR 164.502 — Uses and Disclosures of Protected Health Information: "
                "General rules. A covered entity or business associate may not use or "
                "disclose protected health information, except as permitted or required "
                "by this subpart. A covered entity may use or disclose protected health "
                "information only if such use or disclosure is permitted or required by "
                "this subpart. The minimum necessary standard applies: covered entities "
                "must make reasonable efforts to limit PHI to the minimum necessary to "
                "accomplish the intended purpose."
            ),
            "source": "HIPAA_Privacy_Rule_45CFR164",
            "tier": 1,
        },
        {
            "chunk_id": "hipaa_164_312_technical_safeguards",
            "text": (
                "45 CFR 164.312 — Technical safeguards. A covered entity or business "
                "associate must implement technical policies and procedures for electronic "
                "information systems that maintain electronic PHI to allow access only to "
                "those persons or software programs that have been granted access rights. "
                "Encryption and Decryption (Addressable): Implement a mechanism to encrypt "
                "and decrypt electronic protected health information. NOTE: This is an "
                "ADDRESSABLE specification — the covered entity must assess whether it is "
                "reasonable and appropriate to implement, and if not, document why and "
                "implement an equivalent alternative. Encryption is NOT mandated — it is "
                "one option among reasonable safeguards."
            ),
            "source": "HIPAA_Security_Rule_45CFR164",
            "tier": 1,
        },
        {
            "chunk_id": "hipaa_164_514_deidentification",
            "text": (
                "45 CFR 164.514 — De-identification of protected health information. "
                "Health information is not individually identifiable and thus not PHI "
                "if either: (1) Expert determination: a qualified statistical expert "
                "determines the risk of identification is very small; OR "
                "(2) Safe Harbor: 18 specific identifiers are removed including names, "
                "geographic data smaller than state, dates except year, phone numbers, "
                "email addresses, SSN, medical record numbers, account numbers, "
                "certificate numbers, VIN, device identifiers, URLs, IP addresses, "
                "biometric identifiers, full-face photos, and any other unique identifier. "
                "De-identified data is NOT subject to HIPAA Privacy Rule."
            ),
            "source": "HIPAA_Privacy_Rule_45CFR164",
            "tier": 1,
        },
        {
            "chunk_id": "hipaa_164_528_accounting_disclosures",
            "text": (
                "45 CFR 164.528 — Accounting of Disclosures of PHI. Individuals have "
                "the right to receive an accounting of disclosures of their PHI made by "
                "a covered entity in the six years prior to the request. This right "
                "applies to disclosures made for purposes OTHER than treatment, payment, "
                "and healthcare operations. Disclosures made pursuant to the individual's "
                "authorization are excluded from the accounting requirement."
            ),
            "source": "HIPAA_Privacy_Rule_45CFR164",
            "tier": 1,
        },
        {
            "chunk_id": "hipaa_164_524_access_rights",
            "text": (
                "45 CFR 164.524 — Access of individuals to protected health information. "
                "Individuals have a right of access to inspect and obtain a copy of PHI "
                "about themselves in a designated record set. Covered entities must "
                "provide access within 30 days (with one 30-day extension). Covered "
                "entities may deny access in limited circumstances including: information "
                "compiled in anticipation of legal proceedings, or where a licensed "
                "health care professional determines access would endanger the patient. "
                "As of 2021 Right of Access Initiative: individuals may request "
                "electronic copies and direct PHI to third parties."
            ),
            "source": "HIPAA_Privacy_Rule_45CFR164",
            "tier": 1,
        },
        {
            "chunk_id": "hipaa_164_410_breach_notification",
            "text": (
                "45 CFR 164.410 — Notification by Business Associates. Following the "
                "discovery of a breach of unsecured PHI, a business associate shall "
                "notify the covered entity of the breach without unreasonable delay and "
                "in no case later than 60 days following discovery. For breaches "
                "affecting 500 or more individuals: covered entity must notify HHS "
                "immediately (within 60 days) and prominent media outlets. For breaches "
                "affecting fewer than 500 individuals: covered entity logs and notifies "
                "HHS annually. Affected individuals must be notified without unreasonable "
                "delay and within 60 days of discovery."
            ),
            "source": "HIPAA_Breach_Notification_Rule_45CFR164",
            "tier": 1,
        },
        # ── HHS OCR OFFICIAL GUIDANCE (Tier 2) ──────────────────────────────
        {
            "chunk_id": "hhs_ocr_minimum_necessary_guidance",
            "text": (
                "HHS OCR Guidance on Minimum Necessary Standard: Covered entities must "
                "develop and implement policies that restrict access and uses of PHI "
                "based on the specific roles of workforce members. The minimum necessary "
                "standard does NOT apply to: disclosures to the individual who is subject "
                "of the information, uses or disclosures for treatment purposes, uses or "
                "disclosures made pursuant to authorisation, disclosures to HHS, or uses "
                "required by law. Routine requests for PHI by payers should be handled "
                "with standard protocols — not a case-by-case minimum necessary analysis."
            ),
            "source": "HHS_OCR_Guidance_MinimumNecessary",
            "tier": 2,
        },
        {
            "chunk_id": "hhs_ocr_encryption_guidance_2023",
            "text": (
                "HHS OCR Guidance 2023 — Encryption as an Addressable Specification: "
                "The HIPAA Security Rule does not mandate encryption of ePHI. However, "
                "HHS OCR strongly recommends encryption as a best practice. In practice, "
                "almost all covered entities that have experienced breaches of unencrypted "
                "ePHI have faced significant penalties. As of 2023, HHS OCR considers "
                "failure to encrypt a significant risk factor in breach investigations. "
                "If an entity chooses not to encrypt, it must document the rationale and "
                "implement an equivalent alternative measure. AES-128 or AES-256 are "
                "acceptable NIST-approved encryption standards but neither is specifically "
                "mandated by HIPAA."
            ),
            "source": "HHS_OCR_Guidance_Encryption_2023",
            "tier": 2,
        },
        {
            "chunk_id": "hhs_ocr_third_party_sharing_guidance",
            "text": (
                "HHS OCR FAQ — Sharing PHI with Third Parties: A covered entity may "
                "disclose PHI to a business associate only if the covered entity has "
                "obtained satisfactory assurances that the business associate will "
                "appropriately safeguard the information (i.e., a signed Business "
                "Associate Agreement, BAA). A BAA alone is NOT sufficient authorisation "
                "to share PHI for any purpose — the disclosure must also fit within "
                "a permitted purpose under 45 CFR 164.502. Sharing PHI with a marketing "
                "firm requires individual authorisation under 45 CFR 164.508 — a BAA "
                "does not substitute for individual authorisation for marketing uses."
            ),
            "source": "HHS_OCR_FAQ_ThirdPartySharing",
            "tier": 2,
        },
        # ── ADA (Tier 1) ─────────────────────────────────────────────────────
        {
            "chunk_id": "ada_title_i_medical_records",
            "text": (
                "Americans with Disabilities Act — Title I: Employers may not use "
                "medical information to make employment decisions. Medical records of "
                "employees must be kept separate from general personnel files and "
                "maintained in a confidential manner. Supervisors and managers may be "
                "informed about restrictions on the work or duties of the employee and "
                "necessary accommodations. First aid and safety personnel may be informed "
                "if the disability might require emergency treatment. Disability-related "
                "information must be kept confidential even after employment ends."
            ),
            "source": "ADA_TitleI_42USC12101",
            "tier": 1,
        },
        # ── HITECH ACT (Tier 1) ──────────────────────────────────────────────
        {
            "chunk_id": "hitech_13402_breach_notification",
            "text": (
                "HITECH Act Section 13402 — Notification in the case of breach. "
                "A covered entity that accesses, maintains, retains, modifies, records, "
                "stores, destroys, or otherwise holds, uses, or discloses unsecured PHI "
                "shall provide notification of any breach. HITECH extended breach "
                "notification requirements to business associates directly. HITECH also "
                "increased civil monetary penalties significantly: up to $1.5 million "
                "per violation category per year for willful neglect not corrected. "
                "Note: the 60-day notification clock starts from DISCOVERY of the "
                "breach, not from the breach itself."
            ),
            "source": "HITECH_Act_2009",
            "tier": 1,
        },
        # ── NIST HEALTHCARE FRAMEWORK (Tier 3) ───────────────────────────────
        {
            "chunk_id": "nist_sp800_66_hipaa_security",
            "text": (
                "NIST SP 800-66 Rev 2 — Implementing the HIPAA Security Rule. "
                "This publication provides guidance on implementing the HIPAA Security "
                "Rule. For encryption: NIST recommends AES-128 or AES-256 for data "
                "at rest, and TLS 1.2 or 1.3 for data in transit. However, this is "
                "NIST guidance, not a HIPAA mandate. The HIPAA Security Rule does not "
                "name specific encryption algorithms — entities must implement encryption "
                "or an equivalent alternative based on their risk assessment. "
                "The 'reasonable and appropriate' standard from 45 CFR 164.306 governs."
            ),
            "source": "NIST_SP800_66_Rev2",
            "tier": 3,
        },
        # ── GDPR (Tier 1) — for cross-regulation testing ─────────────────────
        {
            "chunk_id": "gdpr_art32_security_processing",
            "text": (
                "GDPR Article 32 — Security of processing. Taking into account the "
                "state of the art and the costs of implementation, controllers and "
                "processors shall implement appropriate technical and organisational "
                "measures, including as appropriate: (a) pseudonymisation and encryption "
                "of personal data; (b) ability to ensure ongoing confidentiality, "
                "integrity, availability and resilience. Article 32 does NOT mandate "
                "encryption — it is listed as one option among 'appropriate' measures. "
                "The controller must assess risk and choose proportionate measures."
            ),
            "source": "GDPR_2016_679",
            "tier": 1,
        },
        {
            "chunk_id": "gdpr_art83_penalties",
            "text": (
                "GDPR Article 83(5) — Infringements subject to fines up to 20,000,000 EUR "
                "or 4% of global annual turnover, whichever is higher. This applies to "
                "infringements of basic processing principles, data subject rights, and "
                "transfers to third countries. NOTE: the 4% fine applies specifically to "
                "the MOST SERIOUS infringements listed in Art 83(5). Lesser infringements "
                "under Art 83(4) attract fines up to 10,000,000 EUR or 2% of turnover. "
                "Not all GDPR violations attract the maximum penalty."
            ),
            "source": "GDPR_2016_679",
            "tier": 1,
        },
    ]
