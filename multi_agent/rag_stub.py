"""
rag_stub.py — Stub RAG retriever backed by your local JSONL chunk file.

Interface contract (identical to what the real Qdrant retriever will expose):
    retrieve(query: str, top_k: int) -> List[EvidenceChunk]

To swap in real Qdrant later, replace ONLY the body of retrieve() with:
    vectors = embed_model.encode(query)
    hits = qdrant_client.search(collection_name=COLLECTION, query_vector=vectors, limit=top_k)
    return [EvidenceChunk(chunk_id=h.id, text=h.payload["text"], ...) for h in hits]

Everything else (agent_a, agent_b, judge) calls retrieve() unchanged.
"""
from __future__ import annotations

import json
import re
import math
from pathlib import Path
from typing import List, Dict

from multi_agent.models import EvidenceChunk
from multi_agent.config import CHUNKS_JSONL_PATH, TOP_K_CHUNKS

# ── Module-level chunk cache (loaded once) ────────────────────────────────────
_chunk_cache: List[Dict] = []
_loaded: bool = False


def _load_chunks() -> List[Dict]:
    global _chunk_cache, _loaded
    if _loaded:
        return _chunk_cache

    path = Path(CHUNKS_JSONL_PATH)
    if path.exists():
        print(f"[RAG stub] Loading chunks from {path}")
        with open(path, encoding="utf-8") as f:
            _chunk_cache = [json.loads(line) for line in f if line.strip()]
        print(f"[RAG stub] Loaded {len(_chunk_cache)} chunks from JSONL")
    else:
        print(f"[RAG stub] JSONL not found at {path} — using built-in regulatory sample chunks")
        _chunk_cache = _sample_chunks()

    _loaded = True
    return _chunk_cache


def retrieve(query: str, top_k: int = TOP_K_CHUNKS) -> List[EvidenceChunk]:
    """
    Keyword-based TF-IDF-lite retrieval over loaded chunks.
    Swap body here when connecting real Qdrant.
    """
    chunks = _load_chunks()
    if not chunks:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        # Return first top_k chunks as fallback
        return [_to_evidence(c, i) for i, c in enumerate(chunks[:top_k])]

    # Compute TF-IDF-lite scores
    idf = _compute_idf(chunks, query_terms)
    scored: List[tuple[float, EvidenceChunk]] = []

    for i, chunk in enumerate(chunks):
        text = _get_text(chunk)
        chunk_terms = _tokenize(text)
        if not chunk_terms:
            continue

        # TF for each query term in this chunk
        tf_sum = 0.0
        for term in query_terms:
            tf = chunk_terms.count(term) / len(chunk_terms)
            tf_sum += tf * idf.get(term, 0.0)

        if tf_sum > 0:
            scored.append((tf_sum, _to_evidence(chunk, i)))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [ec for _, ec in scored[:top_k]]

    # If nothing matched at all, return top_k anyway (stub behaviour)
    if not results:
        results = [_to_evidence(c, i) for i, c in enumerate(chunks[:top_k])]

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        source=str(chunk.get("source", chunk.get("doc_id", chunk.get("filename", f"doc_{idx}")))),
        tier=int(chunk.get("tier", chunk.get("authority_tier", 1))),
    )


def _compute_idf(chunks: List[Dict], query_terms: List[str]) -> Dict[str, float]:
    N = len(chunks)
    idf: Dict[str, float] = {}
    for term in set(query_terms):
        df = sum(1 for c in chunks if term in _tokenize(_get_text(c)))
        idf[term] = math.log((N + 1) / (df + 1)) + 1.0
    return idf


# ── Built-in fallback sample chunks ──────────────────────────────────────────

def _sample_chunks() -> List[Dict]:
    """
    Realistic GDPR/HIPAA/NIST sample chunks for testing without a JSONL file.
    Covers the example query: "Does GDPR require us to encrypt data at rest?"
    """
    return [
        {
            "chunk_id": "gdpr_art32_p1",
            "text": (
                "GDPR Article 32 — Security of processing. Taking into account the state of the art, "
                "the costs of implementation and the nature, scope, context and purposes of processing "
                "as well as the risk of varying likelihood and severity for the rights and freedoms of "
                "natural persons, the controller and the processor shall implement appropriate technical "
                "and organisational measures to ensure a level of security appropriate to the risk, "
                "including as appropriate: (a) the pseudonymisation and encryption of personal data; "
                "(b) the ability to ensure the ongoing confidentiality, integrity, availability and "
                "resilience of processing systems and services."
            ),
            "source": "GDPR_2016_679",
            "tier": 1,
        },
        {
            "chunk_id": "gdpr_art83_p5",
            "text": (
                "GDPR Article 83(5) — Infringements of the following provisions shall be subject to "
                "administrative fines up to 20,000,000 EUR, or in the case of an undertaking, up to "
                "4% of the total worldwide annual turnover of the preceding financial year, whichever "
                "is higher: (a) the basic principles for processing, including conditions for consent; "
                "(b) the data subjects' rights; (c) the transfers of personal data to a recipient in "
                "a third country or an international organisation. These fines apply to the most serious "
                "infringements — not to all non-compliance."
            ),
            "source": "GDPR_2016_679",
            "tier": 1,
        },
        {
            "chunk_id": "edpb_guidelines_2024_encryption",
            "text": (
                "EDPB Guidelines 2024 on Article 32: The EDPB clarifies that Article 32 GDPR does NOT "
                "mandate any specific technical measure such as AES-256. The controller must assess "
                "the risks and apply appropriate safeguards. Encryption is one recommended measure but "
                "not the only acceptable one. Pseudonymisation is explicitly listed as a valid alternative. "
                "The choice of measure must be proportionate to the risk. AES-256 is referenced in NIST "
                "guidelines as a best practice but is NOT a GDPR legal requirement."
            ),
            "source": "EDPB_Guidelines_2024",
            "tier": 2,
        },
        {
            "chunk_id": "gdpr_art32_p3_exceptions",
            "text": (
                "GDPR Article 32(3) — Adherence to an approved code of conduct as referred to in "
                "Article 40 or an approved certification mechanism as referred to in Article 42 may "
                "be used as an element to demonstrate compliance with the requirements set out in "
                "paragraph 1. This means alternative compliance mechanisms exist beyond direct "
                "technical encryption."
            ),
            "source": "GDPR_2016_679",
            "tier": 1,
        },
        {
            "chunk_id": "nist_sp800_111_encryption",
            "text": (
                "NIST SP 800-111 — Guide to Storage Encryption Technologies for End User Devices. "
                "AES with 128-bit or 256-bit keys is recommended for storage encryption. AES-256-GCM "
                "is the preferred mode. However, this is a NIST recommendation, not a legal mandate. "
                "Organisations choosing not to use AES-256 must justify the alternative's adequacy."
            ),
            "source": "NIST_SP800_111",
            "tier": 2,
        },
        {
            "chunk_id": "hipaa_164_312_technical",
            "text": (
                "HIPAA Security Rule 45 CFR 164.312(a)(2)(iv) — Encryption and Decryption: "
                "Implement a mechanism to encrypt and decrypt electronic protected health information (ePHI). "
                "This is an ADDRESSABLE implementation specification, meaning covered entities must assess "
                "whether it is reasonable and appropriate to implement and, if not, implement an equivalent "
                "alternative measure."
            ),
            "source": "HIPAA_Security_Rule_45CFR",
            "tier": 1,
        },
        {
            "chunk_id": "gdpr_art4_definitions",
            "text": (
                "GDPR Article 4 — Definitions. 'Personal data' means any information relating to an "
                "identified or identifiable natural person ('data subject'). 'Processing' means any "
                "operation or set of operations which is performed on personal data, whether or not "
                "by automated means, such as collection, recording, organisation, structuring, storage, "
                "adaptation, retrieval, consultation, use, disclosure by transmission, dissemination."
            ),
            "source": "GDPR_2016_679",
            "tier": 1,
        },
        {
            "chunk_id": "ccpa_1798_81_5_security",
            "text": (
                "CCPA Section 1798.81.5 — A business that owns, licenses, or maintains personal "
                "information about a California resident shall implement and maintain reasonable "
                "security procedures and practices appropriate to the nature of the information to "
                "protect the personal information from unauthorized access, destruction, use, "
                "modification, or disclosure. The CCPA does not specify encryption standards."
            ),
            "source": "CCPA_2018",
            "tier": 1,
        },
        {
            "chunk_id": "owasp_cryptographic_failures",
            "text": (
                "OWASP Top 10 2021 — A02 Cryptographic Failures (formerly Sensitive Data Exposure). "
                "Encrypt all data at rest. Use strong, up-to-date algorithms: AES-256-GCM, "
                "ChaCha20-Poly1305. Avoid deprecated algorithms: DES, 3DES, RC4, MD5 for hashing. "
                "This is a security best practice framework, not a regulatory mandate."
            ),
            "source": "OWASP_Top10_2021",
            "tier": 3,
        },
        {
            "chunk_id": "gdpr_recital_83_risk_assessment",
            "text": (
                "GDPR Recital 83 — In order to maintain security and to prevent processing in "
                "infringement of this Regulation, the controller or processor should evaluate the "
                "risks inherent in the processing and implement measures to mitigate those risks, "
                "such as encryption. Those measures should ensure an appropriate level of security, "
                "including confidentiality, taking into account the state of the art and the costs "
                "of implementation in relation to the risks and the nature of the personal data to "
                "be protected."
            ),
            "source": "GDPR_2016_679_Recitals",
            "tier": 1,
        },
    ]
