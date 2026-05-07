"""
rag/eval/queries.py — Evaluation query set for the RAG pipeline.
=================================================================

20 regulatory domain queries drawn from the corpus:
  - HIPAA (privacy, security, breach)
  - GDPR (consent, rights, fines)
  - ISO 27001 (ISMS requirements)
  - NIST (frameworks, encryption guidance)
  - CCPA (consumer rights, deletion)
  - OWASP (LLM / AI security risks)
  - EU AI Act / AI governance
  - HITECH Act

All queries are realistic operational questions that an enterprise user
would ask their compliance LLM. No ground-truth answers are provided
because RAGAS will use reference-free metrics (Faithfulness, AnswerRelevancy,
ContextPrecision) that do not require reference answers.

Schema per entry:
    {
        "id":     str           — unique identifier
        "domain": str           — regulatory domain tag
        "query":  str           — the question text
    }
"""
from __future__ import annotations

EVAL_QUERIES: list[dict] = [
    # ── HIPAA ──────────────────────────────────────────────────────────────────
    {
        "id":     "hipaa_01",
        "domain": "HIPAA",
        "query":  "Does HIPAA require covered entities to use AES-256 encryption for ePHI at rest?",
    },
    {
        "id":     "hipaa_02",
        "domain": "HIPAA",
        "query":  "How quickly must a covered entity notify patients after discovering a PHI breach?",
    },
    {
        "id":     "hipaa_03",
        "domain": "HIPAA",
        "query":  "Can a hospital share patient records with a billing company without a Business Associate Agreement?",
    },
    {
        "id":     "hipaa_04",
        "domain": "HIPAA",
        "query":  "What is the minimum necessary standard under HIPAA and when does it not apply?",
    },
    {
        "id":     "hipaa_05",
        "domain": "HIPAA",
        "query":  "Under what circumstances may a covered entity deny a patient's request to access their medical records?",
    },
    # ── GDPR ───────────────────────────────────────────────────────────────────
    {
        "id":     "gdpr_01",
        "domain": "GDPR",
        "query":  "Does GDPR Article 32 mandate encryption of personal data at rest?",
    },
    {
        "id":     "gdpr_02",
        "domain": "GDPR",
        "query":  "What is the maximum fine for the most serious GDPR violations and what triggers that fine level?",
    },
    {
        "id":     "gdpr_03",
        "domain": "GDPR",
        "query":  "What lawful basis is required under GDPR to process personal data for AI model training?",
    },
    {
        "id":     "gdpr_04",
        "domain": "GDPR",
        "query":  "What are the requirements for conducting a Data Protection Impact Assessment (DPIA) for AI systems?",
    },
    # ── ISO 27001 ───────────────────────────────────────────────────────────────
    {
        "id":     "iso_01",
        "domain": "ISO 27001",
        "query":  "What are the core requirements for establishing an information security management system under ISO 27001?",
    },
    {
        "id":     "iso_02",
        "domain": "ISO 27001",
        "query":  "How does ISO 27001 require organizations to handle changes to their information security management system?",
    },
    # ── NIST ────────────────────────────────────────────────────────────────────
    {
        "id":     "nist_01",
        "domain": "NIST",
        "query":  "What encryption standards does NIST SP 800-66 recommend for protecting ePHI?",
    },
    {
        "id":     "nist_02",
        "domain": "NIST",
        "query":  "What does the NIST Cybersecurity Framework 2.0 require for organizational governance of security risks?",
    },
    {
        "id":     "nist_03",
        "domain": "NIST",
        "query":  "How does NIST SSDF SP 800-218 address third-party software supply chain security requirements?",
    },
    # ── CCPA ────────────────────────────────────────────────────────────────────
    {
        "id":     "ccpa_01",
        "domain": "CCPA",
        "query":  "What security standard does CCPA require businesses to apply to California consumer personal information?",
    },
    # ── OWASP ───────────────────────────────────────────────────────────────────
    {
        "id":     "owasp_01",
        "domain": "OWASP",
        "query":  "What are the OWASP Top 10 risks specific to large language model applications?",
    },
    {
        "id":     "owasp_02",
        "domain": "OWASP",
        "query":  "How does OWASP define prompt injection attacks on AI agents and what mitigations are recommended?",
    },
    # ── HITECH ──────────────────────────────────────────────────────────────────
    {
        "id":     "hitech_01",
        "domain": "HITECH",
        "query":  "What civil monetary penalties does HITECH impose for willful neglect of HIPAA rules?",
    },
    # ── AI Governance ────────────────────────────────────────────────────────────
    {
        "id":     "ai_gov_01",
        "domain": "AI Governance",
        "query":  "What accountability and governance requirements apply to organizations using AI systems under the ICO AI auditing framework?",
    },
    {
        "id":     "ai_gov_02",
        "domain": "AI Governance",
        "query":  "How must organizations document and demonstrate compliance when deploying AI systems that process personal data?",
    },
]


def get_queries() -> list[dict]:
    """Return the full evaluation query list."""
    return EVAL_QUERIES


def get_queries_by_domain(domain: str) -> list[dict]:
    """Return queries filtered by domain (case-insensitive prefix match)."""
    d = domain.lower()
    return [q for q in EVAL_QUERIES if q["domain"].lower().startswith(d)]


if __name__ == "__main__":
    print(f"Total evaluation queries: {len(EVAL_QUERIES)}")
    domains: dict[str, int] = {}
    for q in EVAL_QUERIES:
        domains[q["domain"]] = domains.get(q["domain"], 0) + 1
    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count}")
