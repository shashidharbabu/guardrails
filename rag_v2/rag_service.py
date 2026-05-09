#!/usr/bin/env python3
"""
rag_service.py
──────────────
Single RAG endpoint for three consumers:

  1. retrieve_for_llm(query, question_type)
       Called ONCE before the LLM generates its initial answer.
       Also used as CoT step 1 — initial retrieval is identical.
       If question_type="cross_doc", automatically retrieves from BOTH domains
       (healthcare + general_regulatory) so the LLM has context from both
       referenced documents instead of only the dominant one.
       Returns k=5 chunks + formatted context string.

  2. retrieve_for_cot(claim, original_query)
       Called PER CLAIM extracted from the baseline LLM answer.
       This is the step that makes CoT different from plain LLM:
         baseline_answer → extract N claims
         for each claim → retrieve_for_cot(claim) → 3 focused chunks
         LLM reasons over claim + chunks → SUPPORTED / UNSUPPORTED / UNCERTAIN
         repeat on contested claims for 2–3 rounds
       Same pattern as what Agent A does in CoD's first analysis round.

  3. retrieve_for_cod(query, session_id, agent_id, round_num, question_type)
       Called multiple times during a debate session:
         - Round 0 (shared)   : initial chunks for both agents — cross_doc aware
         - Round 1+ (agent_a/b): per-agent claim or challenge retrieval
         - Round 99 (judge)   : judge grounds its final verdict
       All calls tagged with session/agent/round for GRPO training logs.

Cross-doc retrieval (from evals_v2 fix):
  For cross_doc question_type, _retrieve_cross_doc() fetches top-3 from
  each domain separately then merges by rerank score → top-5.
  Prevents hallucination on questions like "How does HIPAA differ from GDPR?"
  where standard retrieval only surfaces one document.

All three methods call the same underlying hybrid retrieval pipeline
(dense Qwen3 + BM25 + BGE rerank, no HyDE).

Returns a consistent dict:
  {
    "chunks":   list[dict]   ← full chunk objects
    "context":  str          ← formatted string ready for LLM prompt
    "evidence": dict         ← {is_sufficient, top_rerank_score, cross_doc, ...}
    "meta":     dict         ← call metadata (task, agent_id, round, query)
  }

Usage:
  from rag_service import get_service
  svc = get_service()

  # 1. Plain LLM — single-doc or cross-doc aware
  r = svc.retrieve_for_llm("What is the HbA1c threshold?")
  r = svc.retrieve_for_llm("How does HIPAA differ from GDPR?", question_type="cross_doc")

  # 2. CoT — per extracted claim after baseline answer
  claims = ["HbA1c ≥ 6.5% is the ADA threshold", "Confirmed on two separate tests"]
  for claim in claims:
      r = svc.retrieve_for_cot(claim, original_query="What is the HbA1c threshold?")

  # 3. CoD — multiple calls across debate rounds
  r0   = svc.retrieve_for_cod(query, session_id="s1", agent_id="shared",  round_num=0)
  r_a2 = svc.retrieve_for_cod(claim, session_id="s1", agent_id="agent_a", round_num=2)
  r_j  = svc.retrieve_for_cod(query, session_id="s1", agent_id="judge",   round_num=99)
  # cross-doc initial:
  r0   = svc.retrieve_for_cod(query, session_id="s1", agent_id="shared",
                               round_num=0, question_type="cross_doc")
"""

import textwrap
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

import os

BASE_DIR = Path(os.environ.get("RAG_V2_BASE_DIR", Path(__file__).resolve().parent))
QDRANT_PATH = Path(os.environ.get("RAG_V2_QDRANT_PATH", str(BASE_DIR / "indexes_qdrant_data")))
COLLECTION_NAME = os.environ.get("RAG_V2_COLLECTION_NAME", "guardrails_rag_v2")
BM25_PATH = Path(os.environ.get("RAG_V2_BM25_PATH", str(BASE_DIR / "bm25_combined.pkl")))
OLD_CHUNKS_PATH = Path(os.environ.get("RAG_V2_OLD_CHUNKS_PATH", str(BASE_DIR / "data" / "enriched_chunks.json")))
HC_CHUNKS_PATH = Path(os.environ.get("RAG_V2_HC_CHUNKS_PATH", str(BASE_DIR / "data" / "healthcare_enriched_chunks.jsonl")))

# Chunk counts per task
K_LLM      = 5   # plain answer generation — broad context
K_COD_INIT = 5   # initial shared debate chunks
K_FOCUSED  = 3   # per-claim CoT + per-agent CoD + judge

# Cross-doc: how many chunks to pull from each domain before merging
K_CROSS_DOC_PER_DOMAIN = 3   # 3 healthcare + 3 regulatory → merge → top 5


# ── RAG Service ───────────────────────────────────────────────────────────────

class RAGService:
    """
    Loads retrieval models once. All three pipeline types share this instance.
    Use get_service() at module level to avoid reloading on every import.
    """

    def __init__(self, verbose: bool = False):
        try:
            from .retrieval_v2 import RetrievalPipeline
        except ImportError:
            from retrieval_v2 import RetrievalPipeline
        self.verbose  = verbose
        self.pipeline = RetrievalPipeline(
            qdrant_path=QDRANT_PATH,
            bm25_path=BM25_PATH,
            old_chunks_path=OLD_CHUNKS_PATH,
            healthcare_chunks_path=HC_CHUNKS_PATH,
            collection_name=COLLECTION_NAME,
        )
        if verbose:
            print(f"  ✓ RAGService ready — {COLLECTION_NAME} @ {QDRANT_PATH}")

    # ── Task 1: Plain LLM (also CoT step 1) ───────────────────────────────────

    def retrieve_for_llm(
        self,
        query: str,
        question_type: str = "",
        domain_filter: Optional[str] = None,
    ) -> dict:
        """
        Single retrieval call before the LLM generates its answer.
        Use the returned 'context' string directly in the LLM prompt.

        If question_type="cross_doc", retrieves from both domains to prevent
        the model hallucinating about the document not in the top results.

        Args:
            query:         The user's question.
            question_type: "cross_doc" triggers dual-domain retrieval.
                           Anything else (or empty) uses normal retrieval.
            domain_filter: Optional — "healthcare" | "general_regulatory".
                           Ignored when question_type="cross_doc".
        """
        if question_type == "cross_doc":
            chunks, evidence = self._retrieve_cross_doc(query)
        else:
            chunks, evidence = self._retrieve(query, k=K_LLM, domain_filter=domain_filter)

        return {
            "chunks":   chunks,
            "context":  _format_context(chunks),
            "evidence": evidence,
            "meta": {
                "task":          "llm",
                "query":         query,
                "question_type": question_type,
                "k":             len(chunks),
            },
        }

    # ── Task 2: CoT per-claim retrieval ───────────────────────────────────────

    def retrieve_for_cot(
        self,
        claim: str,
        original_query: str,
        domain_filter: Optional[str] = None,
    ) -> dict:
        """
        Called ONCE PER CLAIM extracted from the baseline LLM answer.

        Flow in cot_pipeline.py:
            # Step 1 — same as LLM
            init   = svc.retrieve_for_llm(query, question_type=qtype)
            answer = llm(query, init["context"])

            # Step 2 — extract claims, verify each one
            for claim in extract_claims(answer):
                r = svc.retrieve_for_cot(claim, original_query=query)
                verdict = llm_verify(claim, r["context"])
                # → SUPPORTED / UNSUPPORTED / UNCERTAIN + confidence

            # Step 3 — repeat on uncertain/unsupported claims (rounds 2–3)

        Why claim + original_query:
            A bare claim like "HbA1c ≥ 6.5%" retrieves narrowly.
            Adding the original question surfaces the right document section.
            Same trick Agent A uses mid-debate.

        No cross-doc handling here — claims are already extracted from a
        baseline answer, so they're focused single-statement queries.
        """
        combined = f"{claim}. Context: {original_query}"
        chunks, evidence = self._retrieve(combined, k=K_FOCUSED, domain_filter=domain_filter)

        return {
            "chunks":   chunks,
            "context":  _format_context(chunks),
            "evidence": evidence,
            "meta": {
                "task":           "cot",
                "claim":          claim,
                "original_query": original_query,
                "k":              K_FOCUSED,
            },
        }

    # ── Task 3: CoD multi-agent debate ────────────────────────────────────────

    def retrieve_for_cod(
        self,
        query: str,
        session_id: str,
        agent_id: str,
        round_num: int,
        question_type: str = "",
        k: Optional[int] = None,
        domain_filter: Optional[str] = None,
    ) -> dict:
        """
        Called multiple times during a debate session. Each call tagged with
        session/agent/round for GRPO training data.

        Call pattern:
          round_num=0  | agent_id="shared"  → initial chunks for both agents
                                               cross_doc aware (question_type param)
          round_num=1+ | agent_id="agent_a" → Agent A retrieves for a specific claim
          round_num=1+ | agent_id="agent_b" → Agent B retrieves for a challenge
          round_num=99 | agent_id="judge"   → Judge grounds its final verdict

        Args:
            query:         Round 0: original question. Round 1+: claim/challenge text.
            session_id:    Unique debate session ID (for log grouping).
            agent_id:      "shared" | "agent_a" | "agent_b" | "judge"
            round_num:     0=initial, 1/2/3=debate rounds, 99=judge
            question_type: "cross_doc" on round 0 triggers dual-domain retrieval.
            k:             Override chunk count. Auto-defaults by agent_id if None.
            domain_filter: Optional domain restriction (ignored for cross_doc).
        """
        # Round 0 shared: cross-doc aware, broad context
        if round_num == 0 and agent_id == "shared" and question_type == "cross_doc":
            chunks, evidence = self._retrieve_cross_doc(query)
        else:
            if k is None:
                k = K_COD_INIT if (agent_id == "shared") else K_FOCUSED
            chunks, evidence = self._retrieve(query, k=k, domain_filter=domain_filter)

        return {
            "chunks":   chunks,
            "context":  _format_context(chunks),
            "evidence": evidence,
            "meta": {
                "task":          "cod",
                "session_id":    session_id,
                "agent_id":      agent_id,
                "round_num":     round_num,
                "question_type": question_type,
                "query":         query,
                "k":             len(chunks),
            },
        }

    # ── Core retrieval ────────────────────────────────────────────────────────

    def _retrieve(
        self,
        query: str,
        k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> tuple[list[dict], dict]:
        """Single-domain retrieval. Returns (chunks[:k], evidence)."""
        raw      = self.pipeline.retrieve(
            query         = query,
            use_hyde      = False,
            verbose       = self.verbose,
            domain_filter = domain_filter,
        )
        chunks   = raw.get("chunks", [])[:k]
        evidence = raw.get("evidence", {})
        return chunks, evidence

    def _retrieve_cross_doc(self, query: str) -> tuple[list[dict], dict]:
        """
        Dual-domain retrieval for cross-doc questions.
        Fetches top-3 from each domain independently then merges by rerank score.

        Why this matters:
          "How does HIPAA differ from GDPR?" — HIPAA is in healthcare domain,
          GDPR is in general_regulatory. Standard retrieval surfaces only the
          dominant match. This ensures both documents appear in context so the
          model doesn't infer the missing one from parametric memory.
        """
        hc_chunks,  ev_hc  = self._retrieve(query, k=K_CROSS_DOC_PER_DOMAIN,
                                             domain_filter="healthcare")
        reg_chunks, ev_reg = self._retrieve(query, k=K_CROSS_DOC_PER_DOMAIN,
                                            domain_filter="general_regulatory")

        # Merge, deduplicate by (source_file, text prefix), sort by rerank score
        seen, merged = set(), []
        for c in hc_chunks + reg_chunks:
            key = (c.get("source_file", ""), c.get("text", "")[:80])
            if key not in seen:
                seen.add(key)
                merged.append(c)

        merged.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        merged = merged[:5]
        for i, c in enumerate(merged, 1):
            c["rank"] = i

        evidence = {
            "is_sufficient":    ev_hc.get("is_sufficient") or ev_reg.get("is_sufficient"),
            "top_rerank_score": max(ev_hc.get("top_rerank_score", 0),
                                    ev_reg.get("top_rerank_score", 0)),
            "unique_sources":   len({c.get("source_file") for c in merged}),
            "cross_doc":        True,
        }
        return merged, evidence


# ── Context formatter ─────────────────────────────────────────────────────────

def _format_context(chunks: list[dict], max_chars: int = 800) -> str:
    """
    Formats retrieved chunks into a numbered context block for LLM prompts.

    [1] Source: ADA (standards-of-care-2024).pdf  [healthcare]  (relevance: 0.842)
        The recommended HbA1c target for most non-pregnant adults is <7% ...

    [2] Source: t1__gdpr_2016_679.pdf  [general_regulatory]  (relevance: 0.731)
        Article 22 permits automated decision-making where it is necessary for ...
    """
    if not chunks:
        return "No relevant documents retrieved."

    parts = []
    for c in chunks:
        source = c.get("source_file", "Unknown source")
        text   = c.get("text", "").strip()
        domain = c.get("domain", "")
        score  = c.get("rerank_score", 0.0)
        rank   = c.get("rank", "?")

        if len(text) > max_chars:
            text = text[:max_chars] + "…"

        domain_tag = f"  [{domain}]" if domain else ""
        header     = f"[{rank}] Source: {source}{domain_tag}  (relevance: {score:.3f})"
        body       = textwrap.indent(text, "    ")
        parts.append(f"{header}\n{body}")

    return "\n\n".join(parts)


# ── Module-level singleton ────────────────────────────────────────────────────

_service: Optional["RAGService"] = None

def get_service(verbose: bool = False) -> "RAGService":
    """
    Returns (and caches) the module-level RAGService singleton.
    Models load once on first call (~15s GPU, ~45s CPU).
    All three pipelines share this instance — no re-loading.

    from rag_service import get_service
    svc = get_service()
    """
    global _service
    if _service is None:
        _service = RAGService(verbose=verbose)
    return _service


# ── Prompts (imported by cot_pipeline.py and cod_pipeline.py) ─────────────────

# ── LLM ───────────────────────────────────────────────────────────────────────

LLM_SYSTEM = """\
You are a regulatory and clinical compliance expert.
Answer the question using ONLY the retrieved document chunks provided.
Be specific — cite exact thresholds, article numbers, and regulatory language.
If the chunks are insufficient, say so clearly. Do not invent facts.
If the question asks you to compare two specific documents but you only have
chunks from one of them, explicitly state which document is missing from your
context. Do not describe or infer the missing document from memory.\
"""

def build_llm_prompt(query: str, context: str, question_type: str = "") -> str:
    cross_doc_note = (
        "\nNote: this is a cross-document question. Chunks from both referenced "
        "documents have been retrieved where available. If one document is absent, "
        "state that explicitly — do not infer it from memory.\n"
    ) if question_type == "cross_doc" else ""
    return (
        f"Retrieved chunks:\n{context}\n"
        f"{cross_doc_note}\n"
        f"Question: {query}\n\nAnswer:"
    )


# ── CoT ───────────────────────────────────────────────────────────────────────

COT_ROUND1_SYSTEM = """\
You are a regulatory and clinical compliance expert.
Answer the question using ONLY the retrieved document chunks.
Be specific and cite sources. At the end, state your overall confidence (0.0–1.0).\
"""

COT_ROUND2_SYSTEM = """\
You are verifying your own answer to a regulatory compliance question.
For each factual claim you made, retrieve and reason over the provided chunks:
  - SUPPORTED   : directly stated in the chunks (quote the evidence)
  - UNSUPPORTED : not found in the chunks — flag for removal
  - UNCERTAIN   : partially supported — note what is missing
Output a structured claim-by-claim verdict. Do not rewrite the answer yet.\
"""

COT_ROUND3_SYSTEM = """\
You are revising your answer based on your own claim verification.
Using the chunks, your original answer, and the verification verdicts:
  1. Keep only SUPPORTED claims
  2. Remove or soften UNSUPPORTED claims
  3. Note UNCERTAIN claims with "I am not certain about this"
  4. State a final confidence score (0.0–1.0) for the whole answer\
"""

def build_cot_round1(query: str, context: str, question_type: str = "") -> str:
    cross_doc_note = (
        "\nNote: cross-document question — chunks from both documents retrieved "
        "where available. State if one is missing rather than inferring it.\n"
    ) if question_type == "cross_doc" else ""
    return (
        f"Retrieved chunks:\n{context}\n{cross_doc_note}\n"
        f"Question: {query}\n\n"
        f"Answer (state confidence 0.0–1.0 at the end):"
    )

def build_cot_round2(
    query: str, claim_chunks: dict[str, str], baseline_answer: str
) -> str:
    """
    claim_chunks: {"claim text": "formatted context for that claim", ...}
    """
    claims_section = "\n\n".join(
        f"Claim: {claim}\nChunks for this claim:\n{ctx}"
        for claim, ctx in claim_chunks.items()
    )
    return (
        f"Original question: {query}\n\n"
        f"Your baseline answer:\n{baseline_answer}\n\n"
        f"Per-claim evidence:\n{claims_section}\n\n"
        f"Verdict for each claim (SUPPORTED / UNSUPPORTED / UNCERTAIN + evidence quote):"
    )

def build_cot_round3(
    query: str, context: str, baseline_answer: str, verdicts: str
) -> str:
    return (
        f"Retrieved chunks:\n{context}\n\n"
        f"Original question: {query}\n\n"
        f"Baseline answer:\n{baseline_answer}\n\n"
        f"Claim verdicts:\n{verdicts}\n\n"
        f"Revised final answer (confidence score + uncertainty flags):"
    )


# ── CoD ───────────────────────────────────────────────────────────────────────

COD_AGENT_A_SYSTEM = """\
You are Agent A in a regulatory compliance debate.
Using the retrieved document chunks and the question, state your position clearly.
For each factual claim: claim | confidence (0.0–1.0) | source chunk reference\
"""

COD_AGENT_B_SYSTEM = """\
You are Agent B in a regulatory compliance debate.
You have seen the same document chunks as Agent A.
State your own independent position — do not react to Agent A yet.
For each factual claim: claim | confidence (0.0–1.0) | source chunk reference\
"""

COD_REBUTTAL_A_SYSTEM = """\
You are Agent A reviewing Agent B's position.
For each of Agent B's claims:
  - Agree and merge if supported by chunks
  - Challenge with specific chunk evidence if you disagree
  - Acknowledge uncertainty if chunks are insufficient
Update your confidence scores.\
"""

COD_REBUTTAL_B_SYSTEM = """\
You are Agent B reviewing Agent A's position.
For each of Agent A's claims:
  - Agree and merge if supported by chunks
  - Challenge with specific chunk evidence if you disagree
  - Acknowledge uncertainty if chunks are insufficient
Update your confidence scores.\
"""

COD_JUDGE_SYSTEM = """\
You are an impartial judge evaluating a regulatory compliance debate.
You have the original document chunks and the full debate transcript.
For each contested claim, rule: AGENT_A | AGENT_B | BOTH_CORRECT | UNCERTAIN
Cite the specific chunk that settles it. State confidence (0.0–1.0).
Produce a final answer that states verified facts, notes exceptions, and
marks anything uncertain. Cite source documents by name.\
"""

def build_cod_initial(query: str, context: str, question_type: str = "") -> str:
    cross_doc_note = (
        "\nNote: cross-document question — chunks from both referenced documents "
        "included where available.\n"
    ) if question_type == "cross_doc" else ""
    return (
        f"Retrieved chunks:\n{context}\n{cross_doc_note}\n"
        f"Question: {query}\n\n"
        f"State your position (claim | confidence | chunk ref):"
    )

def build_cod_rebuttal(
    query: str,
    context: str,
    own_position: str,
    other_position: str,
    own_label: str = "Your previous position",
    other_label: str = "Opposing agent's position",
) -> str:
    return (
        f"Retrieved chunks:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"{own_label}:\n{own_position}\n\n"
        f"{other_label}:\n{other_position}\n\n"
        f"Revised position after considering the opposing view:"
    )

def build_cod_judge(
    query: str,
    context: str,
    agent_a_final: str,
    agent_b_final: str,
    debate_rounds: int,
) -> str:
    return (
        f"Retrieved chunks:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Debate ({debate_rounds} rounds):\n"
        f"Agent A:\n{agent_a_final}\n\n"
        f"Agent B:\n{agent_b_final}\n\n"
        f"Judge verdict (per-claim ruling + final synthesised answer):"
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  RAG SERVICE — smoke test")
    print("=" * 62)

    svc = RAGService(verbose=True)

    q_single     = "What is the HbA1c threshold for diagnosing diabetes per ADA?"
    q_cross      = "How does HIPAA's breach notification differ from GDPR's personal data breach requirements?"
    claim_sample = "HbA1c ≥ 6.5% confirmed on two separate tests is the ADA diagnostic criterion"

    print(f"\n[Task 1a] retrieve_for_llm — single doc")
    r = svc.retrieve_for_llm(q_single)
    print(f"  chunks={len(r['chunks'])}  cross_doc={r['evidence'].get('cross_doc', False)}")
    print(f"  sources: {[c.get('source_file','?')[:40] for c in r['chunks']]}")

    print(f"\n[Task 1b] retrieve_for_llm — cross_doc")
    r = svc.retrieve_for_llm(q_cross, question_type="cross_doc")
    print(f"  chunks={len(r['chunks'])}  cross_doc={r['evidence'].get('cross_doc', False)}")
    print(f"  sources: {[c.get('source_file','?')[:40] for c in r['chunks']]}")
    print(f"  unique_sources={r['evidence'].get('unique_sources')}")

    print(f"\n[Task 2] retrieve_for_cot — per claim")
    r = svc.retrieve_for_cot(claim_sample, original_query=q_single)
    print(f"  chunks={len(r['chunks'])}  k={r['meta']['k']}")
    print(f"  meta={r['meta']}")

    print(f"\n[Task 3a] retrieve_for_cod — round 0 shared (normal)")
    r = svc.retrieve_for_cod(q_single, session_id="s1", agent_id="shared", round_num=0)
    print(f"  chunks={len(r['chunks'])}  meta={r['meta']}")

    print(f"\n[Task 3b] retrieve_for_cod — round 0 shared (cross_doc)")
    r = svc.retrieve_for_cod(q_cross, session_id="s2", agent_id="shared",
                              round_num=0, question_type="cross_doc")
    print(f"  chunks={len(r['chunks'])}  unique_sources={r['evidence'].get('unique_sources')}")

    print(f"\n[Task 3c] retrieve_for_cod — round 2 agent_a (claim)")
    r = svc.retrieve_for_cod(claim_sample, session_id="s1", agent_id="agent_a", round_num=2)
    print(f"  chunks={len(r['chunks'])}  k={r['meta']['k']}")

    print(f"\n[Task 3d] retrieve_for_cod — judge (round 99)")
    r = svc.retrieve_for_cod(q_single, session_id="s1", agent_id="judge", round_num=99)
    print(f"  chunks={len(r['chunks'])}  k={r['meta']['k']}")

    print("\n✓ RAGService smoke test complete.")
