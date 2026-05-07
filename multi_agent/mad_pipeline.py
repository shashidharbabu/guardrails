"""
mad_pipeline.py — Top-level MAD pipeline orchestrator (loads .env from repo root)
======================================================

ENTRY POINT
-----------
run_mad(query, llm_answer) → MADOutput

This is the single function the gateway (or API) calls. It orchestrates
the full pipeline and handles storage at the start and end.

COMPLETE FLOW
-------------
1.  Generate rollout_id (UUID) — needed for GRPO multi-rollout comparison
2.  Extract atomic claims from LLM answer
3.  [STORAGE] Write query row (TABLE 1)
4.  Run debate engine (which writes claims + attacks rows internally)
5.  Run Judge evaluation (partially blind — confidence scores stripped)
6.  [STORAGE] Write judge_verdicts (TABLE 4)
7.  Run full CSE (DeepEval F/H/Relevancy + MAD judge aggregate)
8.  [STORAGE] Update query with final CSE score, component scores, routing decision
9.  Build transcript
10. Return MADOutput

ROUTING LOGIC
-------------
Hard rule (checked FIRST, overrides everything):
  Any is_material claim with judge score = 0.0 → HARD_BLOCK → human review
  A fabricated regulatory standard (e.g. "HIPAA mandates AES-256") that
  scores 0.0 cannot reach the user via any retry path.

Soft rules (after hard rule passes):
  aggregate > 0.8 → DELIVER
  aggregate 0.4–0.8 → RETRY (send correction signal to LLM)
  aggregate < 0.4 → HUMAN_REVIEW

CSE FORMULA (v2.0 — full 4-component):
  final = 0.30*F_llm + 0.25*(1-H_llm) + 0.10*relevancy + 0.35*judge_eval_score

  F_llm         — FaithfulnessMetric    (DeepEval + Ollama qwen2.5:7b)
  H_llm         — HallucinationMetric   (DeepEval + Ollama qwen2.5:7b)
  relevancy     — ContextualRelevancy   (DeepEval + Ollama qwen2.5:7b)
  judge_eval    — MAD judge aggregate   (min material 70% + mean non-material 30%)

  Falls back to v0.1 (judge-only) if DeepEval / Ollama is unavailable.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

# Add repo root to sys.path so `confidence` package is importable from here
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from multi_agent import storage
from multi_agent.claim_extractor import extract_claims
from multi_agent.config import (
    MAX_CYCLES,
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_LOW,
)
from multi_agent.debate_engine import run_debate, build_transcript
from multi_agent.judge import judge_claims
from multi_agent.models import Claim, JudgeVerdict, MADOutput


def run_mad(query: str, llm_answer: str) -> MADOutput:
    """
    Full MAD pipeline with complete storage integration.

    Every intermediate result is written to SQLite so the GRPO
    feedback loop can consume it after the run completes.
    """
    # ── 1. GENERATE ROLLOUT_ID ─────────────────────────────────────────────────
    rollout_id = str(uuid.uuid4())
    query_id   = str(uuid.uuid4())

    _banner("MAD PIPELINE")
    print(f"  query_id  : {query_id}")
    print(f"  rollout_id: {rollout_id}")
    print(f"  Query     : {query}")
    print(f"  Answer    : {llm_answer[:120]}{'...' if len(llm_answer) > 120 else ''}")

    # ── 2. INITIALISE STORAGE ──────────────────────────────────────────────────
    storage.init_db()

    # ── 3. EXTRACT CLAIMS ──────────────────────────────────────────────────────
    print("\n[Step 1/4] Extracting atomic claims...")
    claims  = extract_claims(query, llm_answer)
    n_mat   = sum(1 for c in claims if c.is_material)
    print(f"           {len(claims)} claims ({n_mat} material, "
          f"{len(claims)-n_mat} contextual)")

    # ── 4. WRITE QUERY ROW (TABLE 1) ───────────────────────────────────────────
    storage.write_query(
        query_id=query_id,
        rollout_id=rollout_id,
        query_text=query,
        llm_answer=llm_answer,
        chunk_ids=[],
    )

    # ── 5. RUN DEBATE ──────────────────────────────────────────────────────────
    print(f"\n[Step 2/4] Running {MAX_CYCLES}-cycle debate...")
    cycles, final_claims, evidence_pool = run_debate(
        query=query,
        claims=claims,
        query_id=query_id,
        rollout_id=rollout_id,
        max_cycles=MAX_CYCLES,
    )

    # ── 6. JUDGE EVALUATION ────────────────────────────────────────────────────
    print("\n[Step 3/4] Judge evaluation (partially blind)...")
    judge_verdicts, correction_signal = judge_claims(
        query=query,
        final_claims=final_claims,
        evidence_pool=evidence_pool,
    )
    _print_judge_verdicts(judge_verdicts)

    # ── 7. WRITE JUDGE VERDICTS (TABLE 4) ─────────────────────────────────────
    pool_ids = [c.chunk_id for c in evidence_pool]
    storage.write_judge_verdicts(
        query_id=query_id,
        rollout_id=rollout_id,
        judge_verdicts=judge_verdicts,
        evidence_pool_ids=pool_ids,
    )

    # ── 8. RUN CONFIDENCE SCORING ENGINE ──────────────────────────────────────
    print("\n[Step 4/4] Confidence Scoring Engine (CSE)...")
    try:
        from confidence.scorer import ConfidenceScorer
        cse_result = ConfidenceScorer().score(
            query=query,
            llm_answer=llm_answer,
            rag_chunks=list(evidence_pool),
            final_claims=final_claims,
            judge_verdicts=judge_verdicts,
        )
        routing   = cse_result.routing_decision
        aggregate = cse_result.final_score
        _print_cse_result(cse_result)
    except Exception as exc:
        print(f"  ⚠  CSE import failed ({exc}) — falling back to judge-only routing")
        routing, aggregate = _compute_routing(final_claims, judge_verdicts)
        cse_result = None

    if correction_signal:
        print(f"           Correction signal   : {correction_signal[:90]}...")

    # ── 9. UPDATE QUERY WITH FINAL RESULTS (TABLE 1) ──────────────────────────
    storage.update_query_cse(
        query_id=query_id,
        rollout_id=rollout_id,
        cse_score=aggregate,
        routing_decision=routing,
        cse_components=cse_result.components.as_dict() if (cse_result and cse_result.components) else None,
        cse_version=cse_result.version if cse_result else "v0.1",
        cse_breakdown=cse_result.as_dict() if cse_result else None,
    )

    # ── 10. BUILD TRANSCRIPT + RETURN ─────────────────────────────────────────
    transcript = build_transcript(
        query=query,
        llm_answer=llm_answer,
        cycles=cycles,
        judge_verdicts=judge_verdicts,
        correction_signal=correction_signal or "",
    )

    return MADOutput(
        query=query,
        llm_answer=llm_answer,
        claims=final_claims,
        debate_cycles=cycles,
        evidence_pool=evidence_pool,
        judge_verdicts=judge_verdicts,
        correction_signal=correction_signal,
        routing_decision=routing,
        aggregate_confidence=aggregate,
        debate_transcript=transcript,
        query_id=query_id,
        rollout_id=rollout_id,
        cse_result=cse_result.as_dict() if cse_result else None,
    )


# ── Routing logic ──────────────────────────────────────────────────────────────

def _compute_routing(
    final_claims:   List[Claim],
    judge_verdicts: List[JudgeVerdict],
) -> Tuple[str, float]:
    """
    Returns (routing_decision, aggregate_confidence).
    Hard rule checked first — overrides aggregate score entirely.
    """
    claim_map = {c.claim_id: c for c in final_claims}
    agg       = _aggregate_score(judge_verdicts)

    for jv in judge_verdicts:
        claim = claim_map.get(jv.claim_id)
        if claim and claim.is_material and jv.score == 0.0:
            print(f"  ⛔ HARD BLOCK — C{jv.claim_id} is_material + v=0.0: "
                  f"\"{jv.claim_text[:60]}\"")
            return "HARD_BLOCK", agg

    if agg >= CONFIDENCE_THRESHOLD_HIGH:
        return "DELIVER", agg
    elif agg >= CONFIDENCE_THRESHOLD_LOW:
        return "RETRY", agg
    else:
        return "HUMAN_REVIEW", agg


def _aggregate_score(judge_verdicts: List[JudgeVerdict]) -> float:
    """
    Weighted min-mean aggregate:
      Material claims:     min (weakest link — safety critical)
      Non-material claims: mean
      Combined:            70% material min + 30% non-material mean
    """
    if not judge_verdicts:
        return 0.5

    mat  = [jv for jv in judge_verdicts if jv.is_material]
    nmat = [jv for jv in judge_verdicts if not jv.is_material]

    if mat and nmat:
        return round(0.70 * min(jv.score for jv in mat)
                     + 0.30 * sum(jv.score for jv in nmat) / len(nmat), 4)
    if mat:
        return round(min(jv.score for jv in mat), 4)
    return round(sum(jv.score for jv in nmat) / len(nmat), 4)


# ── Print helpers ──────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print("\n" + "▓" * 68)
    print(f"  GUARDRAILS GATEWAY — {title}")
    print("▓" * 68)


def _print_judge_verdicts(verdicts: List[JudgeVerdict]) -> None:
    icon = {1.0: "✅", 0.5: "⚠️ ", 0.0: "❌"}
    for jv in verdicts:
        mat = "⚠ material" if jv.is_material else "  context "
        print(f"    {icon.get(jv.score,'?')} [{mat}] C{jv.claim_id}: "
              f"v={jv.score}  \"{jv.claim_text[:60]}\"")


def _print_cse_result(result) -> None:
    print(f"  ─ CSE {result.version} ───────────────────────────────────────")
    # Prefer rich score_breakdown if available
    sb = getattr(result, "score_breakdown", None)
    if sb is not None:
        def _fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"
        print(f"    F_llm         : {_fmt(getattr(sb, 'faithfulness_score', None))}  (faithfulness)")
        print(f"    H_llm_inv     : {_fmt(getattr(sb, 'hallucination_risk_inverse', None))}  (hallucination inverse)")
        print(f"    Relevancy     : {_fmt(getattr(sb, 'contextual_relevancy_score', None))}")
        print(f"    Judge eval    : {_fmt(getattr(sb, 'judge_eval_score', None))}")
    elif result.components is not None:
        c = result.components
        print(f"    F_llm         : {c.f_llm:.4f}  (faithfulness)")
        print(f"    H_llm         : {c.h_llm:.4f}  (hallucination — lower is better)")
        print(f"    Relevancy     : {c.relevancy:.4f}")
        print(f"    Judge eval    : {c.judge_eval:.4f}")
    else:
        print("    Component scores: unavailable")
    print(f"    ── Final score: {result.final_score:.4f}  → {result.routing_decision}")
    explanation = getattr(result, "explanation", None)
    if explanation:
        print(f"    Explanation   : {explanation}")
    if result.error:
        print(f"    ⚠  fallback reason: {result.error[:120]}")
