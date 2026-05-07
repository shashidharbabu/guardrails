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
7.  Compute routing decision (hard rule first, then aggregate score)
8.  [STORAGE] Update query with final CSE score + routing decision
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

WHAT IS AGGREGATE CONFIDENCE?
------------------------------
Currently: min-aggregated judge scores for material claims (70% weight)
           + mean judge scores for non-material claims (30% weight)

Full CSE formula (Phase 2 — when DeepEval Layer 1 is integrated):
  final = 0.30*F_llm + 0.25*(1-H_llm) + 0.10*relevancy + 0.35*judge_eval_score

The paper must report this as "judge-only aggregate (v0.1)" not the full formula.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from . import storage
from . import mad_tracing as lf
from .claim_extractor import extract_claims
from .config import (
    MAX_CYCLES,
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_LOW,
)
from .debate_engine import run_debate, build_transcript
from .judge import judge_claims
from .models import Claim, JudgeVerdict, MADOutput


def run_mad(query: str, llm_answer: str) -> MADOutput:
    """
    Full MAD pipeline with complete storage integration.

    Every intermediate result is written to SQLite so the GRPO
    feedback loop can consume it after the run completes.
    """
    # ── 1. GENERATE ROLLOUT_ID ─────────────────────────────────────────────────
    # rollout_id is a UUID that uniquely identifies THIS run of the pipeline
    # for this query. For GRPO training, you run the same query multiple times
    # (multiple rollouts) and compare Brier rewards across rollouts to compute
    # grpo_advantage = rollout_total - mean_across_rollouts.
    rollout_id = str(uuid.uuid4())
    query_id   = str(uuid.uuid4())

    with lf.observe(
        "span",
        "mad.pipeline",
        input_payload={
            "query_id": query_id,
            "rollout_id": rollout_id,
            "query_chars": len(query),
            "answer_chars": len(llm_answer),
            "max_cycles": MAX_CYCLES,
        },
    ):
        _banner("MAD PIPELINE")
        print(f"  query_id  : {query_id}")
        print(f"  rollout_id: {rollout_id}")
        print(f"  Query     : {query}")
        print(f"  Answer    : {llm_answer[:120]}{'...' if len(llm_answer) > 120 else ''}")

        # ── 2. INITIALISE STORAGE ───────────────────────────────────────────────
        storage.init_db()

        # ── 3. EXTRACT CLAIMS ───────────────────────────────────────────────────
        print("\n[Step 1/4] Extracting atomic claims...")
        with lf.observe(
            "span",
            "mad.claim_extraction",
            input_payload={"query_id": query_id, "rollout_id": rollout_id},
        ):
            claims = extract_claims(query, llm_answer)
        n_mat = sum(1 for c in claims if c.is_material)
        print(f"           {len(claims)} claims ({n_mat} material, "
              f"{len(claims)-n_mat} contextual)")

        # ── 4. WRITE QUERY ROW (TABLE 1) ─────────────────────────────────────────
        # Written BEFORE debate starts. final_cse_score + routing_decision = NULL.
        # Updated at end of pipeline with actual values.
        with lf.observe(
            "span",
            "mad.storage.write_query",
            input_payload={"query_id": query_id, "rollout_id": rollout_id},
        ):
            storage.write_query(
                query_id=query_id,
                rollout_id=rollout_id,
                query_text=query,
                llm_answer=llm_answer,
                chunk_ids=[],  # populated after debate once evidence_pool is known
            )

        # ── 5. RUN DEBATE ──────────────────────────────────────────────────────────
        # debate_engine handles all internal storage writes:
        #   - claims table: post_step_A, post_cycle1, post_cycle2
        #   - attacks table: p_before at challenge time, p_after after revision
        print(f"\n[Step 2/4] Running {MAX_CYCLES}-cycle debate...")
        with lf.observe(
            "span",
            "mad.debate",
            input_payload={"query_id": query_id, "rollout_id": rollout_id},
        ):
            cycles, final_claims, evidence_pool = run_debate(
                query=query,
                claims=claims,
                query_id=query_id,
                rollout_id=rollout_id,
                max_cycles=MAX_CYCLES,
            )

        # ── 6. JUDGE EVALUATION ────────────────────────────────────────────────────
        # Judge is PARTIALLY BLIND:
        #   SEES:     final verdicts + evidence pool + user query
        #   DOES NOT SEE: confidence scores (stripped to prevent anchoring)
        #                 Agent B's challenge framing
        print("\n[Step 3/4] Judge evaluation (partially blind)...")
        with lf.observe(
            "span",
            "mad.judge",
            input_payload={
                "query_id": query_id,
                "rollout_id": rollout_id,
                "evidence_chunks": len(evidence_pool),
            },
        ):
            judge_verdicts, correction_signal = judge_claims(
                query=query,
                final_claims=final_claims,
                evidence_pool=evidence_pool,
            )
        _print_judge_verdicts(judge_verdicts)

        # ── 7. WRITE JUDGE VERDICTS (TABLE 4) ─────────────────────────────────────
        # Last thing MAD writes. After this, data passes to CSE.
        # v_label (1.0/0.5/0.0) is what the feedback loop uses for Brier reward.
        pool_ids = [c.chunk_id for c in evidence_pool]
        with lf.observe(
            "span",
            "mad.storage.write_judge_verdicts",
            input_payload={"query_id": query_id, "rollout_id": rollout_id},
        ):
            storage.write_judge_verdicts(
                query_id=query_id,
                rollout_id=rollout_id,
                judge_verdicts=judge_verdicts,
                evidence_pool_ids=pool_ids,
            )

        # ── 8. COMPUTE ROUTING ─────────────────────────────────────────────────────
        print("\n[Step 4/4] Routing decision...")
        with lf.observe(
            "span",
            "mad.routing",
            input_payload={"query_id": query_id, "rollout_id": rollout_id},
        ):
            routing, aggregate = _compute_routing(final_claims, judge_verdicts)
        print(f"           Aggregate confidence: {aggregate:.4f}")
        print(f"           Routing decision    : {routing}")
        if correction_signal:
            print(f"           Correction signal   : {correction_signal[:90]}...")

        # ── 9. UPDATE QUERY WITH FINAL RESULTS (TABLE 1) ───────────────────────────
        with lf.observe(
            "span",
            "mad.storage.update_query_cse",
            input_payload={"query_id": query_id, "rollout_id": rollout_id},
        ):
            storage.update_query_cse(
                query_id=query_id,
                rollout_id=rollout_id,
                cse_score=aggregate,
                routing_decision=routing,
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

    # ── Hard rule: is_material claim with v=0.0 → HARD_BLOCK ──────────────────
    # A fabricated regulatory claim (v=0.0) that is is_material cannot
    # reach the user under ANY confidence threshold. No retry. Block immediately.
    for jv in judge_verdicts:
        claim = claim_map.get(jv.claim_id)
        if claim and claim.is_material and jv.score == 0.0:
            print(f"  ⛔ HARD BLOCK — C{jv.claim_id} is_material + v=0.0: "
                  f"\"{jv.claim_text[:60]}\"")
            return "HARD_BLOCK", agg

    # ── Soft rules ─────────────────────────────────────────────────────────────
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
