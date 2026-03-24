"""
mad_pipeline.py — Top-level MAD pipeline orchestrator.

Input:  query (str) + llm_answer (str)
Output: MADOutput — full structured result with routing decision + correction signal

Routing (checked in order):
  1. HARD_BLOCK   — any is_material claim with judge score = 0.0
  2. DELIVER      — aggregate confidence >= 0.8
  3. RETRY        — aggregate confidence 0.4–0.8 → send correction signal to LLM
  4. HUMAN_REVIEW — aggregate confidence < 0.4
"""
from __future__ import annotations

from multi_agent.claim_extractor import extract_claims
from multi_agent.debate_engine   import run_debate, build_transcript
from multi_agent.judge           import judge_claims
from multi_agent.models          import MADOutput, JudgeVerdict, Claim
from multi_agent.config          import (
    MAX_CYCLES,
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_LOW,
)


def run_mad(query: str, llm_answer: str) -> MADOutput:
    """
    Full MAD pipeline:
      1. Extract atomic claims (with is_material flag)
      2. Run 2-cycle structured debate (Agent A ↔ Agent B)
      3. Judge evaluation (partially blind)
      4. Routing decision
      5. Return MADOutput
    """
    _banner("MAD PIPELINE STARTING")
    print(f"  Query:  {query}")
    print(f"  Answer: {llm_answer[:120]}{'...' if len(llm_answer) > 120 else ''}")
    _banner_end()

    # ── 1. Claim extraction ──────────────────────────────────────────────────
    print("\n[Step 1/4] Extracting atomic claims...")
    claims = extract_claims(query, llm_answer)
    n_mat  = sum(1 for c in claims if c.is_material)
    print(f"           Extracted {len(claims)} claims ({n_mat} material, {len(claims)-n_mat} contextual)")

    # ── 2. Run debate ────────────────────────────────────────────────────────
    print(f"\n[Step 2/4] Running {MAX_CYCLES}-cycle debate...")
    cycles, final_claims, evidence_pool = run_debate(
        query=query,
        claims=claims,
        max_cycles=MAX_CYCLES,
    )

    # ── 3. Judge evaluation ──────────────────────────────────────────────────
    print("\n[Step 3/4] Judge evaluation (partially blind)...")
    judge_verdicts, correction_signal = judge_claims(
        query=query,
        final_claims=final_claims,
        evidence_pool=evidence_pool,
    )

    _print_judge_verdicts(judge_verdicts)

    # ── 4. Routing decision ──────────────────────────────────────────────────
    print("\n[Step 4/4] Computing routing decision...")
    routing, aggregate_confidence = _compute_routing(final_claims, judge_verdicts)
    print(f"           Aggregate confidence : {aggregate_confidence:.4f}")
    print(f"           Routing decision     : {routing}")
    if correction_signal:
        print(f"           Correction signal   : {correction_signal[:100]}...")

    # ── 5. Build transcript ──────────────────────────────────────────────────
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
        aggregate_confidence=aggregate_confidence,
        debate_transcript=transcript,
    )


# ── Routing logic ─────────────────────────────────────────────────────────────

def _compute_routing(
    final_claims:   list[Claim],
    judge_verdicts: list[JudgeVerdict],
) -> tuple[str, float]:
    """
    Hard rule first, then soft rules on aggregate confidence.
    Returns (routing_decision, aggregate_confidence).
    """
    claim_map   = {c.claim_id: c for c in final_claims}
    agg         = _aggregate_score(judge_verdicts)

    # ── Hard rule: any is_material claim with judge score 0.0 ──
    for jv in judge_verdicts:
        claim = claim_map.get(jv.claim_id)
        if claim and claim.is_material and jv.score == 0.0:
            print(
                f"  ⛔ HARD BLOCK triggered by C{jv.claim_id}: "
                f"\"{jv.claim_text[:60]}\" scored 0.0"
            )
            return "HARD_BLOCK", agg

    # ── Soft rules ──
    if agg >= CONFIDENCE_THRESHOLD_HIGH:
        return "DELIVER", agg
    elif agg >= CONFIDENCE_THRESHOLD_LOW:
        return "RETRY", agg
    else:
        return "HUMAN_REVIEW", agg


def _aggregate_score(judge_verdicts: list[JudgeVerdict]) -> float:
    """
    Weighted aggregate:
      - Material claims: min aggregation (weakest link — safety-critical)
      - Non-material:    mean
      - Combined:        70% material min + 30% non-material mean
    If no verdicts: return 0.5 (uncertain)
    """
    if not judge_verdicts:
        return 0.5

    material     = [jv for jv in judge_verdicts if jv.is_material]
    non_material = [jv for jv in judge_verdicts if not jv.is_material]

    if material and non_material:
        mat_score = min(jv.score for jv in material)
        nm_score  = sum(jv.score for jv in non_material) / len(non_material)
        return round(0.70 * mat_score + 0.30 * nm_score, 4)

    if material:
        return round(min(jv.score for jv in material), 4)

    return round(sum(jv.score for jv in non_material) / len(non_material), 4)


# ── Print helpers ─────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print("\n" + "▓" * 68)
    print(f"  GUARDRAILS GATEWAY — {title}")
    print("▓" * 68)


def _banner_end() -> None:
    print("▓" * 68)


def _print_judge_verdicts(judge_verdicts: list[JudgeVerdict]) -> None:
    print("           Judge verdicts:")
    for jv in judge_verdicts:
        mat = "⚠ material" if jv.is_material else "  context "
        score_icon = {1.0: "✅", 0.5: "⚠️ ", 0.0: "❌"}.get(jv.score, "?")
        print(f"             {score_icon} [{mat}] C{jv.claim_id}: score={jv.score}  \"{jv.claim_text[:60]}\"")
