"""
debate_engine.py — Debate orchestrator

Runs 2 structured challenge-revision cycles:
  Cycle N:
    Step A — Agent A verifies / carries forward claims
    Step B — Agent B challenges (True→Skeptic + gap finding)
    Step C — Agent A revises verdicts (not claim text)
    Signal  — confidence signal sent to CSE (async in full system, returned here)

Compiles evidence pool from both agents (deduplicated, no agent labels).
Builds human-readable transcript for logging and retry context.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from multi_agent.models import Claim, Challenge, DebateCycle, EvidenceChunk, JudgeVerdict, Verdict
from multi_agent import agent_a, agent_b


# ── Public orchestrator ───────────────────────────────────────────────────────

def run_debate(
    query:      str,
    claims:     List[Claim],
    max_cycles: int = 2,
) -> Tuple[List[DebateCycle], List[Claim], List[EvidenceChunk]]:
    """
    Run up to max_cycles challenge-revision cycles.

    Returns:
        cycles        — full record of each cycle (for transcript + paper)
        final_claims  — Agent A's final revised claims
        evidence_pool — all chunks retrieved by both agents (deduped, unlabelled)
    """
    cycles:        List[DebateCycle]  = []
    evidence_map:  dict[str, EvidenceChunk] = {}   # chunk_id → chunk (deduped)
    current_claims: List[Claim]       = claims

    for cycle_num in range(1, max_cycles + 1):
        _print_header(f"DEBATE CYCLE {cycle_num}")

        # ── STEP A: Agent A verifies (cycle 1) or carries forward (cycle 2+) ──
        _print_step(cycle_num, "A", "Agent A — initial verification")
        if cycle_num == 1:
            a_verified, a_evidence = agent_a.verify_claims(query, current_claims)
        else:
            # Cycle 2+: A carries forward revised claims; evidence pool already built
            a_verified = current_claims
            a_evidence = list(evidence_map.values())

        for chunk in a_evidence:
            evidence_map[chunk.chunk_id] = chunk

        _print_claims(a_verified, f"Agent A verdicts (cycle {cycle_num})")

        # ── STEP B: Agent B challenges ──
        _print_step(cycle_num, "B", "Agent B — adversarial challenges")
        b_challenges, b_evidence = agent_b.challenge_claims(
            query=query,
            agent_a_claims=a_verified,
            existing_evidence=list(evidence_map.values()),
            cycle=cycle_num,
        )

        for chunk in b_evidence:
            evidence_map[chunk.chunk_id] = chunk

        _print_challenges(b_challenges, f"Agent B challenges (cycle {cycle_num})")

        # ── STEP C: Agent A revises verdicts ──
        _print_step(cycle_num, "C", "Agent A — revising verdicts")
        a_revised = agent_a.revise_verdicts(
            claims=a_verified,
            challenges=b_challenges,
            agent_b_evidence=b_evidence,
        )

        _print_claims(a_revised, f"Agent A revised verdicts (cycle {cycle_num})")

        # ── Confidence signal for this cycle ──
        conf_signal = _confidence_signal(a_revised)
        print(f"\n  [Cycle {cycle_num}] Confidence signal: {conf_signal:.4f}")

        cycles.append(DebateCycle(
            cycle_number=cycle_num,
            agent_a_report=a_verified,
            agent_b_challenges=b_challenges,
            agent_a_revised=a_revised,
            confidence_signal=conf_signal,
        ))

        current_claims = a_revised

        # ── Early exit: all material claims strongly supported ──
        material = [c for c in current_claims if c.is_material]
        if material and all(
            c.verdict == Verdict.SUPPORTED and c.confidence >= 0.88
            for c in material
        ):
            print(f"\n  [Debate] Early exit — all material claims strongly supported after cycle {cycle_num}")
            break

    final_evidence_pool = list(evidence_map.values())
    print(f"\n  [Debate] Complete. Evidence pool: {len(final_evidence_pool)} chunks across both agents.")
    return cycles, current_claims, final_evidence_pool


# ── Transcript builder ────────────────────────────────────────────────────────

def build_transcript(
    query:             str,
    llm_answer:        str,
    cycles:            List[DebateCycle],
    judge_verdicts:    List[JudgeVerdict],
    correction_signal: str,
) -> str:
    """
    Build a self-contained human-readable debate transcript.
    Used for: logging, the retry loop context, and the paper.
    """
    W = 72
    sep = "=" * W
    thin = "─" * W

    lines = [
        sep,
        "  GUARDRAILS GATEWAY — MAD DEBATE TRANSCRIPT",
        sep,
        f"  Query:  {query}",
        f"  Answer: {llm_answer[:200]}{'...' if len(llm_answer) > 200 else ''}",
        "",
    ]

    for cycle in cycles:
        lines += [thin, f"  CYCLE {cycle.cycle_number}", thin, ""]

        lines.append("  Agent A — Initial Verdicts:")
        for c in cycle.agent_a_report:
            mat = "⚠ MATERIAL" if c.is_material else "  context "
            v   = c.verdict.value if c.verdict else "PENDING"
            lines.append(f"    [{mat}] C{c.claim_id}: {v} (p={c.confidence:.2f})")
            lines.append(f"             \"{c.claim_text[:90]}\"")
            lines.append(f"             Reasoning: {c.reasoning[:120]}")
        lines.append("")

        lines.append("  Agent B — Challenges:")
        if not cycle.agent_b_challenges:
            lines.append("    (no challenges generated this cycle)")
        for ch in cycle.agent_b_challenges:
            sv = f"→ suggests {ch.suggested_verdict.value}" if ch.suggested_verdict else ""
            lines.append(f"    C{ch.claim_id} [{ch.challenge_type.value}] {sv}")
            lines.append(f"             {ch.challenge_text[:120]}")
        lines.append("")

        lines.append("  Agent A — Revised Verdicts:")
        for c in cycle.agent_a_revised:
            v = c.verdict.value if c.verdict else "PENDING"
            lines.append(f"    C{c.claim_id}: {v} (p={c.confidence:.2f}) — {c.claim_text[:80]}")
        lines.append(f"\n  Cycle {cycle.cycle_number} confidence signal: {cycle.confidence_signal:.4f}")
        lines.append("")

    lines += [thin, "  JUDGE VERDICTS", thin]
    for jv in judge_verdicts:
        mat = "⚠" if jv.is_material else " "
        lines.append(f"  {mat} C{jv.claim_id}: score={jv.score}")
        lines.append(f"       \"{jv.claim_text[:80]}\"")
        lines.append(f"       {jv.reasoning[:140]}")
    lines.append("")

    if correction_signal:
        lines += [thin, "  CORRECTION SIGNAL (for LLM retry)", thin]
        lines.append(f"  {correction_signal}")
    else:
        lines.append("  CORRECTION SIGNAL: null (all material claims verified)")

    lines.append(sep)
    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _confidence_signal(claims: List[Claim]) -> float:
    """
    Conservative aggregation:
    - Material claims: use min (weakest link dominates)
    - If no material claims: use mean of all
    - Mix: 70% material min + 30% non-material mean
    """
    material     = [c for c in claims if c.is_material]
    non_material = [c for c in claims if not c.is_material]

    if not claims:
        return 0.5

    if material and non_material:
        mat_score = min(c.confidence for c in material)
        nm_score  = sum(c.confidence for c in non_material) / len(non_material)
        return round(0.70 * mat_score + 0.30 * nm_score, 4)

    if material:
        return round(min(c.confidence for c in material), 4)

    return round(sum(c.confidence for c in non_material) / len(non_material), 4)


def _print_header(title: str) -> None:
    print(f"\n{'█' * 60}")
    print(f"  {title}")
    print(f"{'█' * 60}")


def _print_step(cycle: int, step: str, desc: str) -> None:
    print(f"\n[C{cycle}·{step}] {desc}")


def _print_claims(claims: List[Claim], label: str = "") -> None:
    if label:
        print(f"  {label}:")
    for c in claims:
        mat = "⚠ mat" if c.is_material else "  ctx"
        v   = c.verdict.value if c.verdict else "PENDING"
        print(f"    [{mat}] C{c.claim_id}: {v:14s} p={c.confidence:.2f}  \"{c.claim_text[:65]}\"")


def _print_challenges(challenges: List[Challenge], label: str = "") -> None:
    if label:
        print(f"  {label}:")
    if not challenges:
        print("    (no challenges)")
        return
    for ch in challenges:
        sv = f"→ {ch.suggested_verdict.value}" if ch.suggested_verdict else ""
        print(f"    C{ch.claim_id} [{ch.challenge_type.value:22s}] {sv}")
        print(f"         {ch.challenge_text[:80]}")
