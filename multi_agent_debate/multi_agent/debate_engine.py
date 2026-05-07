"""
debate_engine.py — Debate orchestrator + storage integration
=============================================================

WHAT THIS DOES
--------------
Runs 2 structured challenge-revision cycles and writes to SQLite at
EVERY stage per your friend's storage spec. The data written here is
what the GRPO feedback loop reads to compute rewards.

STORAGE WRITE ORDER (per friend's spec)
----------------------------------------
Stage 1: Query enters     → write_query() [called from mad_pipeline]
Stage 2: Agent A step A   → write_claims_checkpoint(post_step_A)
Stage 3: Agent B cycle 1  → write_attacks(cycle=1, p_before, p_after=NULL)
Stage 4: Agent A step C   → write_claims_checkpoint(post_cycle1)
                          → update_attack_p_after(cycle=1)
Stage 5: Agent B cycle 2  → write_attacks(cycle=2, p_before, p_after=NULL)
Stage 6: Agent A final    → write_claims_checkpoint(post_cycle2)
                          → update_attack_p_after(cycle=2)
Stage 7: Judge runs       → write_judge_verdicts() [called from mad_pipeline]
Stage 8: CSE runs         → update_query_cse() [called from mad_pipeline]

IMPORTANT STORAGE RULES
-----------------------
- claims table: NEVER update existing rows — always INSERT new checkpoint rows
- attacks table: INSERT with p_after=NULL, then UPDATE p_after after revision
- b_reward in attacks: NEVER written by MAD — feedback loop writes this
- agent_b_prompt in attacks: NULL for now — Phase 2

CONFIDENCE SIGNAL
-----------------
After each cycle, a per-cycle confidence signal is computed and stored
in the DebateCycle object. This is the min-aggregated confidence of all
material claims — the weakest link dominates. CSE uses this plus Judge
scores for the final routing decision.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from . import agent_a, agent_b, storage
from . import mad_tracing as lf
from .models import (
    Claim, Challenge, DebateCycle, EvidenceChunk, JudgeVerdict, Verdict,
)


def run_debate(
    query:      str,
    claims:     List[Claim],
    query_id:   str,
    rollout_id: str,
    max_cycles: int = 2,
) -> Tuple[List[DebateCycle], List[Claim], List[EvidenceChunk]]:
    """
    Run the full 2-cycle debate with storage writes at every stage.

    Args:
        query:      original user query
        claims:     atomic claims extracted from LLM answer
        query_id:   UUID for this query (for storage joins)
        rollout_id: UUID for this rollout (for GRPO multi-rollout comparison)
        max_cycles: max debate cycles (default 2)

    Returns:
        cycles        — full record of every cycle (for transcript + paper)
        final_claims  — Agent A's final revised claims after all cycles
        evidence_pool — all chunks from both agents, deduped, no agent labels
    """
    cycles:         List[DebateCycle]           = []
    evidence_map:   Dict[str, EvidenceChunk]    = {}
    current_claims: List[Claim]                 = claims

    # per_claim_prompts carries forward across cycles so each checkpoint
    # can build on the previous prompt (base + B's challenges accumulate)
    per_claim_prompts: Dict[int, str] = {}

    for cycle_num in range(1, max_cycles + 1):
        _header(f"DEBATE CYCLE {cycle_num}")

        # ══════════════════════════════════════════════════════════════════════
        # STEP A — Agent A initial verification (cycle 1 only)
        #          In cycle 2+, A carries forward its revised claims
        # ══════════════════════════════════════════════════════════════════════
        _step(cycle_num, "A", "Agent A — verification")

        if cycle_num == 1:
            # Full RAG retrieval + LLM verification
            a_verified, a_evidence, per_claim_prompts = agent_a.verify_claims(
                query=query,
                claims=current_claims,
            )
            for chunk in a_evidence:
                evidence_map[chunk.chunk_id] = chunk

            # ── STORAGE STAGE 2: write claims at post_step_A ─────────────────
            # Write one row per claim with Agent A's initial verdict + prompt.
            # The per_claim_prompts dict maps claim_id → full prompt string.
            # This is the GRPO training input for post_step_A checkpoint.
            storage.write_claims_checkpoint(
                query_id=query_id,
                rollout_id=rollout_id,
                claims=a_verified,
                checkpoint="post_step_A",
                per_claim_prompts=per_claim_prompts,
            )
            _log_claims(a_verified, "Agent A initial verdicts")

        else:
            # Cycle 2+: A's verdicts carry forward from end of previous cycle
            a_verified = current_claims

        # ══════════════════════════════════════════════════════════════════════
        # STEP B — Agent B adversarial challenges
        # ══════════════════════════════════════════════════════════════════════
        _step(cycle_num, "B", "Agent B — adversarial challenges")

        b_challenges, b_evidence = agent_b.challenge_claims(
            query=query,
            agent_a_claims=a_verified,
            existing_evidence=list(evidence_map.values()),
            cycle=cycle_num,
        )
        for chunk in b_evidence:
            evidence_map[chunk.chunk_id] = chunk

        # ── STORAGE STAGE 3/5: write attacks with p_before, p_after=NULL ─────
        # p_before_attack = Agent A's confidence RIGHT NOW (before revision)
        # p_after_attack  = NULL — filled after Agent A revises in step C
        # b_reward        = NULL — feedback loop fills this later
        storage.write_attacks(
            query_id=query_id,
            rollout_id=rollout_id,
            challenges=b_challenges,
            claims_before=a_verified,
            cycle=cycle_num,
        )
        _log_challenges(b_challenges, f"Agent B challenges (cycle {cycle_num})")

        # ══════════════════════════════════════════════════════════════════════
        # STEP C — Agent A revises verdicts (not claim text)
        # ══════════════════════════════════════════════════════════════════════
        _step(cycle_num, "C", "Agent A — revising verdicts")

        checkpoint = f"post_cycle{cycle_num}"

        a_revised, new_prompts = agent_a.revise_verdicts(
            claims=a_verified,
            challenges=b_challenges,
            agent_b_evidence=b_evidence,
            base_prompts=per_claim_prompts,
            cycle=cycle_num,
        )

        # Update per_claim_prompts — these become the base for the NEXT cycle
        per_claim_prompts = new_prompts

        # ── STORAGE STAGE 4/6a: write claims at post_cycle N ─────────────────
        # New rows — never update existing ones.
        # Each row shows A's revised confidence after seeing B's challenge.
        # The prompt now includes B's challenge text appended to the base.
        storage.write_claims_checkpoint(
            query_id=query_id,
            rollout_id=rollout_id,
            claims=a_revised,
            checkpoint=checkpoint,
            per_claim_prompts=per_claim_prompts,
        )

        # ── STORAGE STAGE 4/6b: update p_after_attack for this cycle ─────────
        # Now that A has revised, fill in p_after_attack for all attacks in
        # this cycle. This is the ONLY place MAD updates an existing row.
        storage.update_attack_p_after(
            query_id=query_id,
            rollout_id=rollout_id,
            revised_claims=a_revised,
            cycle=cycle_num,
        )

        _log_claims(a_revised, f"Agent A revised (cycle {cycle_num})")

        # ── Confidence signal for this cycle ──────────────────────────────────
        with lf.observe(
            "span",
            "mad.debate.post_cycle",
            input_payload={"cycle": cycle_num, "claims_revised": len(a_revised)},
        ) as cycle_obs:
            conf_signal = _confidence_signal(a_revised)
            print(f"\n  [Cycle {cycle_num}] Confidence signal: {conf_signal:.4f}")
            if cycle_obs is not None:
                try:
                    cycle_obs.update(output={"confidence_signal": conf_signal})
                except Exception:
                    pass

        cycles.append(DebateCycle(
            cycle_number=cycle_num,
            agent_a_report=a_verified,
            agent_b_challenges=b_challenges,
            agent_a_revised=a_revised,
            confidence_signal=conf_signal,
        ))

        current_claims = a_revised

        # ── Early exit: all material claims strongly supported ────────────────
        material = [c for c in current_claims if c.is_material]
        if material and all(
            c.verdict == Verdict.SUPPORTED and c.confidence >= 0.88
            for c in material
        ):
            print(f"\n  [Debate] Early exit after cycle {cycle_num} — "
                  f"all material claims strongly supported")
            break

    evidence_pool = _filter_evidence_pool(query, list(evidence_map.values()))
    print(f"\n  [Debate] Complete. {len(evidence_pool)} chunks in evidence pool.")
    return cycles, current_claims, evidence_pool


# ── Transcript builder ─────────────────────────────────────────────────────────

def build_transcript(
    query:             str,
    llm_answer:        str,
    cycles:            List[DebateCycle],
    judge_verdicts:    List[JudgeVerdict],
    correction_signal: str,
) -> str:
    """
    Build a human-readable debate transcript.
    Used for: logging, retry loop context, paper appendix.
    """
    W = 72
    lines = [
        "=" * W,
        "  GUARDRAILS GATEWAY — MAD DEBATE TRANSCRIPT",
        "=" * W,
        f"  Query : {query}",
        f"  Answer: {llm_answer[:200]}{'...' if len(llm_answer) > 200 else ''}",
        "",
    ]

    for cycle in cycles:
        lines += ["─" * W, f"  CYCLE {cycle.cycle_number}", "─" * W, ""]

        lines.append("  Agent A — Initial Verdicts:")
        for c in cycle.agent_a_report:
            mat = "⚠ MATERIAL" if c.is_material else "  context "
            v   = c.verdict.value if c.verdict else "PENDING"
            lines.append(f"    [{mat}] C{c.claim_id}: {v} (p={c.confidence:.2f})")
            lines.append(f"             \"{c.claim_text[:88]}\"")
            lines.append(f"             {c.reasoning[:120]}")
        lines.append("")

        lines.append("  Agent B — Challenges:")
        if not cycle.agent_b_challenges:
            lines.append("    (no challenges this cycle)")
        for ch in cycle.agent_b_challenges:
            sv = f"→ {ch.suggested_verdict.value}" if ch.suggested_verdict else ""
            lines.append(f"    C{ch.claim_id} [{ch.challenge_type.value}] {sv}")
            lines.append(f"             {ch.challenge_text[:120]}")
        lines.append("")

        lines.append("  Agent A — Revised Verdicts:")
        for c in cycle.agent_a_revised:
            v = c.verdict.value if c.verdict else "PENDING"
            lines.append(f"    C{c.claim_id}: {v} p={c.confidence:.2f}  "
                         f"\"{c.claim_text[:76]}\"")
        lines.append(f"\n  Cycle {cycle.cycle_number} confidence: "
                     f"{cycle.confidence_signal:.4f}\n")

    lines += ["─" * W, "  JUDGE VERDICTS", "─" * W]
    score_icon = {1.0: "✅", 0.5: "⚠️ ", 0.0: "❌"}
    for jv in judge_verdicts:
        mat = "⚠" if jv.is_material else " "
        ico = score_icon.get(jv.score, "?")
        lines.append(f"  {ico} {mat} C{jv.claim_id}: v={jv.score}  "
                     f"\"{jv.claim_text[:72]}\"")
        lines.append(f"       {jv.reasoning[:140]}")
    lines.append("")

    if correction_signal:
        lines += ["─" * W, "  CORRECTION SIGNAL (→ LLM retry)", "─" * W]
        lines.append(f"  {correction_signal}")
    else:
        lines.append("  CORRECTION SIGNAL: null — all material claims verified")

    lines.append("=" * W)
    return "\n".join(lines)


# ── Internal helpers ───────────────────────────────────────────────────────────

_REGULATION_DOC_HINTS = {
    # Maps regulation name → substrings that appear in chunk doc_id / chunk_id
    "GDPR":       ["gdpr", "2016_679"],
    "HIPAA":      ["hipaa", "hitech", "hhs_ocr", "45cfr"],
    "EU AI Act":  ["eu_ai_act", "ai_act"],
    "NIS2":       ["nis2", "nis_2"],
    "CCPA":       ["ccpa", "cpra", "california"],
    "HITECH":     ["hitech"],
    "ISO 27001":  ["iso_27001", "iso27001"],
    "NIST":       ["nist", "sp800"],
    "DSA":        ["dsa_2022", "digital_services"],
    "CRA":        ["cra_2024", "cyber_resilience"],
    "PIPL":       ["pipl", "china"],
    "PDPA":       ["pdpa", "singapore"],
    "PDPL":       ["pdpl", "saudi"],
    "APPI":       ["appi", "japan"],
    "ADA":        ["ada_title"],
}

_KNOWN_REGULATIONS = list(_REGULATION_DOC_HINTS.keys())


def _query_regulation(query: str) -> str:
    """Detect primary regulation from query text. Returns '' if unknown."""
    q = query.lower()
    checks = [
        ("GDPR",      ["gdpr", "general data protection regulation", "2016/679"]),
        ("HIPAA",     ["hipaa", "protected health information", " phi", "ephi", "hipaa security rule", "hipaa privacy"]),
        ("EU AI Act", ["eu ai act", "ai act", "artificial intelligence act"]),
        ("NIS2",      ["nis2", "nis 2"]),
        ("CCPA",      ["ccpa", "cpra", "california consumer privacy"]),
        ("HITECH",    ["hitech"]),
        ("ISO 27001", ["iso 27001", "iso27001"]),
        ("NIST",      ["nist csf", "nist sp", "nist cybersecurity", "nist ssdf"]),
        ("DSA",       ["digital services act"]),
        ("CRA",       ["cyber resilience act"]),
        ("PIPL",      ["pipl", "china personal information"]),
        ("PDPA",      ["pdpa"]),
        ("PDPL",      ["pdpl", "saudi"]),
        ("APPI",      ["appi", "japan appi"]),
        ("ADA",       ["americans with disabilities act", "ada title"]),
    ]
    for name, keywords in checks:
        if any(kw in q for kw in keywords):
            return name
    return ""


def _chunk_matches_regulation(chunk: EvidenceChunk, regulation: str) -> bool:
    """Return True if this chunk is from the detected regulation's corpus."""
    if not regulation:
        return True   # no regulation detected — keep all chunks
    hints = _REGULATION_DOC_HINTS.get(regulation, [])
    key = (chunk.chunk_id + chunk.source).lower()
    return any(h in key for h in hints)


def _filter_evidence_pool(
    query: str,
    pool: List[EvidenceChunk],
    min_pool_size: int = 5,
) -> List[EvidenceChunk]:
    """
    Remove off-regulation chunks from the evidence pool before the judge
    sees it. Prevents EU AI Act / DSA / NIS2 chunks from polluting a
    GDPR or HIPAA query.

    Strategy:
      1. Detect the primary regulation from the query text.
      2. Split pool into on-topic and off-topic chunks.
      3. Return on-topic chunks. If fewer than min_pool_size remain,
         backfill with off-topic chunks to avoid an empty evidence pool.
    """
    regulation = _query_regulation(query)
    if not regulation:
        return pool   # can't detect regulation — keep all

    on_topic  = [c for c in pool if     _chunk_matches_regulation(c, regulation)]
    off_topic = [c for c in pool if not _chunk_matches_regulation(c, regulation)]

    if off_topic:
        removed = [c.chunk_id for c in off_topic]
        print(f"  [EvidenceFilter] Detected regulation: {regulation}")
        print(f"  [EvidenceFilter] Removed {len(off_topic)} off-topic chunks: "
              f"{', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}")

    if len(on_topic) >= min_pool_size:
        return on_topic

    # Not enough on-topic chunks — backfill from off-topic
    backfill = off_topic[:min_pool_size - len(on_topic)]
    if backfill:
        print(f"  [EvidenceFilter] Pool too small ({len(on_topic)}); "
              f"backfilling {len(backfill)} off-topic chunks")
    return on_topic + backfill


def _confidence_signal(claims: List[Claim]) -> float:
    """
    Conservative aggregation for the per-cycle confidence signal.
    Material claims: min (weakest link dominates — safety critical).
    Non-material: mean.
    Mix: 70% material min + 30% non-material mean.
    """
    mat  = [c for c in claims if c.is_material]
    nmat = [c for c in claims if not c.is_material]

    if not claims:
        return 0.5
    if mat and nmat:
        return round(0.70 * min(c.confidence for c in mat)
                     + 0.30 * sum(c.confidence for c in nmat) / len(nmat), 4)
    if mat:
        return round(min(c.confidence for c in mat), 4)
    return round(sum(c.confidence for c in nmat) / len(nmat), 4)


def _header(title: str) -> None:
    print(f"\n{'█' * 60}\n  {title}\n{'█' * 60}")


def _step(cycle: int, step: str, desc: str) -> None:
    print(f"\n[C{cycle}·{step}] {desc}")


def _log_claims(claims: List[Claim], label: str) -> None:
    print(f"  {label}:")
    for c in claims:
        mat = "⚠ mat" if c.is_material else "  ctx"
        v   = c.verdict.value if c.verdict else "PENDING"
        print(f"    [{mat}] C{c.claim_id}: {v:14s} p={c.confidence:.2f}  "
              f"\"{c.claim_text[:62]}\"")


def _log_challenges(challenges: List[Challenge], label: str) -> None:
    print(f"  {label}:")
    if not challenges:
        print("    (no challenges)")
        return
    for ch in challenges:
        sv = f"→ {ch.suggested_verdict.value}" if ch.suggested_verdict else ""
        print(f"    C{ch.claim_id} [{ch.challenge_type.value:22s}] {sv}")
        print(f"         {ch.challenge_text[:80]}")
