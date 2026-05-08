#!/usr/bin/env python3
"""
confidence/ablation_study.py — CSE Formula Ablation Study.

PURPOSE
-------
Determine which combination of weights for the CSE formula produces the
best routing accuracy across a representative set of queries. The four
formula components are:

    final = w_f * F_llm
          + w_h * (1 - H_llm)
          + w_r * relevancy
          + w_j * judge_eval
    where w_f + w_h + w_r + w_j = 1.0

HOW IT WORKS
------------
1. SCORE COLLECTION (slow — requires Claude API):
   For each test case, run DeepEval ONCE to get the four component scores
   (F_llm, H_llm, relevancy, judge_eval). These are mode-independent —
   they don't change with different weights.

2. WEIGHT SWEEP (fast — pure Python):
   For each weight configuration × each test case, apply the formula to
   the pre-collected component scores. Record final_score, routing, and
   whether routing matches the expected ground truth.

3. RESULTS:
   - Console table: rows = weight configs, columns = test cases
   - ablation_results_<timestamp>.json: full raw data for further analysis
   - ablation_summary_<timestamp>.csv: pivot table (scores) for spreadsheets

WEIGHT CONFIGURATIONS TESTED
------------------------------
  baseline_v2     : F=0.30  H=0.25  R=0.10  J=0.35  (current formula)
  judge_heavy     : F=0.20  H=0.15  R=0.05  J=0.60  (trust MAD debate most)
  faithful_heavy  : F=0.50  H=0.20  R=0.10  J=0.20  (DeepEval faithfulness dominant)
  halluc_focused  : F=0.25  H=0.45  R=0.05  J=0.25  (penalize hallucination hardest)
  balanced        : F=0.25  H=0.25  R=0.25  J=0.25  (uniform weights)
  deepeval_heavy  : F=0.35  H=0.30  R=0.20  J=0.15  (all three DeepEval metrics dominant)
  judge_only      : F=0.00  H=0.00  R=0.00  J=1.00  (v0.1 baseline — judge aggregate only)

USAGE
-----
    # Full ablation with live DeepEval via Claude API (slow — ~5-15 min for 8 cases × 3 metrics):
    PYTHONPATH=multi_agent_debate:rag_folder:. python confidence/ablation_study.py

    # Skip DeepEval — use neutral defaults for F/H/R (fast, ~10 seconds):
    PYTHONPATH=multi_agent_debate:rag_folder:. python confidence/ablation_study.py --no-deepeval

    # Load component scores from a previous run and re-sweep weights:
    python confidence/ablation_study.py --load-scores ablation_scores_20260505_193000.json

    # Add more weight configs from a JSON file (each entry: name,w_f,w_h,w_r,w_j):
    python confidence/ablation_study.py --extra-configs my_weights.json

EXIT CODES
----------
    0  — study complete, results saved
    1  — fatal error (import / Ollama connection failure)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
for _p in [str(_REPO), str(_REPO / "multi_agent_debate"), str(_REPO / "rag_folder")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_REPO / ".env")

try:
    from multi_agent.models import Claim, JudgeVerdict, EvidenceChunk, Verdict
except ImportError:
    from multi_agent_debate.multi_agent.models import Claim, JudgeVerdict, EvidenceChunk, Verdict


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationCase:
    """One test query with ground-truth verdict and expected routing."""
    name:             str
    query:            str
    llm_answer:       str
    rag_chunks:       List[str]
    claims:           List[Claim]
    judge_verdicts:   List[JudgeVerdict]
    expected_routing: str          # DELIVER / RETRY / HUMAN_REVIEW / HARD_BLOCK


@dataclass
class WeightConfig:
    """One formula weight configuration to test."""
    name:  str
    w_f:   float    # faithfulness
    w_h:   float    # hallucination (applied as 1 - H_llm)
    w_r:   float    # relevancy
    w_j:   float    # judge_eval

    def __post_init__(self) -> None:
        total = self.w_f + self.w_h + self.w_r + self.w_j
        assert abs(total - 1.0) < 1e-6, (
            f"Weights for '{self.name}' must sum to 1.0, got {total:.6f}"
        )

    def apply(self, f_llm: float, h_llm: float, relevancy: float, judge_eval: float) -> float:
        score = (
            self.w_f * f_llm
            + self.w_h * (1.0 - h_llm)
            + self.w_r * relevancy
            + self.w_j * judge_eval
        )
        return round(min(1.0, max(0.0, score)), 4)


@dataclass
class ComponentScoreRecord:
    """Pre-computed component scores for one test case (mode-independent)."""
    case_name:  str
    f_llm:      float
    h_llm:      float
    relevancy:  float
    judge_eval: float
    version:    str    # "v2.0" or "v0.1"
    error:      str = ""


@dataclass
class CellResult:
    """Result for one weight config × one test case."""
    config_name:      str
    case_name:        str
    final_score:      float
    routing_decision: str
    expected_routing: str
    correct:          bool   # routing_decision == expected_routing


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

WEIGHT_CONFIGS: List[WeightConfig] = [
    WeightConfig("baseline_v2",    w_f=0.30, w_h=0.25, w_r=0.10, w_j=0.35),
    WeightConfig("judge_heavy",    w_f=0.20, w_h=0.15, w_r=0.05, w_j=0.60),
    WeightConfig("faithful_heavy", w_f=0.50, w_h=0.20, w_r=0.10, w_j=0.20),
    WeightConfig("halluc_focused", w_f=0.25, w_h=0.45, w_r=0.05, w_j=0.25),
    WeightConfig("balanced",       w_f=0.25, w_h=0.25, w_r=0.25, w_j=0.25),
    WeightConfig("deepeval_heavy", w_f=0.35, w_h=0.30, w_r=0.20, w_j=0.15),
    WeightConfig("judge_only",     w_f=0.00, w_h=0.00, w_r=0.00, w_j=1.00),
]


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES — 8 queries covering the four error types from the evaluation spec
# ─────────────────────────────────────────────────────────────────────────────

def _make_test_cases() -> List[AblationCase]:
    """Build the 8 ablation test cases."""

    # ── Case 1: Fully correct HIPAA answer ────────────────────────────────────
    c1 = AblationCase(
        name="C1_hipaa_encrypt_correct",
        query="Does HIPAA require encryption of ePHI at rest?",
        llm_answer=(
            "Yes. Under the HIPAA Security Rule (45 CFR § 164.312(a)(2)(iv)), "
            "covered entities must implement encryption of ePHI at rest where "
            "determined appropriate by a risk analysis. AES-256 is the "
            "industry-standard cipher used in practice."
        ),
        rag_chunks=[
            "45 CFR § 164.312(a)(2)(iv): Implement a mechanism to encrypt and "
            "decrypt electronic PHI where appropriate.",
            "HHS OCR: Encryption renders ePHI unreadable to unauthorized parties "
            "and constitutes a 'safe harbor' under the Breach Notification Rule.",
            "NIST SP 800-111: AES-256 is recommended for healthcare data at rest.",
        ],
        claims=[
            Claim(claim_id=1, claim_text="HIPAA requires encryption where appropriate per risk analysis",
                  is_material=True,  confidence=0.95, verdict=Verdict.SUPPORTED),
            Claim(claim_id=2, claim_text="AES-256 is used in practice for ePHI",
                  is_material=False, confidence=0.80, verdict=Verdict.SUPPORTED),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="HIPAA requires encryption where appropriate per risk analysis",
                         is_material=True,  score=1.0, reasoning="Directly in 45 CFR § 164.312"),
            JudgeVerdict(claim_id=2, claim_text="AES-256 is used in practice for ePHI",
                         is_material=False, score=1.0, reasoning="NIST SP 800-111 confirms"),
        ],
        expected_routing="DELIVER",
    )

    # ── Case 2: Missing caveat — GDPR erasure not absolute ────────────────────
    c2 = AblationCase(
        name="C2_gdpr_erasure_missing_caveat",
        query="Can users always demand their data be deleted under GDPR?",
        llm_answer=(
            "Yes. Under GDPR Article 17, individuals have an absolute right to "
            "erasure of personal data at any time upon request."
        ),
        rag_chunks=[
            "GDPR Article 17(1): Data subjects have the right to erasure.",
            "GDPR Article 17(3): The right does not apply where processing is "
            "necessary for compliance with a legal obligation, archiving in the "
            "public interest, or establishment of legal claims.",
        ],
        claims=[
            Claim(claim_id=1, claim_text="Individuals have an absolute right to erasure at any time",
                  is_material=True,  confidence=0.75, verdict=Verdict.PARTIAL),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="Individuals have an absolute right to erasure at any time",
                         is_material=True, score=0.5,
                         reasoning="Right exists but is subject to Article 17(3) exceptions"),
        ],
        expected_routing="RETRY",
    )

    # ── Case 3: Hallucinated specific — HARD_BLOCK ────────────────────────────
    c3 = AblationCase(
        name="C3_hipaa_fabricated_standard",
        query="What encryption standard does HIPAA mandate?",
        llm_answer=(
            "HIPAA mandates AES-256-GCM encryption with a 12-byte nonce for all "
            "ePHI at rest. This is specified in HIPAA Technical Safeguard § 164.312(e)(1)."
        ),
        rag_chunks=[
            "45 CFR § 164.312(a)(2)(iv): Entities must implement encryption where "
            "appropriate — no specific algorithm is mandated.",
            "HHS OCR FAQ: HIPAA is technology-neutral; it does not require a specific "
            "encryption standard.",
        ],
        claims=[
            Claim(claim_id=1, claim_text="HIPAA mandates AES-256-GCM with 12-byte nonce",
                  is_material=True,  confidence=0.80, verdict=Verdict.NOT_SUPPORTED),
            Claim(claim_id=2, claim_text="Standard is in § 164.312(e)(1)",
                  is_material=True,  confidence=0.70, verdict=Verdict.NOT_SUPPORTED),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="HIPAA mandates AES-256-GCM with 12-byte nonce",
                         is_material=True, score=0.0,
                         reasoning="HIPAA is technology-neutral — no such mandate exists"),
            JudgeVerdict(claim_id=2, claim_text="Standard is in § 164.312(e)(1)",
                         is_material=True, score=0.0,
                         reasoning="§ 164.312(e)(1) governs transmission security, not storage standard"),
        ],
        expected_routing="HARD_BLOCK",
    )

    # ── Case 4: Jurisdiction-blind — Basel III applied to US bank ─────────────
    c4 = AblationCase(
        name="C4_basel_jurisdiction_blind",
        query="Do US banks need to comply with Basel III liquidity requirements?",
        llm_answer=(
            "All banks worldwide must maintain a Liquidity Coverage Ratio (LCR) "
            "of 100% under Basel III, as defined by the Basel Committee."
        ),
        rag_chunks=[
            "Basel III framework (BIS): LCR requires banks to hold sufficient HQLA "
            "to survive a 30-day stress scenario. LCR ≥ 100%.",
            "US Federal Reserve rule (12 CFR Part 249): Large US BHCs subject to "
            "enhanced prudential standards must maintain LCR ≥ 100%. Smaller banks "
            "may have modified LCR requirements.",
        ],
        claims=[
            Claim(claim_id=1, claim_text="All banks worldwide must maintain LCR of 100%",
                  is_material=True,  confidence=0.70, verdict=Verdict.PARTIAL),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="All banks worldwide must maintain LCR of 100%",
                         is_material=True, score=0.5,
                         reasoning="Correct for large US BHCs but 'all banks worldwide' overstates — smaller banks have modified rules"),
        ],
        expected_routing="RETRY",
    )

    # ── Case 5: Low confidence / vague — HUMAN_REVIEW ────────────────────────
    c5 = AblationCase(
        name="C5_ccpa_vague_answer",
        query="What are the CCPA requirements for data deletion requests?",
        llm_answer="Companies must delete customer data when requested.",
        rag_chunks=[
            "CCPA § 1798.105: Consumers have the right to request deletion of personal "
            "information. Businesses must delete within 45 days and notify service providers.",
            "CCPA § 1798.105(d): Deletion right does not apply when data is necessary for "
            "completing a transaction, security purposes, or legal obligations.",
        ],
        claims=[
            Claim(claim_id=1, claim_text="Companies must delete data when requested",
                  is_material=False, confidence=0.50, verdict=Verdict.PARTIAL),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="Companies must delete data when requested",
                         is_material=False, score=0.5,
                         reasoning="True but omits 45-day deadline and § 1798.105(d) exceptions"),
        ],
        expected_routing="RETRY",
    )

    # ── Case 6: Fully correct GDPR consent answer ─────────────────────────────
    c6 = AblationCase(
        name="C6_gdpr_consent_correct",
        query="What makes consent valid under GDPR?",
        llm_answer=(
            "Under GDPR Article 7, valid consent must be freely given, specific, "
            "informed, and unambiguous. It must be as easy to withdraw consent as "
            "to give it. Pre-ticked boxes do not constitute valid consent."
        ),
        rag_chunks=[
            "GDPR Article 7: Conditions for consent — freely given, specific, "
            "informed and unambiguous indication of the data subject's wishes.",
            "GDPR Recital 32: Pre-ticked boxes or silence should not constitute "
            "consent. Withdrawal must be as easy as giving consent.",
            "EDPB Guidelines 05/2020: Consent requires a clear affirmative act.",
        ],
        claims=[
            Claim(claim_id=1, claim_text="Consent must be freely given, specific, informed, unambiguous",
                  is_material=True,  confidence=0.95, verdict=Verdict.SUPPORTED),
            Claim(claim_id=2, claim_text="Withdrawal must be as easy as giving consent",
                  is_material=True,  confidence=0.95, verdict=Verdict.SUPPORTED),
            Claim(claim_id=3, claim_text="Pre-ticked boxes are not valid consent",
                  is_material=False, confidence=0.90, verdict=Verdict.SUPPORTED),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="Consent must be freely given, specific, informed, unambiguous",
                         is_material=True, score=1.0, reasoning="Article 7 verbatim"),
            JudgeVerdict(claim_id=2, claim_text="Withdrawal must be as easy as giving consent",
                         is_material=True, score=1.0, reasoning="Article 7(3) confirmed"),
            JudgeVerdict(claim_id=3, claim_text="Pre-ticked boxes are not valid consent",
                         is_material=False, score=1.0, reasoning="GDPR Recital 32 confirmed"),
        ],
        expected_routing="DELIVER",
    )

    # ── Case 7: Hallucinated deadline — HARD_BLOCK ────────────────────────────
    c7 = AblationCase(
        name="C7_gdpr_fabricated_deadline",
        query="How quickly must organizations respond to GDPR data subject access requests?",
        llm_answer=(
            "Organizations must respond to data subject access requests within "
            "7 calendar days under GDPR Article 12."
        ),
        rag_chunks=[
            "GDPR Article 12(3): The controller shall provide information on action "
            "taken on a request within one month of receipt. May be extended by "
            "two further months where necessary.",
            "EDPB: The one-month period starts from receipt of the request.",
        ],
        claims=[
            Claim(claim_id=1, claim_text="Organizations must respond within 7 calendar days",
                  is_material=True,  confidence=0.80, verdict=Verdict.NOT_SUPPORTED),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="Organizations must respond within 7 calendar days",
                         is_material=True, score=0.0,
                         reasoning="GDPR Article 12(3) specifies one month, not 7 days — fabricated deadline"),
        ],
        expected_routing="HARD_BLOCK",
    )

    # ── Case 8: Partially correct — penalty for overstatement ─────────────────
    c8 = AblationCase(
        name="C8_hitech_breach_partial",
        query="What does HITECH require for breach notification?",
        llm_answer=(
            "HITECH requires immediate notification to affected individuals "
            "and HHS within 24 hours of discovering a breach of unsecured PHI."
        ),
        rag_chunks=[
            "45 CFR § 164.404: Covered entities must notify affected individuals "
            "without unreasonable delay and no later than 60 days after discovery.",
            "45 CFR § 164.408: HHS must be notified within 60 days for breaches "
            "affecting 500+ individuals; smaller breaches may be reported annually.",
        ],
        claims=[
            Claim(claim_id=1, claim_text="Notification to individuals must be within 24 hours",
                  is_material=True,  confidence=0.80, verdict=Verdict.NOT_SUPPORTED),
            Claim(claim_id=2, claim_text="HHS notification required within 24 hours",
                  is_material=True,  confidence=0.75, verdict=Verdict.NOT_SUPPORTED),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="Notification to individuals must be within 24 hours",
                         is_material=True, score=0.0,
                         reasoning="45 CFR § 164.404 requires 60 days, not 24 hours"),
            JudgeVerdict(claim_id=2, claim_text="HHS notification required within 24 hours",
                         is_material=True, score=0.0,
                         reasoning="45 CFR § 164.408 requires 60 days (or annual for small breaches)"),
        ],
        expected_routing="HARD_BLOCK",
    )

    return [c1, c2, c3, c4, c5, c6, c7, c8]


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING LOGIC (mirrors confidence/scorer.py)
# ─────────────────────────────────────────────────────────────────────────────

def _routing(final_score: float, hard_blocked: bool) -> str:
    from multi_agent.config import CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_LOW
    if hard_blocked:
        return "HARD_BLOCK"
    if final_score >= CONFIDENCE_THRESHOLD_HIGH:
        return "DELIVER"
    if final_score >= CONFIDENCE_THRESHOLD_LOW:
        return "RETRY"
    return "HUMAN_REVIEW"


def _judge_aggregate(judge_verdicts: List[JudgeVerdict]) -> float:
    """Weighted min-mean aggregate — mirrors ConfidenceScorer._aggregate_judge."""
    if not judge_verdicts:
        return 0.5
    mat  = [jv for jv in judge_verdicts if jv.is_material]
    nmat = [jv for jv in judge_verdicts if not jv.is_material]
    if mat and nmat:
        return round(0.70 * min(jv.score for jv in mat)
                     + 0.30 * (sum(jv.score for jv in nmat) / len(nmat)), 4)
    if mat:
        return round(min(jv.score for jv in mat), 4)
    return round(sum(jv.score for jv in nmat) / len(nmat), 4)


def _is_hard_blocked(case: AblationCase) -> bool:
    claim_map = {c.claim_id: c for c in case.claims}
    for jv in case.judge_verdicts:
        c = claim_map.get(jv.claim_id)
        if c and c.is_material and jv.score == 0.0:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SCORE COLLECTION — runs DeepEval once per case
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_F_LLM     = 0.75
_DEFAULT_H_LLM     = 0.25
_DEFAULT_RELEVANCY = 0.75


def _collect_scores(
    cases: List[AblationCase],
    no_deepeval: bool,
) -> List[ComponentScoreRecord]:
    """
    Compute component scores for each test case.

    In --no-deepeval mode: neutral defaults are used for F/H/R.
    In live mode: OllamaJudge + DeepEval metrics are called.
    """
    records: List[ComponentScoreRecord] = []

    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case.name}  ", end="", flush=True)
        judge_eval = _judge_aggregate(case.judge_verdicts)

        if no_deepeval or not case.rag_chunks:
            reason = "--no-deepeval" if no_deepeval else "no_rag_context"
            records.append(ComponentScoreRecord(
                case_name=case.name,
                f_llm=_DEFAULT_F_LLM, h_llm=_DEFAULT_H_LLM,
                relevancy=_DEFAULT_RELEVANCY, judge_eval=judge_eval,
                version="v0.1", error=reason,
            ))
            print(f"v0.1 (judge={judge_eval:.3f})")
            continue

        # Live DeepEval via Claude API
        try:
            from deepeval.metrics import (
                FaithfulnessMetric,
                HallucinationMetric,
                ContextualRelevancyMetric,
            )
            from deepeval.test_case import LLMTestCase
            from confidence.claude_judge import ClaudeJudge

            judge = ClaudeJudge()
            tc = LLMTestCase(
                input=case.query,
                actual_output=case.llm_answer,
                retrieval_context=case.rag_chunks,
                context=case.rag_chunks,
            )
            faith_m = FaithfulnessMetric(model=judge, threshold=0.5, include_reason=False)
            halluc_m = HallucinationMetric(model=judge, threshold=0.5, include_reason=False)
            relev_m  = ContextualRelevancyMetric(model=judge, threshold=0.5, include_reason=False)

            faith_m.measure(tc)
            halluc_m.measure(tc)
            relev_m.measure(tc)

            f_llm    = float(faith_m.score  or 0.0)
            h_llm    = float(halluc_m.score or 0.0)
            relevancy = float(relev_m.score or 0.0)

            records.append(ComponentScoreRecord(
                case_name=case.name,
                f_llm=f_llm, h_llm=h_llm, relevancy=relevancy,
                judge_eval=judge_eval, version="v2.0",
            ))
            print(f"v2.0  F={f_llm:.3f} H={h_llm:.3f} R={relevancy:.3f} J={judge_eval:.3f}")

        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            records.append(ComponentScoreRecord(
                case_name=case.name,
                f_llm=_DEFAULT_F_LLM, h_llm=_DEFAULT_H_LLM,
                relevancy=_DEFAULT_RELEVANCY, judge_eval=judge_eval,
                version="v0.1", error=err,
            ))
            print(f"v0.1 FALLBACK — {err[:60]}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT SWEEP — apply each config to pre-computed scores
# ─────────────────────────────────────────────────────────────────────────────

def _run_sweep(
    cases: List[AblationCase],
    scores: List[ComponentScoreRecord],
    configs: List[WeightConfig],
) -> List[CellResult]:
    """For each config × case, compute final_score and routing."""
    score_map: Dict[str, ComponentScoreRecord] = {s.case_name: s for s in scores}
    results: List[CellResult] = []

    for case in cases:
        sc = score_map[case.name]
        hard_blocked = _is_hard_blocked(case)

        for cfg in configs:
            final_score = cfg.apply(sc.f_llm, sc.h_llm, sc.relevancy, sc.judge_eval)
            routing     = _routing(final_score, hard_blocked)
            results.append(CellResult(
                config_name=cfg.name,
                case_name=case.name,
                final_score=final_score,
                routing_decision=routing,
                expected_routing=case.expected_routing,
                correct=(routing == case.expected_routing),
            ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _print_score_table(
    cases: List[AblationCase],
    configs: List[WeightConfig],
    results: List[CellResult],
) -> None:
    """Print a compact final_score table (configs × cases)."""
    # Build lookup: (config_name, case_name) → CellResult
    table: Dict[tuple, CellResult] = {
        (r.config_name, r.case_name): r for r in results
    }

    short_names = [c.name.split("_", 1)[0] + "_" + c.name.split("_")[1]
                   if "_" in c.name else c.name[:12]
                   for c in cases]

    col_w = 9
    name_w = 16

    # Header
    print("\n" + "=" * (name_w + col_w * len(cases) + 4))
    print("  ABLATION STUDY — FINAL SCORES  (HARD_BLOCK cases marked †)")
    print("=" * (name_w + col_w * len(cases) + 4))
    header = f"  {'Config':<{name_w}}" + "".join(f"{n[:col_w-1]:<{col_w}}" for n in short_names)
    print(header)
    print("  " + "-" * (name_w + col_w * len(cases)))

    for cfg in configs:
        row = f"  {cfg.name:<{name_w}}"
        for case in cases:
            r = table[(cfg.name, case.name)]
            marker = "†" if r.routing_decision == "HARD_BLOCK" else " "
            ok     = "√" if r.correct else "×"
            row   += f"{r.final_score:.3f}{marker}{ok}   "[:col_w]
        print(row)

    print("  " + "-" * (name_w + col_w * len(cases)))
    print(f"  Key: score  √=correct routing  ×=wrong routing  †=HARD_BLOCK")


def _print_accuracy_table(
    configs: List[WeightConfig],
    results: List[CellResult],
    n_cases: int,
) -> None:
    """Print routing accuracy per weight config."""
    print("\n" + "=" * 55)
    print("  ROUTING ACCURACY BY WEIGHT CONFIGURATION")
    print("=" * 55)
    print(f"  {'Config':<18}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>9}")
    print("  " + "-" * 44)

    by_config: Dict[str, int] = {}
    for r in results:
        by_config[r.config_name] = by_config.get(r.config_name, 0) + (1 if r.correct else 0)

    ranked = sorted(by_config.items(), key=lambda x: x[1], reverse=True)
    for name, correct in ranked:
        acc = correct / n_cases * 100
        bar = "#" * int(acc / 5)
        print(f"  {name:<18}  {correct:>7}  {n_cases:>5}  {acc:>8.1f}%  {bar}")

    print("=" * 55)


def _print_component_table(
    cases: List[AblationCase],
    scores: List[ComponentScoreRecord],
) -> None:
    """Print the component scores collected from DeepEval."""
    print("\n" + "=" * 72)
    print("  COMPONENT SCORES  (collected once per case, mode-independent)")
    print("=" * 72)
    print(f"  {'Case':<35} {'F_llm':>6} {'H_llm':>6} {'Relev':>6} {'Judge':>6}  {'Ver'}")
    print("  " + "-" * 68)
    for sc in scores:
        print(f"  {sc.case_name:<35} {sc.f_llm:>6.3f} {sc.h_llm:>6.3f} "
              f"{sc.relevancy:>6.3f} {sc.judge_eval:>6.3f}  {sc.version}")
    print("=" * 72)


def _save_results(
    cases: List[AblationCase],
    configs: List[WeightConfig],
    scores: List[ComponentScoreRecord],
    results: List[CellResult],
    timestamp: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save full results to JSON and summary to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON: full raw data ────────────────────────────────────────────────────
    json_path = output_dir / f"ablation_results_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "configs": [{"name": c.name, "w_f": c.w_f, "w_h": c.w_h, "w_r": c.w_r, "w_j": c.w_j}
                    for c in configs],
        "component_scores": [asdict(s) for s in scores],
        "cell_results": [asdict(r) for r in results],
        "routing_accuracy": {
            cfg.name: sum(1 for r in results if r.config_name == cfg.name and r.correct) / len(cases)
            for cfg in configs
        },
    }
    json_path.write_text(json.dumps(payload, indent=2))

    # ── CSV: pivot table — rows = configs, columns = cases ────────────────────
    csv_path = output_dir / f"ablation_summary_{timestamp}.csv"
    table: Dict[tuple, CellResult] = {(r.config_name, r.case_name): r for r in results}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config"] + [c.name for c in cases] + ["accuracy"])
        for cfg in configs:
            row = [cfg.name]
            correct_count = 0
            for case in cases:
                r = table[(cfg.name, case.name)]
                row.append(f"{r.final_score:.4f} ({r.routing_decision})")
                if r.correct:
                    correct_count += 1
            row.append(f"{correct_count / len(cases) * 100:.1f}%")
            writer.writerow(row)
        # expected routing row
        writer.writerow(["EXPECTED"] + [c.expected_routing for c in cases] + [""])

    return json_path, csv_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSE Formula Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--no-deepeval",   action="store_true",
                        help="Skip DeepEval — use neutral defaults for F/H/R (fast mode)")
    parser.add_argument("--load-scores",   metavar="FILE",
                        help="Load pre-computed component scores from a JSON file")
    parser.add_argument("--extra-configs", metavar="FILE",
                        help="JSON file with extra weight configs to include")
    parser.add_argument("--output-dir",    default="ablation_outputs",
                        help="Directory for result files (default: ablation_outputs/)")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)

    cases = _make_test_cases()
    configs = list(WEIGHT_CONFIGS)

    # Load extra configs if provided
    if args.extra_configs:
        extra_path = Path(args.extra_configs)
        if not extra_path.exists():
            print(f"[ERROR] extra-configs file not found: {extra_path}")
            sys.exit(1)
        for entry in json.loads(extra_path.read_text()):
            configs.append(WeightConfig(**entry))

    mode = "v0.1 (neutral defaults)" if args.no_deepeval else "v2.0 (live DeepEval)"
    print(f"\n{'='*70}")
    print(f"  CSE FORMULA ABLATION STUDY")
    print(f"  Mode    : {mode}")
    print(f"  Cases   : {len(cases)}")
    print(f"  Configs : {len(configs)}")
    print(f"  Output  : {output_dir}/")
    print(f"{'='*70}\n")

    # ── Step 1: collect component scores ──────────────────────────────────────
    if args.load_scores:
        load_path = Path(args.load_scores)
        print(f"[1/3] Loading component scores from {load_path} ...")
        raw = json.loads(load_path.read_text())
        scores = [ComponentScoreRecord(**s) for s in raw["component_scores"]]
    else:
        print(f"[1/3] Collecting component scores ({mode}) ...")
        scores = _collect_scores(cases, no_deepeval=args.no_deepeval)

    _print_component_table(cases, scores)

    # Save component scores early so they can be reloaded with --load-scores
    scores_path = output_dir / f"ablation_scores_{timestamp}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps({
        "timestamp": timestamp,
        "component_scores": [asdict(s) for s in scores],
    }, indent=2))
    print(f"\n  Component scores saved → {scores_path}")
    print(f"  (Use --load-scores {scores_path} to re-sweep weights without re-running DeepEval)\n")

    # ── Step 2: weight sweep ───────────────────────────────────────────────────
    print(f"[2/3] Running weight sweep ({len(configs)} configs × {len(cases)} cases) ...")
    results = _run_sweep(cases, scores, configs)

    # ── Step 3: print results ─────────────────────────────────────────────────
    print(f"\n[3/3] Results\n")
    _print_score_table(cases, configs, results)
    _print_accuracy_table(configs, results, len(cases))

    # ── Save output files ─────────────────────────────────────────────────────
    json_path, csv_path = _save_results(
        cases, configs, scores, results, timestamp, output_dir
    )
    print(f"\n  Results saved:")
    print(f"    JSON (full data) → {json_path}")
    print(f"    CSV  (pivot)     → {csv_path}")
    print()


if __name__ == "__main__":
    main()
