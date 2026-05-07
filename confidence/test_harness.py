"""
confidence/test_harness.py — CSE test harness.

Tests the full Confidence Scoring Engine against synthetic MAD outputs
to verify the formula produces sensible, calibrated scores.

Run from repo root:
    PYTHONPATH=multi_agent_debate:rag_folder:. python confidence/test_harness.py

Flags:
    --no-deepeval   Skip DeepEval metrics (judge-only v0.1 mode)
    --verbose       Print full CSE breakdown per case

Each TestCase carries two sets of expectations:
    expected_routing / expected_range      — used in v0.1 mode (--no-deepeval)
    expected_routing_v2 / expected_range_v2 — used in full DeepEval v2.0 mode

If expected_routing_v2 is None it falls back to expected_routing (covers
cases like HARD_BLOCK where mode doesn't change the outcome).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure all package roots are on sys.path
_repo = Path(__file__).resolve().parent.parent
for _p in [str(_repo), str(_repo / "multi_agent_debate"), str(_repo / "rag_folder")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_repo / ".env")

from multi_agent.models import Claim, JudgeVerdict, EvidenceChunk, Verdict

# ── Synthetic test cases ──────────────────────────────────────────────────────

@dataclass
class TestCase:
    name:           str
    query:          str
    llm_answer:     str
    rag_chunks:     List[str]
    claims:         List[Claim]
    judge_verdicts: List[JudgeVerdict]
    # v0.1 (--no-deepeval) expectations
    expected_routing: str                    # DELIVER / RETRY / HUMAN_REVIEW / HARD_BLOCK
    expected_range:   Tuple[float, float]    # (min, max) for final_score
    # v2.0 (live DeepEval) expectations — None means fall back to v0.1 values above
    expected_routing_v2:  Optional[str]              = field(default=None)
    expected_range_v2:    Optional[Tuple[float, float]] = field(default=None)


TEST_CASES: List[TestCase] = [
    # ── Case 1: High confidence — all claims well-supported ───────────────────
    TestCase(
        name="High-confidence HIPAA answer",
        query="What does HIPAA require for PHI encryption?",
        llm_answer=(
            "HIPAA requires covered entities to implement technical safeguards "
            "for PHI. The Security Rule mandates encryption of PHI at rest and "
            "in transit when it is deemed appropriate based on a risk analysis. "
            "AES-256 is the de facto standard for PHI encryption in healthcare."
        ),
        rag_chunks=[
            "45 CFR § 164.312(a)(2)(iv): Covered entities must implement encryption "
            "and decryption mechanisms for electronic PHI where appropriate.",
            "NIST SP 800-111: AES-256 is recommended for health data encryption at rest.",
            "HHS Guidance on Encryption: Proper encryption renders PHI unusable, "
            "unreadable, or indecipherable to unauthorized individuals.",
        ],
        claims=[
            Claim(claim_id=1, claim_text="HIPAA requires technical safeguards for PHI",
                  is_material=True,  confidence=0.95, verdict=Verdict.SUPPORTED),
            Claim(claim_id=2, claim_text="Security Rule mandates encryption based on risk analysis",
                  is_material=True,  confidence=0.90, verdict=Verdict.SUPPORTED),
            Claim(claim_id=3, claim_text="AES-256 is the de facto standard for PHI encryption",
                  is_material=False, confidence=0.80, verdict=Verdict.SUPPORTED),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="HIPAA requires technical safeguards for PHI",
                         is_material=True,  score=1.0, reasoning="Directly stated in 45 CFR § 164.312"),
            JudgeVerdict(claim_id=2, claim_text="Security Rule mandates encryption based on risk analysis",
                         is_material=True,  score=1.0, reasoning="Confirmed by HHS guidance"),
            JudgeVerdict(claim_id=3, claim_text="AES-256 is the de facto standard for PHI encryption",
                         is_material=False, score=0.5, reasoning="Industry standard but not mandated"),
        ],
        # JUDGE_ONLY_FALLBACK: judge_eval=0.85 re-normalised → score ~0.87 → DELIVER
        # FULL (live DeepEval): high faithfulness from well-supported context → DELIVER
        expected_routing="DELIVER",
        expected_range=(0.75, 1.0),
        expected_routing_v2="DELIVER",
        expected_range_v2=(0.80, 1.0),
    ),

    # ── Case 2: Partial support — one material claim uncertain ────────────────
    TestCase(
        name="Partial support — jurisdiction ambiguity",
        query="Is GDPR Article 17 right to erasure absolute?",
        llm_answer=(
            "Under GDPR Article 17, individuals have an absolute right to erasure "
            "of their personal data when they withdraw consent. "
            "Organizations must delete all copies within 24 hours of the request."
        ),
        rag_chunks=[
            "GDPR Article 17: The data subject shall have the right to obtain from the "
            "controller the erasure of personal data concerning him or her. "
            "This right is subject to exceptions under Article 17(3).",
            "Article 17(3): The right to erasure does not apply where processing is "
            "necessary for compliance with a legal obligation or for public interest.",
        ],
        claims=[
            Claim(claim_id=1, claim_text="GDPR Article 17 grants an absolute right to erasure",
                  is_material=True,  confidence=0.70, verdict=Verdict.PARTIAL),
            Claim(claim_id=2, claim_text="Right applies when consent is withdrawn",
                  is_material=True,  confidence=0.85, verdict=Verdict.SUPPORTED),
            Claim(claim_id=3, claim_text="Organizations must delete within 24 hours",
                  is_material=True,  confidence=0.60, verdict=Verdict.NOT_SUPPORTED),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="GDPR Article 17 grants an absolute right to erasure",
                         is_material=True,  score=0.0,
                         reasoning="Incorrect — Article 17(3) contains substantial exceptions"),
            JudgeVerdict(claim_id=2, claim_text="Right applies when consent is withdrawn",
                         is_material=True,  score=1.0, reasoning="Confirmed by Article 17(1)(b)"),
            JudgeVerdict(claim_id=3, claim_text="Organizations must delete within 24 hours",
                         is_material=True,  score=0.0, reasoning="No 24-hour deadline in GDPR"),
        ],
        # HARD_BLOCK is mode-independent: is_material claim scored 0.0 always blocks
        expected_routing="HARD_BLOCK",
        expected_range=(0.0, 0.9),
        expected_routing_v2="HARD_BLOCK",
        expected_range_v2=(0.0, 0.9),
    ),

    # ── Case 3: Low confidence — vague answer with weak context ───────────────
    TestCase(
        name="Low-confidence vague answer",
        query="What does Basel III require for liquidity?",
        llm_answer="Banks must maintain some level of liquid assets to meet obligations.",
        rag_chunks=["Basel III introduced the Liquidity Coverage Ratio (LCR) requirement."],
        claims=[
            Claim(claim_id=1, claim_text="Banks must maintain liquid assets",
                  is_material=False, confidence=0.50, verdict=Verdict.PARTIAL),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="Banks must maintain liquid assets",
                         is_material=False, score=0.5,
                         reasoning="Partially correct but too vague — no specific LCR mention"),
        ],
        # JUDGE_ONLY_FALLBACK: judge_eval=0.5 → score ~0.49 → HUMAN_REVIEW (below 0.55 threshold)
        # FULL (live DeepEval): weak context + vague answer → low scores → RETRY or HUMAN_REVIEW
        expected_routing="HUMAN_REVIEW",
        expected_range=(0.2, 0.8),
        expected_routing_v2="RETRY",
        expected_range_v2=(0.2, 0.8),
    ),

    # ── Case 4: No context (tests neutral defaults path) ──────────────────────
    TestCase(
        name="No RAG context (neutral defaults)",
        query="What is the CCPA threshold for covered businesses?",
        llm_answer="CCPA applies to businesses with annual gross revenue over $25 million.",
        rag_chunks=[],
        claims=[
            Claim(claim_id=1, claim_text="CCPA threshold is $25 million gross revenue",
                  is_material=True, confidence=0.80, verdict=Verdict.SUPPORTED),
        ],
        judge_verdicts=[
            JudgeVerdict(claim_id=1, claim_text="CCPA threshold is $25 million gross revenue",
                         is_material=True, score=1.0, reasoning="Correct per CCPA Section 1798.140"),
        ],
        # No context → DeepEval skipped in both modes (neutral defaults used)
        # Routing depends entirely on judge_eval = 1.0 → DELIVER in both modes
        expected_routing="DELIVER",
        expected_range=(0.5, 1.0),
        expected_routing_v2="DELIVER",
        expected_range_v2=(0.5, 1.0),
    ),
]


# ── Harness runner ─────────────────────────────────────────────────────────────

def run_harness(no_deepeval: bool = False, verbose: bool = False) -> None:
    from confidence.scorer import ConfidenceScorer

    if no_deepeval:
        # Monkey-patch _run_deepeval to simulate DeepEval being unavailable.
        # Returns (None, None, None, reason) — scorer falls back to JUDGE_ONLY_FALLBACK.
        def _neutral(self, *args, **kwargs):
            return None, None, None, "no_deepeval_flag"
        ConfidenceScorer._run_deepeval = _neutral

    scorer   = ConfidenceScorer()
    passed   = 0
    failed   = 0
    failures = []

    mode_label = "v0.1  (judge-only / --no-deepeval)" if no_deepeval else "v2.0  (live DeepEval)"

    header = "=" * 70
    print(f"\n{header}")
    print("  CONFIDENCE SCORING ENGINE — TEST HARNESS")
    print(f"  Mode: {mode_label}")
    print(header)

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {tc.name}")
        print(f"  Query     : {tc.query[:80]}")

        result = scorer.score(
            query          = tc.query,
            llm_answer     = tc.llm_answer,
            rag_chunks     = tc.rag_chunks,
            final_claims   = tc.claims,
            judge_verdicts = tc.judge_verdicts,
        )

        if verbose:
            print(f"  CSE result: {result}")
            print(f"  Components: {result.components.as_dict()}")

        # ── Select the right expectations for the current mode ────────────────
        if no_deepeval or tc.expected_routing_v2 is None:
            exp_routing = tc.expected_routing
            exp_range   = tc.expected_range
        else:
            exp_routing = tc.expected_routing_v2
            exp_range   = tc.expected_range_v2

        # ── Assertions ────────────────────────────────────────────────────────
        score_ok   = exp_range[0] <= result.final_score <= exp_range[1]
        routing_ok = result.routing_decision == exp_routing

        status = "PASS" if (score_ok and routing_ok) else "FAIL"
        print(f"  Score     : {result.final_score:.4f}  "
              f"(expected {exp_range[0]}–{exp_range[1]})  "
              f"{'OK' if score_ok else 'FAIL'}")
        print(f"  Routing   : {result.routing_decision}  "
              f"(expected {exp_routing})  "
              f"{'OK' if routing_ok else 'FAIL'}")
        print(f"  Version   : {result.version}")

        # ── DeepEval value sanity check in FULL mode ──────────────────────────
        from confidence.cse_types import ScoringMode
        if not no_deepeval and result.scoring_mode == ScoringMode.FULL.value:
            comps = result.components
            stuck = []
            if comps.f_llm in (0.0, 1.0):
                stuck.append(f"F_llm={comps.f_llm}")
            if comps.h_llm in (0.0, 1.0):
                stuck.append(f"H_llm={comps.h_llm}")
            if comps.relevancy in (0.0, 1.0):
                stuck.append(f"relevancy={comps.relevancy}")
            if stuck:
                print(f"  [WARN]  Possibly stuck DeepEval scores: {', '.join(stuck)}")

        print(f"  [{status}]")

        if score_ok and routing_ok:
            passed += 1
        else:
            failed += 1
            failures.append(f"Case {i}: {tc.name}")

    print(f"\n{header}")
    print(f"  RESULTS: {passed}/{len(TEST_CASES)} passed  (mode: {mode_label})")
    if failures:
        print("  FAILURES:")
        for f in failures:
            print(f"    - {f}")
    print(header + "\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSE test harness")
    parser.add_argument("--no-deepeval", action="store_true",
                        help="Skip DeepEval metrics (judge-only v0.1 mode)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full CSE breakdown for each case")
    args = parser.parse_args()
    run_harness(no_deepeval=args.no_deepeval, verbose=args.verbose)
