"""
run_test.py — CLI test runner for the MAD pipeline.

Usage:
  # Run with the built-in GDPR example (quick sanity check):
  python -m multi_agent.run_test

  # Run with a custom query and answer:
  python -m multi_agent.run_test \\
    --query "Does HIPAA require encryption of ePHI at rest?" \\
    --answer "Yes, HIPAA mandates AES-256 encryption for all ePHI at rest under 45 CFR 164.312."

  # Save full JSON output:
  python -m multi_agent.run_test --output-json output.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from multi_agent.mad_pipeline import run_mad

# ── Built-in test cases ───────────────────────────────────────────────────────

EXAMPLES = {
    "gdpr_encryption": {
        "query": "Does GDPR require us to encrypt customer data at rest?",
        "answer": (
            "Yes. GDPR Article 32 explicitly mandates encryption of personal data at rest. "
            "The regulation specifies AES-256 as the required encryption standard. "
            "Organizations that fail to encrypt data at rest face fines up to 4% of global "
            "annual turnover."
        ),
        "description": "Classic example — contains 1 accurate claim, 1 partial, 1 hallucinated standard",
    },
    "hipaa_phi": {
        "query": "Does HIPAA require encryption of PHI stored on servers?",
        "answer": (
            "HIPAA requires all covered entities to encrypt PHI at rest using FIPS-140-2 "
            "approved encryption. Failure to encrypt is a direct HIPAA violation and results "
            "in automatic fines of $50,000 per incident."
        ),
        "description": "HIPAA example — tests addressable vs required distinction",
    },
    "ccpa_deletion": {
        "query": "Does CCPA give customers the right to delete their data?",
        "answer": (
            "Yes, CCPA grants California residents the absolute right to delete all personal "
            "information held by any business, without any exceptions. Businesses must delete "
            "all data within 24 hours of a deletion request."
        ),
        "description": "CCPA example — tests absolutist claims and exception detection",
    },
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Guardrails Gateway MAD pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query",       default=None, help="Custom query string")
    parser.add_argument("--answer",      default=None, help="Custom LLM answer string")
    parser.add_argument("--example",     default="gdpr_encryption",
                        choices=list(EXAMPLES.keys()),
                        help="Which built-in example to run (default: gdpr_encryption)")
    parser.add_argument("--output-json", default=None, metavar="FILE",
                        help="Save full MADOutput to a JSON file")
    parser.add_argument("--list-examples", action="store_true",
                        help="List all built-in examples and exit")
    args = parser.parse_args()

    if args.list_examples:
        print("\nBuilt-in examples:")
        for name, ex in EXAMPLES.items():
            print(f"  --example {name}")
            print(f"    {ex['description']}")
            print(f"    Query:  {ex['query']}")
            print()
        sys.exit(0)

    # Use custom or built-in
    if args.query and args.answer:
        query      = args.query
        llm_answer = args.answer
        print(f"\nUsing custom query and answer.")
    else:
        ex         = EXAMPLES[args.example]
        query      = ex["query"]
        llm_answer = ex["answer"]
        print(f"\nRunning built-in example: '{args.example}'")
        print(f"Description: {ex['description']}")

    print(f"\n{'─'*70}")
    print(f"Query:  {query}")
    print(f"Answer: {llm_answer}")
    print(f"{'─'*70}")

    # ── Run pipeline ──
    started = datetime.now()
    result  = run_mad(query=query, llm_answer=llm_answer)
    elapsed = (datetime.now() - started).total_seconds()

    # ── Print results ──
    W = 70
    print(f"\n{'='*W}")
    print("  FINAL RESULTS")
    print(f"{'='*W}")
    print(f"  Routing decision    : {result.routing_decision}")
    print(f"  Aggregate confidence: {result.aggregate_confidence:.4f}")
    print(f"  Time elapsed        : {elapsed:.1f}s")
    print(f"  Debate cycles run   : {len(result.debate_cycles)}")
    print(f"  Evidence pool size  : {len(result.evidence_pool)} chunks")

    print(f"\n  Claims extracted: {len(result.claims)}")
    for c in result.claims:
        mat = "⚠ material" if c.is_material else "  context "
        v   = c.verdict.value if c.verdict else "PENDING"
        print(f"    [{mat}] C{c.claim_id}: {v} (p={c.confidence:.2f})")
        print(f"             \"{c.claim_text[:85]}\"")

    print(f"\n  Judge verdicts:")
    for jv in result.judge_verdicts:
        mat        = "⚠ material" if jv.is_material else "  context "
        score_icon = {1.0: "✅", 0.5: "⚠️ ", 0.0: "❌"}.get(jv.score, "?")
        print(f"    {score_icon} [{mat}] C{jv.claim_id}: score={jv.score}")
        print(f"              \"{jv.claim_text[:85]}\"")
        print(f"              {jv.reasoning[:120]}")

    if result.correction_signal:
        print(f"\n  Correction signal:")
        # Word-wrap at 68 chars
        words = result.correction_signal.split()
        line  = "    "
        for w in words:
            if len(line) + len(w) + 1 > 72:
                print(line)
                line = "    " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(line)

    print(f"\n{'─'*W}")
    print("  FULL DEBATE TRANSCRIPT")
    print(f"{'─'*W}")
    print(result.debate_transcript)

    # ── Optional JSON output ──
    if args.output_json:
        out_path = args.output_json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        print(f"\n  Full MADOutput saved → {out_path}")


if __name__ == "__main__":
    main()
