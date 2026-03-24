"""
run_test.py — CLI test runner for the MAD pipeline
====================================================

Tests the full pipeline end-to-end including storage writes.
Uses Healthcare domain examples (your paper's primary eval domain).

Usage:
  # Default HIPAA encryption example:
  python -m multi_agent.run_test

  # Choose a specific example:
  python -m multi_agent.run_test --example hipaa_phi_sharing

  # Custom query:
  python -m multi_agent.run_test \\
    --query "Can we email patient lab results directly?" \\
    --answer "Yes, HIPAA allows emailing lab results to patients directly."

  # Save full JSON output:
  python -m multi_agent.run_test --output-json results.json

  # Verify storage was written correctly (checks SQLite tables):
  python -m multi_agent.run_test --verify-storage

  # List all examples:
  python -m multi_agent.run_test --list
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from multi_agent.mad_pipeline import run_mad
from multi_agent.config import DB_PATH

# ── Healthcare test cases ──────────────────────────────────────────────────────
# These mirror your 200-example synthetic evaluation dataset structure.
# All queries are realistic enterprise employee questions.
# LLM answers contain deliberate error types for testing.
EXAMPLES = {
    "hipaa_encryption": {
        "query": "Does HIPAA require us to encrypt ePHI stored on our servers?",
        "answer": (
            "Yes. HIPAA mandates encryption of all electronic protected health "
            "information at rest using AES-256 encryption. Any covered entity that "
            "fails to encrypt ePHI is in direct violation of the HIPAA Security Rule "
            "and faces automatic penalties of up to $50,000 per violation."
        ),
        "error_type": "hallucinated_specific",
        "description": (
            "LLM hallucinated: (1) encryption is mandatory under HIPAA "
            "(it's addressable, not required), (2) AES-256 is specified "
            "(not named in HIPAA), (3) $50k penalty is automatic "
            "(penalties depend on culpability tier). Should HARD_BLOCK."
        ),
    },
    "hipaa_phi_sharing": {
        "query": (
            "Can we share patient records with our insurance billing partner "
            "if we have a signed Business Associate Agreement?"
        ),
        "answer": (
            "Yes. With a signed Business Associate Agreement (BAA) in place, "
            "you can share patient PHI with any business partner for any "
            "business purpose. The BAA fully authorises the data sharing "
            "under HIPAA."
        ),
        "error_type": "missing_caveat",
        "description": (
            "LLM missed: BAA authorises sharing for permitted purposes only "
            "(treatment, payment, healthcare operations). "
            "Marketing or other uses require individual patient authorisation "
            "even with a BAA. Should RETRY."
        ),
    },
    "hipaa_patient_access": {
        "query": "How quickly must we respond to a patient's request for their medical records?",
        "answer": (
            "Under HIPAA, covered entities must provide patients access to "
            "their medical records within 30 days of the request. "
            "You may request one 30-day extension if needed. "
            "Patients have the right to receive records in electronic format "
            "if they request it."
        ),
        "error_type": "fully_correct",
        "description": (
            "LLM answer is correct: 30-day timeline, one extension allowed, "
            "electronic format right. Should DELIVER."
        ),
    },
    "hipaa_breach_notification": {
        "query": "When does HIPAA require us to notify patients of a data breach?",
        "answer": (
            "HIPAA requires notification to affected patients within 60 days "
            "of discovering a breach. For breaches affecting more than 500 "
            "patients in a state, you must also notify prominent media outlets "
            "and HHS simultaneously. Small breaches under 500 patients only "
            "require annual reporting to HHS."
        ),
        "error_type": "fully_correct",
        "description": (
            "LLM answer is accurate. Should DELIVER or RETRY "
            "(may catch nuance about 'discovery' clock start)."
        ),
    },
    "hipaa_employee_health_records": {
        "query": (
            "Can our HR manager access employee health records to track "
            "which staff are on sick leave?"
        ),
        "answer": (
            "Yes. HR managers have legitimate business access to employee "
            "health records for workforce management purposes. HIPAA does "
            "not restrict employer access to employee health information "
            "when used for internal HR decisions."
        ),
        "error_type": "jurisdiction_blind",
        "description": (
            "LLM is wrong: HIPAA does not cover employment records. "
            "But ADA Title I restricts employer access to disability-related "
            "information and requires medical records be kept separate. "
            "The answer is dangerously misleading. Should HARD_BLOCK or RETRY."
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guardrails Gateway MAD pipeline test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--example",        default="hipaa_encryption",
                        choices=list(EXAMPLES.keys()))
    parser.add_argument("--query",          default=None)
    parser.add_argument("--answer",         default=None)
    parser.add_argument("--output-json",    default=None, metavar="FILE")
    parser.add_argument("--verify-storage", action="store_true")
    parser.add_argument("--list",           action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nHealthcare test examples:")
        for name, ex in EXAMPLES.items():
            print(f"\n  --example {name}")
            print(f"    Error type : {ex['error_type']}")
            print(f"    Description: {ex['description']}")
        sys.exit(0)

    # Resolve query + answer
    if args.query and args.answer:
        query, llm_answer = args.query, args.answer
        print("\nRunning custom query.")
    else:
        ex        = EXAMPLES[args.example]
        query     = ex["query"]
        llm_answer = ex["answer"]
        print(f"\nExample    : {args.example}")
        print(f"Error type : {ex['error_type']}")
        print(f"Description: {ex['description']}")

    print(f"\n{'─'*68}")
    print(f"Query : {query}")
    print(f"Answer: {llm_answer[:120]}{'...' if len(llm_answer) > 120 else ''}")
    print(f"{'─'*68}")

    # ── Run pipeline ───────────────────────────────────────────────────────────
    started = datetime.now()
    result  = run_mad(query=query, llm_answer=llm_answer)
    elapsed = (datetime.now() - started).total_seconds()

    # ── Print results ──────────────────────────────────────────────────────────
    W = 68
    print(f"\n{'='*W}")
    print("  RESULTS")
    print(f"{'='*W}")
    print(f"  Routing decision    : {result.routing_decision}")
    print(f"  Aggregate confidence: {result.aggregate_confidence:.4f}")
    print(f"  Time elapsed        : {elapsed:.1f}s")
    print(f"  Debate cycles       : {len(result.debate_cycles)}")
    print(f"  Evidence pool       : {len(result.evidence_pool)} chunks")
    print(f"  Storage IDs         : query_id={result.query_id[:8]}... "
          f"rollout_id={result.rollout_id[:8]}...")

    print(f"\n  Claims ({len(result.claims)}):")
    for c in result.claims:
        mat = "⚠ material" if c.is_material else "  context "
        v   = c.verdict.value if c.verdict else "PENDING"
        print(f"    [{mat}] C{c.claim_id}: {v} (p={c.confidence:.2f})")
        print(f"             \"{c.claim_text[:82]}\"")

    print(f"\n  Judge verdicts:")
    icon = {1.0: "✅", 0.5: "⚠️ ", 0.0: "❌"}
    for jv in result.judge_verdicts:
        mat = "⚠ material" if jv.is_material else "  context "
        print(f"    {icon.get(jv.score,'?')} [{mat}] C{jv.claim_id}: v={jv.score}")
        print(f"              \"{jv.claim_text[:80]}\"")
        print(f"              {jv.reasoning[:120]}")

    if result.correction_signal:
        print(f"\n  Correction signal:")
        words, line = result.correction_signal.split(), "    "
        for w in words:
            if len(line) + len(w) + 1 > 72:
                print(line)
                line = "    " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(line)

    print(f"\n{'─'*W}\n  DEBATE TRANSCRIPT\n{'─'*W}")
    print(result.debate_transcript)

    # ── Verify storage ─────────────────────────────────────────────────────────
    if args.verify_storage:
        _verify_storage(result.query_id, result.rollout_id)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        print(f"\n  Full MADOutput saved → {args.output_json}")


def _verify_storage(query_id: str, rollout_id: str) -> None:
    """
    Verify that all 4 tables were written correctly for this run.
    This is what you run after your first 5 test queries to confirm
    the storage is working before showing your friend.
    """
    db = Path(DB_PATH)
    if not db.exists():
        print(f"\n  [Storage check] Database not found at {DB_PATH}")
        return

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    print(f"\n{'─'*68}")
    print("  STORAGE VERIFICATION")
    print(f"{'─'*68}")

    # Table 1: queries
    rows = con.execute(
        "SELECT * FROM queries WHERE query_id=? AND rollout_id=?",
        (query_id, rollout_id)
    ).fetchall()
    print(f"\n  queries table: {len(rows)} row(s)")
    for r in rows:
        print(f"    routing_decision: {r['routing_decision']}")
        print(f"    final_cse_score : {r['final_cse_score']}")

    # Table 2: claims — should be 3 rows per claim (3 checkpoints)
    rows = con.execute(
        "SELECT claim_id, checkpoint, confidence_p, verdict FROM claims "
        "WHERE query_id=? AND rollout_id=? ORDER BY claim_id, record_id",
        (query_id, rollout_id)
    ).fetchall()
    print(f"\n  claims table: {len(rows)} row(s) (expect 3 per claim)")
    for r in rows:
        print(f"    C{r['claim_id']} [{r['checkpoint']:12s}] "
              f"p={r['confidence_p']:.2f}  {r['verdict']}")
    # Verify agent_a_prompt is stored
    prompt_rows = con.execute(
        "SELECT claim_id, checkpoint, length(agent_a_prompt) as prompt_len "
        "FROM claims WHERE query_id=? AND rollout_id=?",
        (query_id, rollout_id)
    ).fetchall()
    print(f"\n  agent_a_prompt lengths (should be >0 for all rows):")
    for r in prompt_rows:
        ok = "✅" if r["prompt_len"] > 0 else "❌"
        print(f"    {ok} C{r['claim_id']} [{r['checkpoint']:12s}] "
              f"{r['prompt_len']} chars")

    # Table 3: attacks
    rows = con.execute(
        "SELECT claim_id, cycle, b_challenge_type, p_before_attack, "
        "p_after_attack, b_reward FROM attacks "
        "WHERE query_id=? AND rollout_id=?",
        (query_id, rollout_id)
    ).fetchall()
    print(f"\n  attacks table: {len(rows)} row(s)")
    for r in rows:
        p_after = f"{r['p_after_attack']:.2f}" if r["p_after_attack"] else "NULL"
        reward  = str(r["b_reward"]) if r["b_reward"] is not None else "NULL (expected)"
        print(f"    C{r['claim_id']} cycle={r['cycle']}  "
              f"{r['b_challenge_type'][:20]:20s}  "
              f"p_before={r['p_before_attack']:.2f}  "
              f"p_after={p_after}  b_reward={reward}")

    # Table 4: judge_verdicts
    rows = con.execute(
        "SELECT claim_id, v_label FROM judge_verdicts "
        "WHERE query_id=? AND rollout_id=?",
        (query_id, rollout_id)
    ).fetchall()
    print(f"\n  judge_verdicts table: {len(rows)} row(s)")
    for r in rows:
        print(f"    C{r['claim_id']}  v_label={r['v_label']}")

    con.close()
    print(f"\n  Database path: {DB_PATH}")
    print("  Storage verification complete.")


if __name__ == "__main__":
    main()
