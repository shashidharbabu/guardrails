#!/usr/bin/env python3
"""
scripts/cse_e2e_smoke.py — End-to-end CSE smoke test.

What this does:
  1. Runs one real query end-to-end through the full MAD+CSE pipeline
     (requires Ollama running with qwen2.5:7b).
  2. Reads the written row back from SQLite and verifies all 5 CSE columns
     are populated with non-null values.
  3. Prints a full breakdown of the CSE result.

Prerequisites:
    ollama pull qwen2.5:7b
    ollama serve                   # in a separate terminal

Run from repo root:
    python scripts/cse_e2e_smoke.py

Optional flags:
    --query   "Your custom query"
    --answer  "LLM answer to verify"
    --verbose  Print the full MAD transcript

Exit codes:
    0  — CSE scores written successfully
    1  — Pipeline errored or CSE columns are NULL after the run
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
for _p in [str(_REPO), str(_REPO / "multi_agent_debate"), str(_REPO / "rag_folder")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_REPO / ".env")

# ── Default test query ─────────────────────────────────────────────────────────
# A realistic Healthcare query with a good answer that DeepEval should score high.
_DEFAULT_QUERY = (
    "Does HIPAA require covered entities to encrypt ePHI at rest?"
)
_DEFAULT_ANSWER = (
    "Yes. Under the HIPAA Security Rule (45 CFR § 164.312(a)(2)(iv)), "
    "covered entities must implement a mechanism to encrypt and decrypt "
    "electronic Protected Health Information (ePHI) where determined "
    "appropriate through a risk analysis. While HIPAA does not mandate "
    "a specific encryption algorithm, NIST recommends AES-128 or AES-256 "
    "for data at rest. Failure to encrypt ePHI when risks are identified "
    "constitutes a Security Rule violation."
)


def _read_cse_row(db_path: str, query_id: str, rollout_id: str) -> dict | None:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM queries WHERE query_id = ? AND rollout_id = ?",
        (query_id, rollout_id),
    ).fetchone()
    con.close()
    return dict(row) if row else None


def _check_cse_columns(row: dict) -> tuple[bool, list[str]]:
    """Return (all_present, list_of_nulls)."""
    required = ["final_cse_score", "routing_decision", "cse_f_llm",
                "cse_h_llm", "cse_relevancy", "cse_judge_eval", "cse_version"]
    nulls = [c for c in required if row.get(c) is None]
    return len(nulls) == 0, nulls


def main() -> None:
    parser = argparse.ArgumentParser(description="CSE end-to-end smoke test")
    parser.add_argument("--query",   default=_DEFAULT_QUERY,  help="Query to run")
    parser.add_argument("--answer",  default=_DEFAULT_ANSWER, help="LLM answer to verify")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full transcript")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  CSE END-TO-END SMOKE TEST")
    print("=" * 70)
    print(f"\n  Query : {args.query[:80]}")
    print(f"  Answer: {args.answer[:80]}...")

    # ── Step 1: Run MAD+CSE pipeline ──────────────────────────────────────────
    print("\n[1/3] Running MAD+CSE pipeline (this may take 1–3 min with Ollama)...")
    try:
        from multi_agent.mad_pipeline import run_mad
        from multi_agent.config import DB_PATH
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        print("  Make sure you're running from repo root with the correct PYTHONPATH.")
        sys.exit(1)

    try:
        result = run_mad(query=args.query, llm_answer=args.answer)
    except Exception as e:
        print(f"[ERROR] run_mad() raised: {type(e).__name__}: {e}")
        print("\n  Is Ollama running? Try: ollama serve")
        print("  Is qwen2.5:7b pulled? Try: ollama pull qwen2.5:7b")
        sys.exit(1)

    print(f"  Pipeline complete. query_id={result.query_id}, rollout_id={result.rollout_id}")

    # ── Step 2: Read back from DB ─────────────────────────────────────────────
    print(f"\n[2/3] Reading CSE row from DB ({DB_PATH}) ...")
    row = _read_cse_row(DB_PATH, result.query_id, result.rollout_id)
    if row is None:
        print(f"[ERROR] Row not found in DB for query_id={result.query_id}")
        sys.exit(1)

    # ── Step 3: Verify CSE columns ────────────────────────────────────────────
    print(f"\n[3/3] Verifying CSE columns ...")
    all_present, nulls = _check_cse_columns(row)

    print(f"\n  {'Column':<22} {'Value'}")
    print(f"  {'-'*50}")
    for col in ["final_cse_score", "routing_decision", "cse_version",
                "cse_f_llm", "cse_h_llm", "cse_relevancy", "cse_judge_eval"]:
        val  = row.get(col)
        mark = "✓" if val is not None else "✗ NULL"
        if isinstance(val, float):
            val = f"{val:.4f}"
        print(f"  {mark}  {col:<20} {val}")

    print()
    if nulls:
        print(f"[FAIL]  {len(nulls)} column(s) are NULL: {nulls}")
        print("  This means CSE did not run or failed silently.")
        print("  Check logs above for '[CSE]' lines — look for v0.1 fallback messages.")
        sys.exit(1)
    else:
        version = row.get("cse_version", "unknown")
        routing = row.get("routing_decision", "unknown")
        score   = row.get("final_cse_score", 0)
        f_llm   = row.get("cse_f_llm", 0)
        h_llm   = row.get("cse_h_llm", 0)
        relev   = row.get("cse_relevancy", 0)
        judge   = row.get("cse_judge_eval", 0)

        print(f"[OK]    All CSE columns written successfully.")
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║  CSE RESULT SUMMARY                  ║")
        print(f"  ╠══════════════════════════════════════╣")
        print(f"  ║  Version      : {version:<20}  ║")
        print(f"  ║  Final Score  : {score:<20.4f}  ║")
        print(f"  ║  Routing      : {routing:<20}  ║")
        print(f"  ╠══════════════════════════════════════╣")
        print(f"  ║  F_llm        : {f_llm:<20.4f}  ║")
        print(f"  ║  H_llm        : {h_llm:<20.4f}  ║")
        print(f"  ║  Relevancy    : {relev:<20.4f}  ║")
        print(f"  ║  Judge eval   : {judge:<20.4f}  ║")
        print(f"  ╚══════════════════════════════════════╝")

        if version == "v0.1":
            print()
            print("  ⚠  Version is v0.1 — DeepEval fell back to judge-only mode.")
            print("     F_llm / H_llm / relevancy are neutral defaults (not real scores).")
            print("     Check Ollama is running and responding within timeout.")
        else:
            # Sanity-check: warn if DeepEval scores look stuck at boundary values
            stuck_warn = []
            if f_llm in (0.0, 1.0):
                stuck_warn.append(f"F_llm={f_llm}")
            if h_llm in (0.0, 1.0):
                stuck_warn.append(f"H_llm={h_llm}")
            if relev in (0.0, 1.0):
                stuck_warn.append(f"Relevancy={relev}")
            if stuck_warn:
                print()
                print(f"  ⚠  Possibly stuck scores: {', '.join(stuck_warn)}")
                print("     DeepEval metrics returned extreme boundary values (0.0 or 1.0).")
                print("     This can happen when Ollama outputs malformed JSON.")
                print("     Try running again — qwen2.5:7b can produce inconsistent output.")

    if args.verbose and hasattr(result, "transcript"):
        print("\n" + "=" * 70)
        print("  FULL MAD TRANSCRIPT")
        print("=" * 70)
        print(result.transcript)


if __name__ == "__main__":
    main()
