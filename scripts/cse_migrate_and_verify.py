#!/usr/bin/env python3
"""
scripts/cse_migrate_and_verify.py — DB Migration for CSE columns.

What this does:
  1. Calls init_db() — safe to run on an existing DB (idempotent).
     Adds the 5 CSE columns to the 'queries' table if they're missing:
       cse_f_llm, cse_h_llm, cse_relevancy, cse_judge_eval, cse_version
  2. Verifies all expected columns are present.
  3. Prints a snapshot of the current DB state (row counts + how many
     rows already have CSE scores filled in).

Run from repo root:
    python scripts/cse_migrate_and_verify.py

Expected output:
    [migrate] Running init_db() on /path/to/mad_store.db ...
    [Storage] Database initialised at /path/to/mad_store.db
    [verify]  Column check on 'queries' table ...
    [verify]  ✓ cse_f_llm      REAL
    [verify]  ✓ cse_h_llm      REAL
    ...
    [stats]   queries total: N  |  with CSE scores: M
    [OK]      Migration complete. DB is ready for CSE writes.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# ── Path bootstrap — same logic as confidence/test_harness.py ─────────────────
_REPO = Path(__file__).resolve().parent.parent
for _p in [str(_REPO), str(_REPO / "multi_agent_debate"), str(_REPO / "rag_folder")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_REPO / ".env")

# ── Required CSE columns and their types ──────────────────────────────────────
_CSE_COLUMNS: dict[str, str] = {
    "final_cse_score":  "REAL",
    "routing_decision": "TEXT",
    "cse_f_llm":        "REAL",
    "cse_h_llm":        "REAL",
    "cse_relevancy":    "REAL",
    "cse_judge_eval":   "REAL",
    "cse_version":      "TEXT",
}

# ── Expected tables ───────────────────────────────────────────────────────────
_EXPECTED_TABLES = ["queries", "claims", "attacks", "judge_verdicts"]


def _run_init_db() -> str:
    from multi_agent.storage import init_db
    from multi_agent.config import DB_PATH
    print(f"\n[migrate] Running init_db() on {DB_PATH} ...")
    init_db()
    return DB_PATH


def _verify_columns(db_path: str) -> bool:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("PRAGMA table_info(queries)").fetchall()
    con.close()

    existing = {r["name"]: r["type"] for r in rows}
    print(f"\n[verify]  Column check on 'queries' table ({db_path}) ...")

    ok = True
    for col, expected_type in _CSE_COLUMNS.items():
        if col in existing:
            print(f"[verify]  ✓  {col:<20} {existing[col]}")
        else:
            print(f"[verify]  ✗  {col:<20} MISSING  (expected {expected_type})")
            ok = False
    return ok


def _verify_tables(db_path: str) -> bool:
    con = sqlite3.connect(db_path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    con.close()

    ok = True
    print(f"\n[verify]  Table check ...")
    for t in _EXPECTED_TABLES:
        if t in tables:
            print(f"[verify]  ✓  {t}")
        else:
            print(f"[verify]  ✗  {t}  MISSING")
            ok = False
    return ok


def _print_db_stats(db_path: str) -> None:
    con = sqlite3.connect(db_path)

    try:
        total = con.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        with_cse = con.execute(
            "SELECT COUNT(*) FROM queries WHERE final_cse_score IS NOT NULL"
        ).fetchone()[0]
        with_v2 = con.execute(
            "SELECT COUNT(*) FROM queries WHERE cse_version = 'v2.0'"
        ).fetchone()[0]
        with_v01 = con.execute(
            "SELECT COUNT(*) FROM queries WHERE cse_version = 'v0.1'"
        ).fetchone()[0]
        print(f"\n[stats]   queries table:")
        print(f"[stats]     Total rows         : {total}")
        print(f"[stats]     With CSE score      : {with_cse}")
        print(f"[stats]     Version v2.0 (full) : {with_v2}")
        print(f"[stats]     Version v0.1 (judge): {with_v01}")
        print(f"[stats]     Pending CSE write   : {total - with_cse}")
    except Exception as e:
        print(f"[stats]   Could not read stats: {e}")
    finally:
        con.close()


def main() -> None:
    db_path = _run_init_db()
    tables_ok = _verify_tables(db_path)
    columns_ok = _verify_columns(db_path)
    _print_db_stats(db_path)

    print()
    if tables_ok and columns_ok:
        print("[OK]      Migration complete — DB is ready for CSE writes.")
    else:
        print("[ERROR]   Migration incomplete — see missing items above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
