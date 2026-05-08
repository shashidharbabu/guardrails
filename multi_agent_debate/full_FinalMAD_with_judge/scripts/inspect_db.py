"""
Inspect the MAD SQLite database.

Usage:
  python scripts/inspect_db.py --db mad.db
  python scripts/inspect_db.py --db mad.db --run-id my-run-01
  python scripts/inspect_db.py --db mad.db --query-id q_001
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from configs import config
from src.db.db import connect, fetch_all, fetch_one


def print_summary(db_path: str, run_id: str | None = None) -> None:
    filter_sql = " WHERE run_id = ?" if run_id else ""
    params = (run_id,) if run_id else ()

    queries = fetch_all(f"SELECT query_id, run_id, user_query FROM queries{filter_sql}", params, db_path)
    print(f"\n=== DB SUMMARY: {db_path} ===")
    print(f"Queries:  {len(queries)}")

    claim_count_rows = fetch_all(
        f"SELECT COUNT(*) AS n FROM claims c JOIN queries q ON q.query_id = c.query_id{filter_sql}",
        params, db_path
    )
    print(f"Claims:   {claim_count_rows[0]['n'] if claim_count_rows else 0}")

    out_count_rows = fetch_all(
        f"SELECT COUNT(*) AS n FROM agent_outputs ao JOIN claims c ON c.claim_id = ao.claim_id JOIN queries q ON q.query_id = c.query_id{filter_sql}",
        params, db_path
    )
    print(f"Agent outputs: {out_count_rows[0]['n'] if out_count_rows else 0}")

    delta_count_rows = fetch_all(
        f"SELECT COUNT(*) AS n FROM agent_deltas ad JOIN claims c ON c.claim_id = ad.claim_id JOIN queries q ON q.query_id = c.query_id{filter_sql}",
        params, db_path
    )
    print(f"Agent deltas:  {delta_count_rows[0]['n'] if delta_count_rows else 0}")

    verdict_rows = fetch_all(
        f"SELECT j.v_label, COUNT(*) AS n FROM judge_verdicts j JOIN claims c ON c.claim_id = j.claim_id JOIN queries q ON q.query_id = c.query_id{filter_sql} GROUP BY j.v_label ORDER BY j.v_label",
        params, db_path
    )
    if verdict_rows:
        label_map = {"0.0": "NOT_SUPPORTED", "0.5": "PARTIAL", "1.0": "SUPPORTED"}
        dist = {label_map.get(str(r["v_label"]), str(r["v_label"])): r["n"] for r in verdict_rows}
        total = sum(dist.values())
        print(f"Judge verdicts: {total} total  {dist}")
    else:
        print("Judge verdicts: 0 (run judge to populate)")

    print("\nQueries:")
    for q in queries:
        print(f"  [{q['run_id']}] {q['query_id']}: {q['user_query'][:60]}...")


def print_query(db_path: str, query_id: str) -> None:
    q = fetch_one("SELECT * FROM queries WHERE query_id = ?", (query_id,), db_path)
    if not q:
        print(f"Query {query_id} not found")
        return
    claims = fetch_all("SELECT * FROM claims WHERE query_id = ? ORDER BY claim_index", (query_id,), db_path)
    print(f"\n=== QUERY: {query_id} ===")
    print(f"Run ID:  {q['run_id']}")
    print(f"Query:   {q['user_query']}")
    print(f"Baseline answer (first 200 chars): {q['baseline_answer'][:200]}")
    print(f"\nClaims ({len(claims)}):")
    for c in claims:
        outputs = fetch_all(
            "SELECT agent_role, round_num, verdict, confidence_internal FROM agent_outputs WHERE claim_id = ? ORDER BY round_num, agent_role",
            (c["claim_id"],), db_path
        )
        verdicts = fetch_all("SELECT v_label, judge_confidence FROM judge_verdicts WHERE claim_id = ?", (c["claim_id"],), db_path)
        print(f"\n  [{c['claim_index']}] {c['claim_text'][:80]}")
        print(f"       material={c['is_material']} critical={c['is_critical']} conf_prior={c['confidence_prior']}")
        for o in outputs:
            print(f"       {o['agent_role']} R{o['round_num']}: {o['verdict']} (conf={o['confidence_internal']:.2f})")
        if verdicts:
            v = verdicts[0]
            label_map = {0.0: "NOT_SUPPORTED", 0.5: "PARTIAL", 1.0: "SUPPORTED"}
            print(f"       JUDGE: {label_map.get(v['v_label'], str(v['v_label']))} (judge_conf={v['judge_confidence']:.2f})")
        else:
            print("       JUDGE: not yet run")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MAD SQLite database")
    parser.add_argument("--db", default=None, help="SQLite DB path")
    parser.add_argument("--run-id", default=None, help="Filter by run ID")
    parser.add_argument("--query-id", default=None, help="Show detailed info for a query")
    args = parser.parse_args()

    db_path = args.db or config.SQLITE_DB_PATH
    if not Path(db_path).exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    if args.query_id:
        print_query(db_path, args.query_id)
    else:
        print_summary(db_path, args.run_id)


if __name__ == "__main__":
    main()
