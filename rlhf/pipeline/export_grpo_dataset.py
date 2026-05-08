"""Export JSONL for Colab / TRL GRPO from scored rewards + post_cycle2 prompts."""
from __future__ import annotations

import argparse
import json
import sys

from rlhf.feedback_loop.config import get_db_path
from rlhf.feedback_loop.db import connect, init_feedback_schema


def export_jsonl(
    out_path: str,
    db_path: str | None = None,
    *,
    require_clean: bool = True,
) -> int:
    init_feedback_schema(db_path)
    path = db_path or get_db_path()
    n = 0
    with connect(path) as con:
        clean_clause = "AND r.is_clean = 1" if require_clean else ""
        rows = con.execute(
            f"""
            SELECT r.query_id, r.rollout_id, r.claim_id,
                   r.final_reward, r.auto_reward, r.grpo_advantage,
                   r.brier_reward, r.is_clean, c.agent_a_prompt
            FROM rewards r
            JOIN claims c
              ON c.query_id = r.query_id AND c.rollout_id = r.rollout_id
             AND c.claim_id = r.claim_id AND c.checkpoint = 'post_cycle2'
            WHERE r.final_reward IS NOT NULL
              {clean_clause}
            ORDER BY r.scored_at ASC
            """
        ).fetchall()
        with open(out_path, "w", encoding="utf-8") as fout:
            for row in rows:
                rec = {
                    "prompt": row["agent_a_prompt"],
                    "reward": float(row["final_reward"]),
                    "metadata": {
                        "query_id": row["query_id"],
                        "rollout_id": row["rollout_id"],
                        "claim_id": row["claim_id"],
                        "auto_reward": row["auto_reward"],
                        "brier_reward": row["brier_reward"],
                        "grpo_advantage": row["grpo_advantage"],
                        "is_clean": row["is_clean"],
                    },
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export GRPO JSONL dataset")
    p.add_argument("output", help="Output .jsonl path")
    p.add_argument("--db", default=None, help="SQLite path")
    p.add_argument(
        "--include-unclean",
        action="store_true",
        help="Include is_clean=0 rows (default: only clean)",
    )
    args = p.parse_args(argv)
    count = export_jsonl(
        args.output,
        args.db,
        require_clean=not args.include_unclean,
    )
    print(f"Wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
