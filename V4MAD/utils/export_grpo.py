"""
Export GRPO training data from the completed MAD v4 run.

Reads mad_v4.db and produces:
  data/grpo_agent_a.jsonl   — Agent A R0 prompts + v_labels (590 rows combined)
  data/grpo_agent_b.jsonl   — Agent B R0 prompts + v_labels
  data/grpo_combined.jsonl  — Both agents merged (use this for single-model GRPO)

Each JSONL row:
  {
    "prompt":       [{"role":"system","content":"..."}, {"role":"user","content":"..."}],
    "v_label":      0.5,       # judge ground truth — used in Brier reward
    "claim_id":     "...",     # for tracing
    "agent_role":   "agent_a",
    "query_id":     "...",
    "claim_text":   "..."
  }

Why R0 only (not R1):
  R0 is the cleaner signal — no peer dependency, directly measures how well
  the agent calibrates against evidence. R1 prompts contain the peer's output
  which would need to be frozen from the current run, making the prompt too long
  and coupling training to a specific peer behavior.

Run after step5 completes:
  python utils/export_grpo.py
"""

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH
from prompts import AGENT_A_SYSTEM, AGENT_B_SYSTEM, format_chunks, round0_user


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Load all the tables we need
    claims_map = {
        r["claim_id"]: dict(r)
        for r in conn.execute("SELECT * FROM claims").fetchall()
    }
    queries_map = {
        r["query_id"]: dict(r)
        for r in conn.execute("SELECT * FROM queries").fetchall()
    }
    verdicts_map = {
        r["claim_id"]: float(r["v_label"])
        for r in conn.execute("SELECT claim_id, v_label FROM judge_verdicts").fetchall()
    }
    chunks_map = {
        r["claim_id"]: {
            "agent_a": json.loads(r["agent_a_chunks"] or "[]"),
            "agent_b": json.loads(r["agent_b_chunks"] or "[]"),
        }
        for r in conn.execute("SELECT * FROM claim_chunks").fetchall()
    }

    # Only R0 agent outputs
    agent_rows = conn.execute(
        "SELECT * FROM agent_outputs WHERE round_num = 0"
    ).fetchall()

    records_a, records_b = [], []
    skipped = 0

    for row in agent_rows:
        claim_id   = row["claim_id"]
        agent_role = row["agent_role"]

        # Skip if no judge verdict for this claim
        if claim_id not in verdicts_map:
            skipped += 1
            continue

        claim  = claims_map.get(claim_id)
        if not claim:
            skipped += 1
            continue

        query  = queries_map.get(claim["query_id"])
        if not query:
            skipped += 1
            continue

        chunks_for_agent = chunks_map.get(claim_id, {}).get(agent_role, [])

        # Reconstruct the exact prompt the agent received in step4
        system_prompt = AGENT_A_SYSTEM if agent_role == "agent_a" else AGENT_B_SYSTEM
        user_prompt   = round0_user(claim, chunks_for_agent, query["user_query"])

        record = {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "v_label":    verdicts_map[claim_id],
            "claim_id":   claim_id,
            "agent_role": agent_role,
            "query_id":   claim["query_id"],
            "claim_text": claim["claim_text"],
        }

        if agent_role == "agent_a":
            records_a.append(record)
        else:
            records_b.append(record)

    data_dir = Path(DB_PATH).parent / "data"
    data_dir.mkdir(exist_ok=True)

    def _write(path, records):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  Wrote {len(records)} rows → {path}")

    _write(data_dir / "grpo_agent_a.jsonl", records_a)
    _write(data_dir / "grpo_agent_b.jsonl", records_b)
    _write(data_dir / "grpo_combined.jsonl", records_a + records_b)

    if skipped:
        print(f"  Skipped {skipped} rows (missing verdict or claim data)")

    # Distribution report
    all_records = records_a + records_b
    dist = Counter(r["v_label"] for r in all_records)
    print()
    print("v_label distribution in exported data:")
    for v, label in [(0.0, "NOT_SUPPORTED"), (0.5, "PARTIAL"), (1.0, "SUPPORTED")]:
        n = dist.get(v, 0)
        print(f"  {v}  {label:<15}: {n:3d} ({n/len(all_records)*100:.1f}%)")

    # GRPO readiness check
    print()
    print("GRPO readiness:")
    print(f"  Total prompts (combined)   : {len(all_records)}")
    print(f"  Effective completions/epoch: {len(all_records) * 4}  (num_generations=4)")
    if dist.get(1.0, 0) < 20:
        print(f"  ⚠ Only {dist.get(1.0,0)} SUPPORTED rows — Brier reward will mostly push")
        print(f"    confidence DOWN. That's fine — baseline LLM hallucinated a lot by design.")
    print()
    print("Next step:")
    print("  Upload grpo_combined.jsonl (or per-agent) to your GRPO training environment.")
    print("  In train_grpo.py, set:  DATA_PATH = 'data/grpo_agent_a.jsonl'  (or combined)")


if __name__ == "__main__":
    main()
