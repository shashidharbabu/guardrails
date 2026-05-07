#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  Run Feedback Loop on Real MAD Database
═══════════════════════════════════════════════════════════════

Self-contained script — does NOT import pipeline.py.
All reward computation and training data prep is embedded here.

Usage (Colab):
  Upload this file + mad_store.db to /content/
  Run: !python run_on_real_data.py

Usage (local):
  python run_on_real_data.py --db /path/to/mad_store.db

Does NOT run training automatically.
"""

import os
import sys
import shutil
import json
import sqlite3
import logging
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# PII keywords for verdict mapping
PII_KEYWORDS = [
    "ssn", "social security", "medical record", "fmla",
    "date of birth", "dob", "passport", "credit card",
    "bank account", "salary", "compensation", "health",
]
# Safety-type queries (jailbreak, prompt injection) — synthetic IDs
SAFETY_QUERY_IDS = {"q_004", "q_006"}


# ═══════════════════════════════════════════════════════════════
# Database discovery
# ═══════════════════════════════════════════════════════════════

def find_database(cli_path=None):
    """Find the MAD database file."""
    candidates = [
        cli_path,
        "/content/mad_store.db",
        os.path.join(SCRIPT_DIR, "..", "data", "mad_store.db"),
        os.path.expanduser("~/Downloads/mad_store.db"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


# ═══════════════════════════════════════════════════════════════
# Database inspection
# ═══════════════════════════════════════════════════════════════

def inspect_database(db_path):
    """Print database summary before processing."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print("=" * 60)
    print("Database Inspection")
    print("=" * 60)
    print(f"  Source: {db_path}")
    print()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row["name"] for row in cur.fetchall()]
    print(f"  Tables found: {tables}")

    for table in tables:
        if table == "sqlite_sequence":
            continue
        cur.execute(f"SELECT COUNT(*) as cnt FROM [{table}]")
        cnt = cur.fetchone()["cnt"]
        print(f"    {table}: {cnt} rows")

    expected = {"queries", "claims", "attacks", "judge_verdicts"}
    missing = expected - set(tables)
    if missing:
        print(f"\n  MISSING TABLES: {missing}")
        print("  Cannot proceed without these tables.")
        conn.close()
        return False

    cur.execute("SELECT checkpoint, COUNT(*) as cnt FROM claims GROUP BY checkpoint")
    print("\n  Claims by checkpoint:")
    for row in cur.fetchall():
        print(f"    {row['checkpoint']}: {row['cnt']}")

    cur.execute("SELECT v_label, COUNT(*) as cnt FROM judge_verdicts GROUP BY v_label")
    print("\n  Judge verdict distribution:")
    for row in cur.fetchall():
        print(f"    v_label={row['v_label']}: {row['cnt']}")

    cur.execute("SELECT COUNT(*) as cnt FROM claims WHERE checkpoint='post_cycle2'")
    post_cycle2 = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) as cnt FROM judge_verdicts")
    judge_count = cur.fetchone()["cnt"]
    print(f"\n  post_cycle2 claims: {post_cycle2}")
    print(f"  Judge verdicts:     {judge_count}")
    if judge_count < post_cycle2:
        print(f"  NOTE: {post_cycle2 - judge_count} claims have no judge verdict")
        print(f"        These will default to v_label=0.5")

    cur.execute("""
        SELECT query_id, COUNT(DISTINCT rollout_id) as rollout_count
        FROM queries GROUP BY query_id
    """)
    rollout_counts = [row["rollout_count"] for row in cur.fetchall()]
    max_rollouts = max(rollout_counts) if rollout_counts else 0
    print(f"\n  Unique queries: {len(rollout_counts)}")
    print(f"  Rollouts per query: {max_rollouts}")
    if max_rollouts == 1:
        print("  NOTE: Single rollout per query — GRPO advantages will be 0")
        print("        GRPOTrainer computes its own advantages internally")

    cur.execute("""
        SELECT COUNT(*) as cnt FROM claims
        WHERE agent_a_prompt IS NOT NULL AND LENGTH(agent_a_prompt) > 0
    """)
    prompted = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) as cnt FROM claims")
    total_claims = cur.fetchone()["cnt"]
    print(f"\n  agent_a_prompt populated: {prompted}/{total_claims}")

    if "rewards" in tables:
        cur.execute("SELECT COUNT(*) as cnt FROM rewards")
        print(f"  rewards table: {cur.fetchone()['cnt']} rows (will be cleared)")
    else:
        print("  rewards table: does not exist (will be created)")

    conn.close()
    print()
    return True


# ═══════════════════════════════════════════════════════════════
# Phase 2: compute_rewards()
# ═══════════════════════════════════════════════════════════════

def compute_rewards(db_path):
    """
    Reads claims (post_cycle2), attacks, and judge_verdicts.
    Computes b_reward, brier_reward, is_clean, rollout_total, grpo_advantage.
    Writes results to the rewards table.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Create rewards table if it does not exist (real DB won't have it)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rewards (
            query_id TEXT NOT NULL,
            rollout_id TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            p_final REAL,
            v_label REAL,
            brier_reward REAL,
            is_clean INTEGER,
            rollout_total REAL,
            grpo_advantage REAL,
            timestamp TEXT,
            PRIMARY KEY (query_id, rollout_id, claim_id)
        )
    """)
    # Clear previous rewards so this is idempotent
    cur.execute("DELETE FROM rewards")
    cur.execute("UPDATE attacks SET b_reward = NULL")

    # ── Step 1: Compute b_reward for every attack ──
    cur.execute("""
        SELECT a.attack_id, a.query_id, a.rollout_id, a.claim_id,
               a.cycle, a.p_before_attack, a.p_after_attack
        FROM attacks a
    """)
    attacks = cur.fetchall()

    # Build v_label lookup
    cur.execute("SELECT query_id, rollout_id, claim_id, v_label FROM judge_verdicts")
    v_lookup = {}
    null_vlabel_count = 0
    for row in cur.fetchall():
        v_label = row["v_label"]
        if v_label is None:
            v_label = 0.5
            null_vlabel_count += 1
            logging.warning(
                f"NULL v_label for {row['query_id']}/{row['rollout_id']}/"
                f"{row['claim_id']} — defaulting to 0.5"
            )
        v_lookup[(row["query_id"], row["rollout_id"], row["claim_id"])] = v_label

    # Aggregate attack deltas per claim
    claim_attack_agg = defaultdict(lambda: {"max_p_before": 0.0, "min_p_after": 1.0, "attack_ids": []})
    for a in attacks:
        key = (a["query_id"], a["rollout_id"], a["claim_id"])
        agg = claim_attack_agg[key]
        if a["p_before_attack"] > agg["max_p_before"]:
            agg["max_p_before"] = a["p_before_attack"]
        if a["p_after_attack"] is not None and a["p_after_attack"] < agg["min_p_after"]:
            agg["min_p_after"] = a["p_after_attack"]
        agg["attack_ids"].append(a["attack_id"])

    no_verdict_count = 0
    for key, agg in claim_attack_agg.items():
        qid, rid, cid = key
        v_label = v_lookup.get(key)
        if v_label is None:
            v_label = 0.5
            no_verdict_count += 1
        delta_p = agg["max_p_before"] - agg["min_p_after"]

        if delta_p >= 0.2 and v_label < 1.0:
            b_reward = 1.0
        elif delta_p >= 0.2 and v_label == 1.0:
            b_reward = -1.0
        else:
            b_reward = 0.0

        for aid in agg["attack_ids"]:
            cur.execute("UPDATE attacks SET b_reward = ? WHERE attack_id = ?", (b_reward, aid))

    conn.commit()

    # ── Step 2: Compute brier_reward and is_clean per claim ──
    cur.execute("""
        SELECT query_id, rollout_id, claim_id, confidence_p
        FROM claims
        WHERE checkpoint = 'post_cycle2'
    """)
    final_claims = cur.fetchall()

    # Build b_reward lookup
    b_reward_lookup = {}
    for key, agg in claim_attack_agg.items():
        qid, rid, cid = key
        v_label = v_lookup.get(key, 0.5)
        delta_p = agg["max_p_before"] - agg["min_p_after"]
        if delta_p >= 0.2 and v_label < 1.0:
            b_reward_lookup[key] = 1.0
        elif delta_p >= 0.2 and v_label == 1.0:
            b_reward_lookup[key] = -1.0
        else:
            b_reward_lookup[key] = 0.0

    # Compute per-claim rewards
    claim_rewards = []
    no_judge_verdict_count = 0
    for fc in final_claims:
        qid = fc["query_id"]
        rid = fc["rollout_id"]
        cid = fc["claim_id"]
        p_final = fc["confidence_p"]
        v_label = v_lookup.get((qid, rid, cid))
        if v_label is None:
            v_label = 0.5
            no_judge_verdict_count += 1
            logging.warning(
                f"No judge verdict for claim {cid} "
                f"({qid}/{rid}) — defaulting v_label to 0.5"
            )

        brier_reward = (2.0 * p_final * v_label) - (p_final ** 2)

        b_rew = b_reward_lookup.get((qid, rid, cid))
        if b_rew is not None and b_rew == -1.0:
            is_clean = 0
        else:
            is_clean = 1

        claim_rewards.append({
            "query_id": qid, "rollout_id": rid, "claim_id": cid,
            "p_final": p_final, "v_label": v_label,
            "brier_reward": brier_reward, "is_clean": is_clean,
        })

    # ── Step 3: Compute rollout_total ──
    rollout_totals = defaultdict(float)
    for cr in claim_rewards:
        if cr["is_clean"] == 1:
            rollout_totals[(cr["query_id"], cr["rollout_id"])] += cr["brier_reward"]

    for cr in claim_rewards:
        key = (cr["query_id"], cr["rollout_id"])
        if key not in rollout_totals:
            rollout_totals[key] = 0.0

    # NOTE: grpo_advantage computed here is for analysis and logging only.
    # The GRPOTrainer computes its own advantages internally from
    # 4 live generations per prompt at training time (num_generations=4).

    # ── Step 4: Compute grpo_advantage ──
    query_rollout_totals = defaultdict(list)
    for (qid, rid), total in rollout_totals.items():
        query_rollout_totals[qid].append(total)

    mean_totals = {}
    for qid, totals in query_rollout_totals.items():
        mean_totals[qid] = sum(totals) / len(totals)

    # ── Step 5: Write to rewards table ──
    now = datetime.now().isoformat()
    for cr in claim_rewards:
        qid, rid = cr["query_id"], cr["rollout_id"]
        rt = rollout_totals[(qid, rid)]
        adv = rt - mean_totals[qid]

        cur.execute(
            "INSERT INTO rewards VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cr["query_id"], cr["rollout_id"], cr["claim_id"],
             cr["p_final"], cr["v_label"], cr["brier_reward"],
             cr["is_clean"], rt, adv, now),
        )

    conn.commit()

    # ── Print summary ──
    total_rewards = len(claim_rewards)
    clean_count = sum(1 for cr in claim_rewards if cr["is_clean"] == 1)
    gaslighted_count = sum(1 for cr in claim_rewards if cr["is_clean"] == 0)
    avg_brier = sum(cr["brier_reward"] for cr in claim_rewards) / total_rewards if total_rewards else 0

    print("=" * 60)
    print("Phase 2: compute_rewards() complete")
    print("=" * 60)
    print(f"  Reward computation complete:")
    print(f"    Total records:                  {total_rewards}")
    print(f"    Clean (is_clean=1):             {clean_count}")
    print(f"    Excluded - gaslighted (b=-1):   {gaslighted_count}")
    print(f"    NULL v_label warnings:          {null_vlabel_count}")
    print(f"    No judge verdict (default 0.5): {no_judge_verdict_count}")
    print(f"    Average Brier reward:           {avg_brier:.4f}")
    print()

    # Detect rollouts-per-query for GRPO advantage warning
    max_rollouts = max(len(v) for v in query_rollout_totals.values()) if query_rollout_totals else 0
    if max_rollouts == 1:
        print("  WARNING: Only 1 rollout per query detected.")
        print("  GRPO database-level advantages will be 0.")
        print("  GRPOTrainer will compute advantages internally")
        print("  from 4 live generations — training still works.")
        print()

    # Show per-query rollout totals
    print("  Rollout totals and GRPO advantages:")
    for qid in sorted(query_rollout_totals.keys()):
        qid_short = qid[:12] + "..." if len(qid) > 12 else qid
        print(f"    {qid_short} (mean = {mean_totals[qid]:.4f}):")
        for (q, rid), total in rollout_totals.items():
            if q == qid:
                adv = total - mean_totals[qid]
                rid_short = rid[:12] + "..." if len(rid) > 12 else rid
                print(f"      {rid_short}: total={total:+.4f}  advantage={adv:+.4f}")

    conn.close()
    return claim_rewards


# ═══════════════════════════════════════════════════════════════
# Phase 3: prepare_training_data()
# ═══════════════════════════════════════════════════════════════

def prepare_training_data(db_path, output_dir):
    """
    Build GRPO training dataset from rewards + claims.
    Reads is_clean=1 rewards, joins with claims (post_cycle2),
    saves training_data.json.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT r.query_id, r.rollout_id, r.claim_id,
               r.brier_reward, r.v_label, r.p_final,
               c.agent_a_prompt, c.claim_text,
               c.is_material
        FROM rewards r
        JOIN claims c
            ON r.query_id = c.query_id
            AND r.rollout_id = c.rollout_id
            AND r.claim_id = c.claim_id
            AND c.checkpoint = 'post_cycle2'
        WHERE r.is_clean = 1
    """)
    all_rows = cur.fetchall()
    conn.close()

    # Exclude double-uncertain records
    idk_excluded = 0
    rows = []
    for row in all_rows:
        p_final = row["p_final"]
        v_label = row["v_label"]

        is_double_uncertain = (
            v_label == 0.5 and
            p_final is not None and
            0.4 <= p_final <= 0.6
        )

        if is_double_uncertain:
            idk_excluded += 1
            continue
        rows.append(row)

    def _map_expected_verdict(v_label, query_id, claim_text, is_critical):
        """Map v_label + context to expected verdict label."""
        if v_label == 1.0:
            return "SUPPORTED"
        if v_label == 0.5:
            return "PARTIAL"
        # v_label == 0.0
        if is_critical and query_id in SAFETY_QUERY_IDS:
            return "UNSAFE"
        claim_lower = (claim_text or "").lower()
        if any(kw in claim_lower for kw in PII_KEYWORDS):
            return "PII_LEAK"
        return "NOT_SUPPORTED"

    # Build training dataset
    dataset = []
    for row in rows:
        prompt = row["agent_a_prompt"]
        if prompt is None or prompt.strip() == "":
            continue
        # is_critical not in real DB — fall back to is_material
        is_critical = row["is_critical"] if "is_critical" in row.keys() else row["is_material"]
        expected_verdict = _map_expected_verdict(
            row["v_label"], row["query_id"],
            row["claim_text"], is_critical,
        )
        dataset.append({
            "prompt": prompt,
            "v_label": row["v_label"],
            "expected_verdict": expected_verdict,
            "reference_reward": row["brier_reward"],
            "reference_p": row["p_final"],
        })

    # Save to JSON
    out_path = os.path.join(output_dir, "training_data.json")
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Print summary
    rewards = [d["reference_reward"] for d in dataset]
    avg_r = sum(rewards) / len(rewards) if rewards else 0
    min_r = min(rewards) if rewards else 0
    max_r = max(rewards) if rewards else 0

    print("=" * 60)
    print("Phase 3: prepare_training_data() complete")
    print("=" * 60)
    print(f"  Total clean records:      {len(all_rows)}")
    print(f"  Excluded - double uncertain (v=0.5, p=0.4-0.6): {idk_excluded}")
    print(f"  Training examples:        {len(dataset)}")
    print(f"  Average reference reward: {avg_r:.4f}")
    print(f"  Min reference reward:     {min_r:.4f}")
    print(f"  Max reference reward:     {max_r:.4f}")
    print()

    verdict_counts = {}
    for d in dataset:
        v = d["expected_verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    print(f"  Expected verdict distribution:")
    for v in sorted(verdict_counts):
        print(f"    {v}: {verdict_counts[v]}")
    print()

    print("  Sample examples:")
    for i, ex in enumerate(dataset[:2]):
        print(f"    [{i+1}] prompt: \"{ex['prompt'][:100]}...\"")
        print(f"        v_label: {ex['v_label']}, expected_verdict: {ex['expected_verdict']}, "
              f"ref_reward: {ex['reference_reward']:.4f}")
    print(f"\n  Saved to: {out_path}")

    return dataset


# ═══════════════════════════════════════════════════════════════
# Training data summary
# ═══════════════════════════════════════════════════════════════

def print_training_summary(data_path):
    """Print detailed summary of generated training data."""
    if not os.path.exists(data_path):
        print("  ERROR: training data file was not created.")
        return

    with open(data_path, "r") as f:
        data = json.load(f)

    print("\n" + "=" * 60)
    print("Training Data Summary")
    print("=" * 60)

    print(f"  Total examples: {len(data)}")

    if not data:
        print("  WARNING: No training examples generated.")
        print("  Check that judge_verdicts has rows and claims have agent_a_prompt.")
        return

    # Expected verdict distribution
    verdict_counts = {}
    for d in data:
        v = d["expected_verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    print(f"\n  Expected verdict distribution:")
    for v in sorted(verdict_counts):
        print(f"    {v}: {verdict_counts[v]}")

    # v_label distribution
    vlabel_counts = {}
    for d in data:
        vl = d["v_label"]
        vlabel_counts[vl] = vlabel_counts.get(vl, 0) + 1
    print(f"\n  v_label distribution:")
    for vl in sorted(vlabel_counts):
        print(f"    {vl}: {vlabel_counts[vl]}")

    # Reward statistics
    rewards = [d["reference_reward"] for d in data]
    min_r = min(rewards)
    max_r = max(rewards)
    avg_r = sum(rewards) / len(rewards)
    pos_count = sum(1 for r in rewards if r > 0)
    neg_count = sum(1 for r in rewards if r < 0)
    zero_count = sum(1 for r in rewards if r == 0)

    print(f"\n  Reference reward statistics:")
    print(f"    Min:      {min_r:.4f}")
    print(f"    Max:      {max_r:.4f}")
    print(f"    Average:  {avg_r:.4f}")
    print(f"    Positive: {pos_count}")
    print(f"    Negative: {neg_count}")
    print(f"    Zero:     {zero_count}")

    # GRPO suitability check
    print(f"\n  GRPO suitability check:")
    if pos_count > 0 and neg_count > 0:
        print(f"    GOOD — both positive and negative rewards present.")
        print(f"    GRPO can learn from the contrast between good and bad calibration.")
    elif pos_count > 0 and neg_count == 0:
        print(f"    WARNING — no negative rewards. All claims are well-calibrated.")
        print(f"    GRPO may have limited learning signal (no bad examples to push away from).")
    elif neg_count > 0 and pos_count == 0:
        print(f"    WARNING — no positive rewards. All claims are miscalibrated.")
        print(f"    GRPO can still learn but has no good examples to reinforce.")
    else:
        print(f"    WARNING — all rewards are zero. No learning signal.")

    # Prompt length stats
    prompt_lens = [len(d["prompt"]) for d in data]
    print(f"\n  Prompt lengths:")
    print(f"    Min: {min(prompt_lens)} chars")
    print(f"    Max: {max(prompt_lens)} chars")
    print(f"    Avg: {sum(prompt_lens) // len(prompt_lens)} chars")

    print(f"\n  Training data saved to: {data_path}")
    print(f"  File size: {os.path.getsize(data_path):,} bytes")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    # Parse optional --db argument
    cli_path = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--db" and i < len(sys.argv) - 1:
            cli_path = sys.argv[i + 1]

    # Find database
    db_path = find_database(cli_path)
    if not db_path:
        print("ERROR: Could not find mad_store.db")
        print("  Provide path with: python run_on_real_data.py --db /path/to/mad_store.db")
        sys.exit(1)

    print()
    print("  Enterprise Guardrail Feedback Loop")
    print("  Running on Real MAD Data")
    print()

    # Step 1: Inspect
    if not inspect_database(db_path):
        sys.exit(1)

    # Step 2: Create working copy (never modify original)
    working_dir = os.path.dirname(db_path)
    working_db = os.path.join(working_dir, "mad_store_working.db")
    print(f"  Creating working copy: {working_db}")
    shutil.copy2(db_path, working_db)
    print(f"  Original will not be modified.")
    print()

    # Step 3: Set output directory
    if os.path.isdir("/content"):
        output_dir = "/content"
    else:
        output_dir = os.path.join(SCRIPT_DIR, "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    # Step 4: Run compute_rewards
    print("\n" + "=" * 60)
    print("Running Phase 2: compute_rewards()")
    print("=" * 60)
    compute_rewards(working_db)

    # Step 5: Run prepare_training_data
    print("\n" + "=" * 60)
    print("Running Phase 3: prepare_training_data()")
    print("=" * 60)
    dataset = prepare_training_data(working_db, output_dir)

    # Step 6: Rename to real_training_data.json
    default_path = os.path.join(output_dir, "training_data.json")
    training_data_path = os.path.join(output_dir, "real_training_data.json")
    if os.path.exists(default_path):
        shutil.move(default_path, training_data_path)
        print(f"\n  Renamed: training_data.json → {os.path.basename(training_data_path)}")

    # Step 7: Print training summary
    print_training_summary(training_data_path)

    # Step 8: Final instructions
    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print()
    print("  Training data ready. Review the summary above.")
    print()
    print("  To run training:")
    print(f"    python run_on_colab.py")
    print(f"    but first update training_data.json path to:")
    print(f"    {training_data_path}")
    print()
    print("  Or modify run_on_colab.py line 75:")
    print(f'    found["training_data.json"] = "{training_data_path}"')
    print()
    print(f"  Working database: {working_db}")
    print(f"  Original database: {db_path} (unchanged)")
    print()


if __name__ == "__main__":
    main()
