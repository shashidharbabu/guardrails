#!/usr/bin/env python3
"""Pipeline: Phases 2–5 for the Enterprise Guardrail Feedback Loop.
Reads from guardrails.db created by phase1_synthetic_data.py."""

import sqlite3
import os
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
DB_PATH = os.path.join(DATA_DIR, "guardrails.db")


# ═══════════════════════════════════════════════════════════════
# Phase 2: compute_rewards()
# ═══════════════════════════════════════════════════════════════

def compute_rewards():
    """
    Reads claims (post_cycle2), attacks, and judge_verdicts.
    Computes:
      - b_reward for each attack  → UPDATE attacks
      - brier_reward per claim
      - is_clean flag per claim
      - rollout_total per (query_id, rollout_id)
      - grpo_advantage = rollout_total - mean_total
    Writes results to the rewards table.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Create rewards table if it does not exist (real DB may not have it)
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
    # For each attack we need: delta_p and v_label from judge_verdicts
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
        # Handle NULL v_label defensively — default to 0.5
        if v_label is None:
            v_label = 0.5
            null_vlabel_count += 1
            import logging
            logging.warning(
                f"NULL v_label for {row['query_id']}/{row['rollout_id']}/"
                f"{row['claim_id']} — defaulting to 0.5"
            )
        v_lookup[(row["query_id"], row["rollout_id"], row["claim_id"])] = v_label

    # For b_reward we need the total delta across all cycles for that claim+rollout,
    # but the spec computes delta_p per-claim (across cycles) not per-attack.
    # Gather per-(qid, rid, cid): min p_after and max p_before across cycles.
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
    # Read post_cycle2 claims (these give us p_final)
    cur.execute("""
        SELECT query_id, rollout_id, claim_id, confidence_p
        FROM claims
        WHERE checkpoint = 'post_cycle2'
    """)
    final_claims = cur.fetchall()

    # Build b_reward lookup per (qid, rid, cid)
    b_reward_lookup = {}
    for key, agg in claim_attack_agg.items():
        qid, rid, cid = key
        v_label = v_lookup.get(key, 0.5)  # default 0.5 if no judge verdict
        delta_p = agg["max_p_before"] - agg["min_p_after"]
        if delta_p >= 0.2 and v_label < 1.0:
            b_reward_lookup[key] = 1.0
        elif delta_p >= 0.2 and v_label == 1.0:
            b_reward_lookup[key] = -1.0
        else:
            b_reward_lookup[key] = 0.0

    # Compute per-claim rewards
    claim_rewards = []  # list of dicts
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
            import logging
            logging.warning(
                f"No judge verdict for claim {cid} "
                f"({qid}/{rid}) — defaulting v_label to 0.5"
            )

        brier_reward = (2.0 * p_final * v_label) - (p_final ** 2)

        # is_clean: 0 if any attack on this claim had b_reward == -1
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

    # ── Step 3: Compute rollout_total (sum of brier for is_clean=1 claims in rollout) ──
    rollout_totals = defaultdict(float)
    for cr in claim_rewards:
        if cr["is_clean"] == 1:
            rollout_totals[(cr["query_id"], cr["rollout_id"])] += cr["brier_reward"]

    # For rollouts that have ALL claims excluded (is_clean=0), rollout_total stays 0
    # But we still need entries for them
    for cr in claim_rewards:
        key = (cr["query_id"], cr["rollout_id"])
        if key not in rollout_totals:
            rollout_totals[key] = 0.0

    # NOTE: grpo_advantage computed here is for
    # analysis and logging purposes only.
    # The GRPOTrainer computes its own advantages
    # internally from 4 live generations per prompt
    # at training time (num_generations=4).
    # These database-level advantages are NOT passed
    # to the trainer and do NOT affect training.
    # They are kept here to allow offline analysis
    # of which query rollouts produced better rewards.

    # ── Step 4: Compute grpo_advantage = rollout_total - mean_total across rollouts ──
    # Group rollout_totals by query_id
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

    print("=" * 50)
    print("Phase 2: compute_rewards() complete")
    print("=" * 50)
    print(f"  Reward computation complete:")
    print(f"    Total records:                {total_rewards}")
    print(f"    Clean (is_clean=1):           {clean_count}")
    print(f"    Excluded - gaslighted (b=-1): {gaslighted_count}")
    print(f"    NULL v_label warnings:        {null_vlabel_count}")
    print(f"    No judge verdict (default 0.5): {no_judge_verdict_count}")
    print(f"    Average Brier reward:         {avg_brier:.4f}")
    print()

    # Detect rollouts-per-query for GRPO advantage warning
    max_rollouts_per_query = max(len(v) for v in query_rollout_totals.values()) if query_rollout_totals else 0
    if max_rollouts_per_query == 1:
        print("  WARNING: Only 1 rollout per query detected.")
        print("  GRPO database-level advantages will be 0.")
        print("  GRPOTrainer will compute advantages internally")
        print("  from 4 live generations — training still works.")
        print()

    # Show per-query rollout totals and GRPO advantages
    print("  Rollout totals and GRPO advantages:")
    for qid in sorted(query_rollout_totals.keys()):
        print(f"    {qid[:12]}... (mean = {mean_totals[qid]:.4f}):")
        for (q, rid), total in rollout_totals.items():
            if q == qid:
                adv = total - mean_totals[qid]
                rid_display = rid[:12] + "..." if len(rid) > 12 else rid
                print(f"      {rid_display}: total={total:+.4f}  advantage={adv:+.4f}")

    conn.close()


# ═══════════════════════════════════════════════════════════════
# Phase 3: prepare_training_data()
# ═══════════════════════════════════════════════════════════════

def prepare_training_data():
    """
    Phase 3: Build GRPO training dataset from rewards + claims.
    - Reads is_clean=1 rewards
    - Joins with claims (post_cycle2) to get agent_a_prompt
    - Saves training_data.json
    - Returns list of {prompt, reward} dicts
    """
    conn = sqlite3.connect(DB_PATH)
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

    # Exclude records where Judge could not resolve
    # the claim (v=0.5) AND Agent A's final confidence
    # was in the ambiguous middle range (0.4-0.6).
    # These double-uncertain records produce noisy
    # Brier rewards near zero and are not useful
    # for training.
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

    # PII keywords for verdict mapping
    PII_KEYWORDS = [
        "ssn", "social security", "medical record", "fmla",
        "date of birth", "dob", "passport", "credit card",
        "bank account", "salary", "compensation", "health",
    ]
    # Safety-type queries (jailbreak, prompt injection)
    SAFETY_QUERY_IDS = {"q_004", "q_006"}

    def _map_expected_verdict(v_label, query_id, claim_text, is_critical):
        """Map v_label + context to expected verdict label."""
        if v_label == 1.0:
            return "SUPPORTED"
        if v_label == 0.5:
            return "PARTIAL"
        # v_label == 0.0: determine UNSAFE vs PII_LEAK vs NOT_SUPPORTED
        # is_critical not in real DB — default to is_material
        if is_critical and query_id in SAFETY_QUERY_IDS:
            return "UNSAFE"
        claim_lower = (claim_text or "").lower()
        if any(kw in claim_lower for kw in PII_KEYWORDS):
            return "PII_LEAK"
        return "NOT_SUPPORTED"

    # Filter out NULL or empty prompts
    # Include v_label so the reward function can score completions live
    # Include expected_verdict so the reward function can score verdict labels
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
    import json
    out_path = os.path.join(DATA_DIR, "training_data.json")
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Print summary
    rewards = [d["reference_reward"] for d in dataset]
    avg_r = sum(rewards) / len(rewards) if rewards else 0
    min_r = min(rewards) if rewards else 0
    max_r = max(rewards) if rewards else 0

    print("=" * 50)
    print("Phase 3: prepare_training_data() complete")
    print("=" * 50)
    print(f"  Total clean records:      {len(all_rows)}")
    print(f"  Excluded - double uncertain (v=0.5, p=0.4-0.6): {idk_excluded}")
    print(f"  Training examples:        {len(dataset)}")
    print(f"  Average reference reward: {avg_r:.4f}")
    print(f"  Min reference reward:     {min_r:.4f}")
    print(f"  Max reference reward:     {max_r:.4f}")
    print()
    # Verdict distribution
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
# Phase 4: run_grpo_training()
# ═══════════════════════════════════════════════════════════════

def run_grpo_training(smoke_test=True):
    """
    Phase 4: Fine-tune Qwen 2.5 7B Instruct with GRPO + LoRA.

    Args:
        smoke_test: If True, run 1 step only and skip saving.
                    If False, run full epoch and save adapter weights.
    """
    import json

    # ── Lazy imports (heavy dependencies) ──
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
    except ImportError as e:
        print(f"Phase 4: Missing dependency — {e}")
        print("  Install with: pip install torch transformers peft trl datasets bitsandbytes")
        return

    mode = "SMOKE TEST" if smoke_test else "FULL TRAINING"
    print("=" * 50)
    print(f"Phase 4: run_grpo_training() [{mode}]")
    print("=" * 50)

    # ── Load training data ──
    data_path = os.path.join(DATA_DIR, "training_data.json")
    if not os.path.exists(data_path):
        print(f"  ERROR: {data_path} not found. Run Phase 3 first.")
        return

    with open(data_path, "r") as f:
        raw_data = json.load(f)

    print(f"  Loaded {len(raw_data)} training examples")

    # ── Load model + tokenizer first (needed for chat template) ──
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"  Loading model: {model_name} (4-bit quantized)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batch generation

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── Apply Qwen chat template to prompts ──
    processed_prompts = []
    for d in raw_data:
        messages = json.loads(d["prompt"])
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processed_prompts.append(formatted)

    prompt_to_vlabel = {}
    prompt_to_expected_verdict = {}
    for i, d in enumerate(raw_data):
        prompt_to_vlabel[processed_prompts[i]] = d["v_label"]
        prompt_to_expected_verdict[processed_prompts[i]] = d.get("expected_verdict", "")

    hf_dataset = Dataset.from_list([{"prompt": p} for p in processed_prompts])
    print(f"  Prompts formatted with chat template ({len(processed_prompts)} examples)")

    # ── LoRA config ──
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"  LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
          f"targets={lora_config.target_modules}")

    # ── Reward function ──
    # Parse all claim_N/confidence_N pairs from completion.
    # Compute Brier reward per claim, return average.
    import re

    def _parse_claims_from_completion(text):
        """Parse all confidence values from structured completion."""
        confidences = []
        for match in re.finditer(r'confidence_\d+\s*:\s*([01]\.?\d*)', text, re.IGNORECASE):
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 1.0:
                    confidences.append(val)
            except (ValueError, IndexError):
                continue
        if not confidences:
            for match in re.finditer(r'[Cc]onfidence[\s:=]+([01]\.?\d*)', text):
                try:
                    val = float(match.group(1))
                    if 0.0 <= val <= 1.0:
                        confidences.append(val)
                except (ValueError, IndexError):
                    continue
        if not confidences:
            for match in re.finditer(r'\b(0\.\d{1,4})\b', text):
                try:
                    val = float(match.group(1))
                    if 0.0 <= val <= 1.0:
                        confidences.append(val)
                except (ValueError, IndexError):
                    continue
        return confidences

    def _parse_verdict_from_completion(text):
        """Parse verdict label from structured completion."""
        m = re.search(
            r'verdict_?\d*\s*:\s*(SUPPORTED|PARTIAL|NOT_SUPPORTED|UNSAFE|PII_LEAK)',
            text, re.IGNORECASE,
        )
        return m.group(1).upper() if m else None

    def reward_fn(completions, prompts=None, **kwargs):
        """Parse verdict + confidence from completion.
        reward = brier_score + verdict_bonus, clipped to [-1, +1]."""
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if prompts else None
            v_label = prompt_to_vlabel.get(prompt, 0.5) if prompt else 0.5
            expected_v = prompt_to_expected_verdict.get(prompt, "") if prompt else ""

            text = completion if isinstance(completion, str) else str(completion)
            confidences = _parse_claims_from_completion(text)

            # Brier component
            if not confidences:
                brier_score = -1.0
            else:
                brier_scores = [(2.0 * p * v_label) - (p ** 2) for p in confidences]
                brier_score = sum(brier_scores) / len(brier_scores)

            # Verdict bonus component
            parsed_verdict = _parse_verdict_from_completion(text)
            if parsed_verdict is None or not expected_v:
                verdict_bonus = 0.0
            elif parsed_verdict == expected_v:
                verdict_bonus = 0.2
            else:
                verdict_bonus = -0.2

            total = max(-1.0, min(1.0, brier_score + verdict_bonus))
            rewards.append(total)
        return rewards

    # ── GRPO config ──
    output_dir = os.path.join(DATA_DIR, "grpo_output")
    grpo_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_completion_length=256,
        temperature=1.0,
        num_generations=4,
        logging_steps=1,
        fp16=False,  # let quantized model handle precision
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    if smoke_test:
        grpo_kwargs["max_steps"] = 1
    else:
        grpo_kwargs["save_steps"] = 50

    grpo_config = GRPOConfig(**grpo_kwargs)

    # ── Trainer ──
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=lora_config,
    )

    print("  Starting training...")
    train_result = trainer.train()

    # ── Results ──
    loss = train_result.training_loss
    print(f"  Training loss: {loss:.4f}")

    if smoke_test:
        print("  Smoke test complete — 1 step ran successfully")
    else:
        save_path = os.path.join(output_dir, "final_adapter")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  Full training complete")
        print(f"  Adapter weights saved to: {save_path}")

    return train_result


# ═══════════════════════════════════════════════════════════════
# Phase 4B: run_grpo_training_agent_b()
# ═══════════════════════════════════════════════════════════════

def run_grpo_training_agent_b(smoke_test=True):
    """
    Fine-tune Agent B (challenger) with GRPO + LoRA.
    Reward: +1 if challenge targets weak/unsafe claim (v<1.0),
            -1 if challenge targets valid claim (v=1.0),
             0 if no meaningful challenge found.
    """
    import json
    import re

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
    except ImportError as e:
        print(f"Phase 4B: Missing dependency — {e}")
        return

    mode = "SMOKE TEST" if smoke_test else "FULL TRAINING"
    print("=" * 50)
    print(f"Phase 4B: run_grpo_training_agent_b() [{mode}]")
    print("=" * 50)

    # ── Load Agent B training data from attacks table ──
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT a.attack_id, a.query_id, a.rollout_id, a.claim_id,
               a.agent_b_prompt, a.b_challenge_type,
               jv.v_label
        FROM attacks a
        JOIN judge_verdicts jv
            ON a.query_id = jv.query_id
            AND a.rollout_id = jv.rollout_id
            AND a.claim_id = jv.claim_id
        WHERE a.agent_b_prompt IS NOT NULL
          AND a.agent_b_prompt != ''
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("  ERROR: No Agent B training data found.")
        return

    # Truncate prompts
    MAX_PROMPT_CHARS = 1500
    training_data = []
    for row in rows:
        prompt = row["agent_b_prompt"]
        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[:MAX_PROMPT_CHARS]
        training_data.append({
            "prompt": prompt,
            "v_label": row["v_label"],
        })

    prompt_to_vlabel = {d["prompt"]: d["v_label"] for d in training_data}
    hf_dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in training_data])
    print(f"  Loaded {len(training_data)} Agent B training examples")

    # ── Load model ──
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"  Loading model: {model_name} (4-bit quantized)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    # ── Agent B reward function ──
    # Parses challenge quality from completion.
    # +1 if challenge targets weak/unsafe claim (v < 1.0) — good challenge
    # -1 if challenge targets valid claim (v == 1.0) — gaslighting
    #  0 if no meaningful challenge found
    CHALLENGE_KEYWORDS = [
        "contradicted", "unsupported", "hallucinated", "no evidence",
        "not supported", "fabricated", "citation gap", "scope error",
        "unsafe", "harmful", "pii", "injection", "leak",
    ]

    def reward_fn_b(completions, prompts=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if prompts else None
            v_label = prompt_to_vlabel.get(prompt, 0.5) if prompt else 0.5

            text = completion if isinstance(completion, str) else str(completion)
            text_lower = text.lower()

            # Check if completion contains a meaningful challenge
            has_challenge = any(kw in text_lower for kw in CHALLENGE_KEYWORDS)

            if not has_challenge:
                rewards.append(0.0)
            elif v_label < 1.0:
                # Good: challenged a weak/wrong/unsafe claim
                rewards.append(1.0)
            else:
                # Bad: challenged a valid claim (gaslighting)
                rewards.append(-1.0)
        return rewards

    # ── GRPO config ──
    output_dir = os.path.join(DATA_DIR, "grpo_output")
    grpo_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_completion_length=256,
        temperature=1.0,
        num_generations=4,
        logging_steps=1,
        fp16=False, bf16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    if smoke_test:
        grpo_kwargs["max_steps"] = 1
    else:
        grpo_kwargs["save_steps"] = 50

    grpo_config = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn_b,
        peft_config=lora_config,
    )

    print("  Starting Agent B training...")
    train_result = trainer.train()
    loss = train_result.training_loss
    print(f"  Training loss: {loss:.4f}")

    if smoke_test:
        print("  Smoke test complete — 1 step ran successfully")
    else:
        save_path = os.path.join(output_dir, "agent_b_adapter")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  Full training complete")
        print(f"  Agent B adapter saved to: {save_path}")

    return train_result


# ═══════════════════════════════════════════════════════════════
# Phase 5: evaluate()
# ═══════════════════════════════════════════════════════════════

def evaluate():
    """Phase 5: Before/after comparison, Brier improvement, calibration. Not yet implemented."""
    print("Phase 5: evaluate() — not yet implemented")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found. Run phase1_synthetic_data.py first.")
        return

    compute_rewards()
    prepare_training_data()
    run_grpo_training(smoke_test=True)
    evaluate()


if __name__ == "__main__":
    main()
