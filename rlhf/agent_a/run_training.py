#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  Enterprise Guardrail Feedback Loop — Colab Runner
═══════════════════════════════════════════════════════════════

HOW TO USE:
  1. Open Google Colab (ideally Colab Pro with A100/T4 GPU)
  2. Set runtime → GPU (T4 or A100)
  3. Upload these files to Colab's /content/ directory:
       - run_on_colab.py      (this file)
       - guardrails.db        (from Phase 1)
       - training_data.json   (from Phase 3)
  4. Run this entire script: !python run_on_colab.py

WHAT IT DOES:
  - Installs all required packages
  - Runs a 1-step smoke test to verify everything works
  - Asks you to confirm before running full training
  - Saves adapter weights to Google Drive (if mounted)
  - Falls back to /content/grpo_output/ if Drive not available

EXPECTED RUNTIME:
  - Smoke test: ~2-3 minutes (mostly model loading)
  - Full training (35 examples, 1 epoch): ~15-30 min on T4
═══════════════════════════════════════════════════════════════
"""

import subprocess
import sys
import os
import json

SKIP_CONFIRMATION = True  # Set to False to require manual confirmation

# ═══════════════════════════════════════════════════════════════
# Step 1: Install dependencies
# ═══════════════════════════════════════════════════════════════

def install_packages():
    packages = [
        "torch",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "trl>=0.8.0",
        "datasets",
        "accelerate",
        "bitsandbytes",
    ]
    print("=" * 60)
    print("Step 1: Installing packages")
    print("=" * 60)
    for pkg in packages:
        print(f"  Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print("  All packages installed.\n")


# ═══════════════════════════════════════════════════════════════
# Step 2: Verify uploaded files
# ═══════════════════════════════════════════════════════════════

def verify_files():
    print("=" * 60)
    print("Step 2: Verifying uploaded files")
    print("=" * 60)

    # Check multiple possible locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = ["/content", os.getcwd(), script_dir, os.path.join(script_dir, "..", "data")]
    # Look for database — accept either name
    db_names = ["mad_store_working.db", "mad_store.db", "guardrails.db"]
    found = {}
    for fname in db_names:
        for d in search_dirs:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                found["guardrails.db"] = path
                break
        if "guardrails.db" in found:
            break

    # Look for real training data first, fall back to synthetic
    training_data_names = ["real_training_data.json", "training_data.json"]
    for fname in training_data_names:
        for d in search_dirs:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                found["training_data.json"] = path
                break
        if "training_data.json" in found:
            break

    all_needed = ["guardrails.db", "training_data.json"]
    for fname in all_needed:
        if fname in found:
            size = os.path.getsize(found[fname])
            print(f"  FOUND: {found[fname]} ({size:,} bytes)")
        else:
            print(f"  MISSING: {fname}")
            print(f"  Upload {fname} to /content/ and re-run.")
            sys.exit(1)

    # Validate training data
    with open(found["training_data.json"], "r") as f:
        data = json.load(f)
    print(f"  Training data: {len(data)} examples loaded")

    if len(data) == 0:
        print("  ERROR: training_data.json is empty.")
        sys.exit(1)

    print()
    return found


# ═══════════════════════════════════════════════════════════════
# Step 3: Run training (smoke test or full)
# ═══════════════════════════════════════════════════════════════

def run_training(data_path, smoke_test=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    mode = "SMOKE TEST (1 step)" if smoke_test else "FULL TRAINING"
    print("=" * 60)
    print(f"Step 3: {mode}")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  WARNING: No GPU detected. Training will be very slow.")

    # ── Load training data ──
    with open(data_path, "r") as f:
        raw_data = json.load(f)

    print(f"  Dataset: {len(raw_data)} examples")

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
    print("  Model loaded.")

    # ── Apply Qwen chat template to prompts ──
    # Synthetic data stores prompts as JSON message lists: [{"role":"system","content":"..."}]
    # Real MAD data stores prompts as raw text strings: "System: You are Agent A..."
    # Handle both formats.
    processed_prompts = []
    for d in raw_data:
        prompt_raw = d["prompt"]
        try:
            messages = json.loads(prompt_raw)
            # Successfully parsed as JSON — apply chat template
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (json.JSONDecodeError, TypeError):
            # Raw text prompt from real MAD database — use directly
            formatted = prompt_raw
        processed_prompts.append(formatted)

    # Build formatted prompt → v_label and expected_verdict lookups
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
    print(f"  LoRA: r=16, alpha=32, targets=['q_proj', 'v_proj']")

    # ── Reward function ──
    # Parse all claim_N/confidence_N pairs from completion.
    # Compute Brier reward per claim, return average.
    import re

    def _parse_claims_from_completion(text):
        """Parse all (claim_text, confidence) pairs from structured completion.
        Returns list of floats (confidence values found)."""
        confidences = []
        # Match: confidence_1: 0.85, confidence_2: 0.42, etc.
        for match in re.finditer(r'confidence_\d+\s*:\s*([01]\.?\d*)', text, re.IGNORECASE):
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 1.0:
                    confidences.append(val)
            except (ValueError, IndexError):
                continue

        # Fallback: look for any confidence/score patterns
        if not confidences:
            for match in re.finditer(r'[Cc]onfidence[\s:=]+([01]\.?\d*)', text):
                try:
                    val = float(match.group(1))
                    if 0.0 <= val <= 1.0:
                        confidences.append(val)
                except (ValueError, IndexError):
                    continue

        # Last resort: any decimal between 0 and 1
        if not confidences:
            for match in re.finditer(r'\b(0\.\d{1,4})\b', text):
                try:
                    val = float(match.group(1))
                    if 0.0 <= val <= 1.0:
                        confidences.append(val)
                except (ValueError, IndexError):
                    continue

        return confidences

    VERDICT_LABELS = {"SUPPORTED", "PARTIAL", "NOT_SUPPORTED", "UNSAFE", "PII_LEAK"}

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

            # ── Brier component ──
            if not confidences:
                brier_score = -1.0
            else:
                brier_scores = [(2.0 * p * v_label) - (p ** 2) for p in confidences]
                brier_score = sum(brier_scores) / len(brier_scores)

            # ── Verdict bonus component ──
            parsed_verdict = _parse_verdict_from_completion(text)
            if parsed_verdict is None or not expected_v:
                verdict_bonus = 0.0
            elif parsed_verdict == expected_v:
                verdict_bonus = 0.2
            else:
                verdict_bonus = -0.2

            # Combine and clip
            total = max(-1.0, min(1.0, brier_score + verdict_bonus))
            rewards.append(total)
        return rewards

    # ── GRPO config ──
    output_dir = "/content/grpo_output" if os.path.isdir("/content") else "./grpo_output"

    grpo_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_completion_length=512,
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

    # ── Train ──
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=lora_config,
    )

    print("  Training started...")
    train_result = trainer.train()
    loss = train_result.training_loss
    print(f"  Training loss: {loss:.4f}")

    if smoke_test:
        print("  Smoke test complete — 1 step ran successfully\n")
    else:
        save_path = os.path.join(output_dir, "final_adapter")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  Full training complete.")
        print(f"  Adapter saved to: {save_path}\n")

    return trainer, output_dir


# ═══════════════════════════════════════════════════════════════
# Step 4: Save to Google Drive (optional)
# ═══════════════════════════════════════════════════════════════

def save_to_drive(output_dir):
    print("=" * 60)
    print("Step 4: Save to Google Drive")
    print("=" * 60)

    drive_path = "/content/drive/MyDrive"

    # Try mounting Drive
    if not os.path.isdir(drive_path):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception:
            print("  Google Drive not available. Skipping.")
            print(f"  Weights are saved locally at: {output_dir}")
            return

    if os.path.isdir(drive_path):
        import shutil
        dest = os.path.join(drive_path, "guardrail_grpo_adapter")
        src = os.path.join(output_dir, "final_adapter")
        if os.path.isdir(src):
            if os.path.isdir(dest):
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            print(f"  Adapter copied to: {dest}")
        else:
            print(f"  No adapter found at {src}. Run full training first.")
    else:
        print(f"  Drive mount failed. Weights at: {output_dir}")
    print()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print()
    print("  Enterprise Guardrail Feedback Loop")
    print("  GRPO Fine-tuning — Colab Runner")
    print()

    # Step 1: Install
    install_packages()

    # Step 2: Verify files
    found = verify_files()
    data_path = found["training_data.json"]

    # Step 3a: Smoke test
    print("Running smoke test first (1 training step)...\n")
    trainer, output_dir = run_training(data_path, smoke_test=True)

    # Step 3b: Ask before full training
    print("=" * 60)
    print("Smoke test passed. Ready for full training.")
    print("=" * 60)

    if SKIP_CONFIRMATION:
        print("  SKIP_CONFIRMATION=True — proceeding to full training automatically")
        answer = "yes"
    else:
        try:
            answer = input("\n  Run full training? (yes/no): ").strip().lower()
        except EOFError:
            answer = "no"

    if answer in ("yes", "y"):
        # Free memory from smoke test
        del trainer
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc
        gc.collect()

        trainer, output_dir = run_training(data_path, smoke_test=False)

        # Step 4: Save to Drive
        save_to_drive(output_dir)

        print("=" * 60)
        print("All done!")
        print("=" * 60)
        print(f"  Adapter weights: {output_dir}/final_adapter")
        print("  You can load them with:")
        print("    from peft import PeftModel")
        print("    model = PeftModel.from_pretrained(base_model, "
              f"'{output_dir}/final_adapter')")
    else:
        print("\n  Skipped full training. You can re-run anytime.")


if __name__ == "__main__":
    main()
