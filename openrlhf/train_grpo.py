"""
Standalone GRPO training script for Guardrails Agent A fine-tuning.

Mirrors the Colab notebook but runs from CLI — suitable for local GPU, cloud VMs,
or any environment with a CUDA device. No Jupyter required.

Usage:
    python -m openrlhf.train_grpo \
        --dataset /path/to/grpo_train.jsonl \
        --output  ./runs/guardrails-grpo \
        [--model   Qwen/Qwen2.5-7B-Instruct] \
        [--steps   150] \
        [--no-presidio]

Generate the JSONL first:
    python -m rlhf.pipeline.export_grpo_dataset --db rlhf/mad_store.db

See docs/RLHF_IMPLEMENTATION.md for full hyperparameter rationale.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt (matches RLHF_IMPLEMENTATION.md §9)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Agent A in a Multi-Agent Debate pipeline for healthcare regulatory AI. "
    "Your role is to provide a well-reasoned, evidence-based claim with a calibrated confidence score.\n"
    "RULES:\n"
    "(1) Never include PHI or patient identifiers.\n"
    "(2) Always cite a specific source: HIPAA section, CFR part, drug label, or peer-reviewed study.\n"
    "(3) End your response with: CONFIDENCE: <float 0.0-1.0>.\n"
    "(4) If uncertain, express lower confidence — do not hallucinate certainty."
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(jsonl_path: str, tokenizer):
    from datasets import Dataset

    rows = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No rows found in {jsonl_path}. Run export_grpo_dataset first.")

    def make_prompt(query: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return Dataset.from_list([
        {
            "prompt":        make_prompt(r["prompt"]),
            "judge_verdict": float(r.get("judge_verdict", 0.5)),
        }
        for r in rows
    ])


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def build_reward_fn(use_presidio: bool = True):
    from openrlhf.reward_fn.healthcare_reward import scalar_reward

    def grpo_reward_fn(completions, **kwargs):
        judge_verdicts = kwargs.get("judge_verdict", [0.5] * len(completions))
        return [
            scalar_reward(c, float(v), use_presidio=use_presidio)
            for c, v in zip(completions, judge_verdicts)
        ]

    return grpo_reward_fn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import GRPOConfig, GRPOTrainer

    log.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    log.info("Loading dataset: %s", args.dataset)
    ds = load_dataset(args.dataset, tokenizer)
    log.info("Dataset: %d examples", len(ds))

    log.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA — §8 config
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    reward_fn = build_reward_fn(use_presidio=not args.no_presidio)

    training_args = GRPOConfig(
        output_dir=args.output,
        # Training duration
        max_steps=args.steps,
        # Batch — tuned for A100 40 GB with 7B model
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=2,
        max_completion_length=256,
        # Optimiser
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        # RL
        beta=0.05,
        # Precision
        bf16=True,
        gradient_checkpointing=True,
        # Logging / checkpointing
        logging_steps=10,
        save_steps=50,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    log.info("Starting GRPO training — %d steps", args.steps)
    trainer.train()

    final_dir = pathlib.Path(args.output) / "final"
    trainer.save_model(str(final_dir))
    log.info("Model saved to %s", final_dir)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from openrlhf.reward_fn.healthcare_reward import scalar_reward, extract_confidence, has_citation

    EVAL_PROMPTS = [
        "What are the HIPAA technical safeguard requirements under 45 CFR §164.312?",
        "When must an adverse event be reported to the FDA under 21 CFR §803?",
        "What constitutes a HIPAA breach and what are the notification timelines?",
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, str(pathlib.Path(args.output) / "final"))
    model.eval()

    def make_prompt(query: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("\n=== Post-training evaluation ===\n")
    for prompt in EVAL_PROMPTS:
        inputs = tokenizer(make_prompt(prompt), return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        conf = extract_confidence(completion)
        cited = has_citation(completion)
        reward = scalar_reward(completion, judge_verdict=0.8)
        print(f"Prompt : {prompt[:70]}...")
        print(f"Conf   : {conf}   Cited: {cited}   Reward: {reward:.4f}")
        print(f"Output : {completion[:300]}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning for Guardrails Agent A (Qwen2.5-7B + LoRA)"
    )
    parser.add_argument("--dataset",     required=True,  help="Path to grpo_train.jsonl")
    parser.add_argument("--output",      default="./runs/guardrails-grpo", help="Output directory")
    parser.add_argument("--model",       default="Qwen/Qwen2.5-7B-Instruct", help="HF model ID or local path")
    parser.add_argument("--steps",       type=int, default=150, help="Training steps (default: 150)")
    parser.add_argument("--no-presidio", action="store_true", help="Disable Presidio PHI detection in reward")
    parser.add_argument("--eval-only",   action="store_true", help="Skip training, only run post-training eval")
    args = parser.parse_args()

    if not args.eval_only:
        train(args)
    evaluate(args)


if __name__ == "__main__":
    main()
