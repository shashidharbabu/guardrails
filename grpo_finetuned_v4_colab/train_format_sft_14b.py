#!/usr/bin/env python3
"""
Short 14B JSON/schema warm-up before GRPO.

This is not quality SFT. It only teaches Qwen2.5-14B-Instruct to emit compact
valid JSON with the required debate-agent schema before Brier GRPO calibration.
It uses Unsloth 4-bit QLoRA and transformers.Trainer, not TRL SFTTrainer.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MAX_SEQ_LEN = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
SEED = 42

GENERATION_CONTRACT = """Return valid JSON only:
{
  "verdict": "SUPPORTED | PARTIAL | NOT_SUPPORTED | IDK",
  "reasoning": "short reasoning",
  "evidence_cited": [{"chunk_id": "...", "relevant_quote": "..."}],
  "confidence_internal": 0.0
}

Rules:
- No markdown.
- No prose outside JSON.
- reasoning max 20 words.
- evidence_cited may be [].
- confidence_internal is NOT confidence in your chosen verdict.
- confidence_internal is your estimated probability that the claim is fully SUPPORTED by the evidence.
- Calibration:
  - SUPPORTED: usually 0.75-1.00
  - PARTIAL: usually 0.35-0.65
  - NOT_SUPPORTED: usually 0.00-0.25
  - IDK: usually 0.25-0.55
- Stop immediately after the closing }."""


def _prepare_prompt(raw_prompt: Any, tokenizer: Any) -> str:
    if isinstance(raw_prompt, list):
        return tokenizer.apply_chat_template(
            raw_prompt,
            tokenize=False,
            add_generation_prompt=True,
        ).rstrip()
    return str(raw_prompt).rstrip()


def _synthetic_completion(v_label: float) -> str:
    if v_label == 1.0:
        verdict = "SUPPORTED"
        confidence = 0.90
        reasoning = "Evidence directly supports the claim."
    elif v_label == 0.5:
        verdict = "PARTIAL"
        confidence = 0.50
        reasoning = "Evidence partially supports the claim."
    elif v_label == 0.0:
        verdict = "NOT_SUPPORTED"
        confidence = 0.10
        reasoning = "Evidence does not fully support the claim."
    else:
        verdict = "IDK"
        confidence = 0.40
        reasoning = "Evidence is insufficient to determine support."

    return json.dumps(
        {
            "verdict": verdict,
            "reasoning": reasoning,
            "evidence_cited": [],
            "confidence_internal": confidence,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def load_rows(path: Path, tokenizer: Any) -> list[dict[str, str]]:
    rows = []
    counts = {0.0: 0, 0.5: 0, 1.0: 0}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            v_label = float(row["v_label"])
            counts[v_label] = counts.get(v_label, 0) + 1
            prompt = _prepare_prompt(row["prompt"], tokenizer) + "\n\n" + GENERATION_CONTRACT
            rows.append({"prompt": prompt, "completion": _synthetic_completion(v_label)})

    print(f"[Data] rows={len(rows)} from {path}")
    print(
        f"  NOT_SUPPORTED={counts[0.0]}  PARTIAL={counts[0.5]}  SUPPORTED={counts[1.0]}"
    )
    return rows


class CompletionOnlyDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        tokenizer: Any,
        *,
        max_seq_length: int,
        max_prompt_length: int,
        max_completion_length: int,
    ):
        self.examples = []
        eos = tokenizer.eos_token or ""
        prompt_limit = min(max_prompt_length, max_seq_length - max_completion_length)
        if prompt_limit < 256:
            raise ValueError("max_seq_length must leave room for completion tokens")

        clipped_prompts = 0
        for row in rows:
            prompt_text = row["prompt"].rstrip() + "\n"
            completion_text = row["completion"].strip() + eos

            prompt_enc = tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=prompt_limit,
            )
            if len(prompt_enc["input_ids"]) >= prompt_limit:
                clipped_prompts += 1

            completion_enc = tokenizer(
                completion_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_completion_length,
            )
            input_ids = prompt_enc["input_ids"] + completion_enc["input_ids"]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_enc["input_ids"]) + completion_enc["input_ids"]

            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
                attention_mask = attention_mask[:max_seq_length]
                labels = labels[:max_seq_length]

            if all(x == -100 for x in labels):
                continue

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        print(
            f"[Data] tokenized examples={len(self.examples)} "
            f"prompt_limit={prompt_limit} clipped_prompts={clipped_prompts}"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {k: torch.tensor(v, dtype=torch.long) for k, v in ex.items()}


def collate_completion_only(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for ex in batch:
        n = len(ex["input_ids"])
        pad = max_len - n
        out["input_ids"].append(ex["input_ids"].tolist() + [pad_id] * pad)
        out["attention_mask"].append(ex["attention_mask"].tolist() + [0] * pad)
        out["labels"].append(ex["labels"].tolist() + [-100] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def train(
    *,
    data_path: Path,
    output_dir: Path,
    max_steps: int,
    save_steps: int,
    logging_steps: int,
    max_prompt_length: int,
    max_completion_length: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    from functools import partial

    from unsloth import FastLanguageModel
    from transformers import Trainer, TrainingArguments

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"14B format warm-up model={MODEL_NAME}")
    print(f"data={data_path}")
    print(f"output={output_dir}")
    print(f"use_vllm=False")
    print(f"{'=' * 60}\n")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    model.print_trainable_parameters()

    rows = load_rows(data_path, tokenizer)
    dataset = CompletionOnlyDataset(
        rows,
        tokenizer,
        max_seq_length=MAX_SEQ_LEN,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )
    if not dataset:
        raise RuntimeError("No usable format warm-up examples after tokenization.")

    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        bf16=True,
        fp16=False,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=partial(collate_completion_only, pad_id=tokenizer.pad_token_id),
    )

    result = trainer.train()
    print(f"[SFT] train_loss={result.training_loss:.6f}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[SFT] final adapter saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Short 14B JSON-format warm-up before GRPO")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "grpo_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "format_sft_combined")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        batch_size=args.batch_size,
        grad_accum=args.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    main()
