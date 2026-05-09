#!/usr/bin/env python3
"""
GRPO fine-tuning for MAD debate agents.

GRPO starts directly from Qwen2.5-14B-Instruct with 4-bit QLoRA and a fresh
trainable LoRA adapter initialized by Unsloth. Single model instance.

Run:
    python train_grpo.py --agent a
    python train_grpo.py --agent b
    python train_grpo.py --agent a --resume-from-checkpoint outputs/grpo_agent_a/checkpoint-50
"""

import argparse
import hashlib
import inspect
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def _load_env_files() -> None:
    """Load optional Langfuse/API env files without making them required."""
    for env_path in (
        Path("/app/finalMAD/.env"),
        Path("/app/finalMAD/grpo_finetune_package/.env"),
        ROOT.parent / ".env",
        ROOT / ".env",
    ):
        try:
            if env_path.exists():
                load_dotenv(env_path, override=False)
                print(f"[env] Loaded {env_path}")
        except Exception as exc:
            print(f"[env] Skipped {env_path}: {exc}")


_load_env_files()

# ─── Paths ─────────────────────────────────────────────────────────────────────

DATA_FILES = {
    "agent_a": ROOT / "data" / "agent_a_grpo.jsonl",
    "agent_b": ROOT / "data" / "agent_b_grpo.jsonl",
    "combined": ROOT / "data" / "grpo_train.jsonl",
}
OUTPUT_DIRS = {
    "agent_a": ROOT / "outputs" / "grpo_agent_a",
    "agent_b": ROOT / "outputs" / "grpo_agent_b",
    "combined": ROOT / "outputs" / "grpo_combined",
}

MODEL_NAME   = "Qwen/Qwen2.5-14B-Instruct"
MAX_SEQ_LEN  = 2048
LORA_R       = 16
LORA_ALPHA   = 16

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

# ─── Global lookup & stats ─────────────────────────────────────────────────────

# Built once at dataset load. Used inside reward_fn instead of relying on kwargs.
_prompt_to_vlabel: dict[str, float] = {}
_prompt_to_meta: dict[str, dict] = {}
_reward_call_count = 0
_reward_log_every = 5
_reward_tokenizer = None

_stats = {
    "n":            0,
    "reward_sum":   0.0,
    "invalid_json": 0,
    "valid_json":   0,
    "required_keys": 0,
    "valid_conf":   0,
    "missing_conf": 0,
    "invalid_conf": 0,
}

# ─── Langfuse (optional) ───────────────────────────────────────────────────────

_langfuse = None


def _init_langfuse() -> None:
    global _langfuse
    try:
        from langfuse import Langfuse
        pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sk = os.getenv("LANGFUSE_SECRET_KEY", "")
        if not pk or not sk:
            print("[Langfuse] Keys not set — tracing disabled.")
            return
        host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        _langfuse = Langfuse(public_key=pk, secret_key=sk, host=host)
        print(f"[Langfuse] Tracing enabled → {host}")
    except Exception as exc:
        print(f"[Langfuse] Init failed ({exc}) — tracing disabled.")


def _lf_run_span(
    agent: str,
    dataset_size: int,
    *,
    num_generations: int,
    max_completion_length: int,
) -> None:
    if _langfuse is None:
        return
    try:
        _langfuse.trace(
            name=f"grpo-run-{agent}",
            input={
                "agent":         agent,
                "dataset_size":  dataset_size,
                "model":         MODEL_NAME,
                "lora_r":        LORA_R,
                "num_generations": num_generations,
                "learning_rate": 5e-6,
                "max_completion_length": max_completion_length,
            },
        )
    except Exception:
        pass


def _lf_reward_sample(
    *,
    prompt:       str,
    completion:   str,
    v_label:      float,
    p:            Optional[float],
    reward:       float,
    claim_id:     str,
    query_id:     str,
    invalid_json: bool,
    missing_conf: bool,
    invalid_conf: bool,
) -> None:
    if _langfuse is None:
        return
    try:
        _langfuse.generation(
            name="grpo-reward-sample",
            input={
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8],
                "claim_id":    claim_id,
                "query_id":    query_id,
                "v_label":     v_label,
            },
            output={
                "reward":              reward,
                "confidence_internal": p,
                "invalid_json":        invalid_json,
                "missing_confidence":  missing_conf,
                "invalid_confidence":  invalid_conf,
                "completion_prefix":   completion[:200],
            },
        )
    except Exception:
        pass


def _lf_periodic_stats() -> None:
    if _langfuse is None or _stats["n"] == 0:
        return
    try:
        mean_r    = _stats["reward_sum"]   / _stats["n"]
        inv_rate  = _stats["invalid_json"] / _stats["n"]
        miss_rate = _stats["missing_conf"] / _stats["n"]
        _langfuse.trace(
            name="grpo-periodic-stats",
            input={"n": _stats["n"]},
            output={
                "mean_reward":      mean_r,
                "invalid_json_rate": inv_rate,
                "missing_conf_rate": miss_rate,
            },
        )
    except Exception:
        pass


# ─── JSON parsing ──────────────────────────────────────────────────────────────

_MD_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_REQUIRED_AGENT_KEYS = {
    "verdict",
    "reasoning",
    "evidence_cited",
    "confidence_internal",
}


def _completion_to_text(completion: Any) -> str:
    """Normalize TRL completion payloads across plain and chat-style APIs."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def _parse_agent_json(text: str) -> Any:
    """
    Three-pass parser. Returns dict with agent keys or None.
    Starting from a base model, ~30–40% of outputs will fail — that is expected.
    """
    text = _completion_to_text(text).strip()

    # Pass 1: direct parse (works once model learns format)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Pass 2: strip markdown code fence
    m = _MD_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Pass 3: extract first balanced { ... } from anywhere in the text
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _extract_agent_fields(parsed: Any) -> dict:
    """
    Keep the schema surface explicit. Missing fields are left as None so reward
    penalties can distinguish malformed JSON from a missing confidence field.
    """
    if not isinstance(parsed, dict):
        return {
            "verdict": None,
            "reasoning": None,
            "evidence_cited": None,
            "confidence_internal": None,
            "has_confidence_internal": False,
            "has_all_required_keys": False,
        }
    return {
        "verdict": parsed.get("verdict"),
        "reasoning": parsed.get("reasoning"),
        "evidence_cited": parsed.get("evidence_cited"),
        "confidence_internal": parsed.get("confidence_internal"),
        "has_confidence_internal": "confidence_internal" in parsed,
        "has_all_required_keys": _REQUIRED_AGENT_KEYS.issubset(parsed.keys()),
    }


def _load_init_adapter(model, adapter_dir: Path) -> None:
    adapter_dir = adapter_dir.resolve()
    cfg_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not cfg_path.exists() or not weights_path.exists():
        sys.exit(
            f"[ERROR] --init-adapter must contain adapter_config.json and "
            f"adapter_model.safetensors: {adapter_dir}"
        )

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    base = str(cfg.get("base_model_name_or_path", ""))
    if "7B" in base or "7b" in base:
        sys.exit(
            f"[ERROR] Refusing incompatible 7B init adapter for 14B GRPO: {adapter_dir}\n"
            f"  adapter base_model_name_or_path={base}"
        )
    if base and "14B" not in base and "14b" not in base and MODEL_NAME not in base:
        sys.exit(
            f"[ERROR] Init adapter base model does not look like Qwen2.5-14B: {adapter_dir}\n"
            f"  adapter base_model_name_or_path={base}"
        )

    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_state = load_file(str(weights_path))
    model_state = model.state_dict()
    missing = []
    mismatched = []
    for key, tensor in adapter_state.items():
        candidates = (
            key,
            key.replace(".weight", ".default.weight"),
            key.replace(".default.weight", ".weight"),
        )
        model_key = next((k for k in candidates if k in model_state), None)
        if model_key is None:
            missing.append(key)
            continue
        if tuple(model_state[model_key].shape) != tuple(tensor.shape):
            mismatched.append(
                f"{key}: adapter {tuple(tensor.shape)} vs model {tuple(model_state[model_key].shape)}"
            )

    if missing or mismatched:
        detail = []
        if missing:
            detail.append("missing keys, first 5: " + ", ".join(missing[:5]))
        if mismatched:
            detail.append("shape mismatches, first 5: " + " | ".join(mismatched[:5]))
        sys.exit(
            f"[ERROR] Init adapter is incompatible with current 14B LoRA model: {adapter_dir}\n"
            + "\n".join(detail)
        )

    result = set_peft_model_state_dict(model, adapter_state, adapter_name="default")
    print(f"[Init adapter] Loaded {adapter_dir}")
    if result is not None:
        print(f"[Init adapter] PEFT load report: {result}")


# ─── Reward function ───────────────────────────────────────────────────────────

_LOG_RATE = 0.03   # 3% of completions sampled to Langfuse — keep memory flat


def brier_reward(
    completions: list[str],
    prompts:     list[str] = None,
    **kwargs,
) -> list[float]:
    """
    Brier reward: R = 2 * p * v - p²

    p = confidence_internal from generated JSON (0–1)
    v = v_label from dataset, fetched via global lookup (NOT from kwargs)

    Penalties / rewards:
      invalid JSON                         → -1.0
      valid JSON but missing confidence    → -1.0
      valid JSON but missing other keys    → -0.7
      required keys but bad confidence     → -0.5
      valid confidence                     → Brier + 0.1 format bonus, capped [-1, 1]
    """
    # TRL may pass prompts positionally or as kwarg; handle both
    if prompts is None:
        prompts = kwargs.get("prompts", [""] * len(completions))

    global _reward_call_count
    _reward_call_count += 1
    rewards: list[float] = []
    batch_invalid_json = 0
    batch_valid_json = 0
    batch_required_keys = 0
    batch_valid_conf = 0
    batch_invalid_conf = 0
    batch_completion_token_lengths: list[int] = []

    for i, (completion, prompt) in enumerate(zip(completions, prompts)):
        prompt = _completion_to_text(prompt)
        completion_text = _completion_to_text(completion)
        if _reward_tokenizer is not None:
            try:
                batch_completion_token_lengths.append(
                    len(_reward_tokenizer(completion_text, add_special_tokens=False)["input_ids"])
                )
            except Exception:
                pass
        v = _prompt_to_vlabel.get(prompt)
        if v is None:
            # Should never happen — all training prompts are in the lookup
            rewards.append(-1.0)
            _stats["n"] += 1
            continue

        parsed       = _parse_agent_json(completion_text)
        invalid_json = parsed is None
        missing_keys = False
        invalid_conf = False
        p: Optional[float] = None

        if invalid_json:
            reward = -1.0
            _stats["invalid_json"] += 1
            batch_invalid_json += 1

        else:
            _stats["valid_json"] += 1
            batch_valid_json += 1
            fields = _extract_agent_fields(parsed)

            if not fields["has_confidence_internal"]:
                missing_keys = True
                reward = -1.0
                _stats["missing_conf"] += 1

            elif not fields["has_all_required_keys"]:
                missing_keys = True
                reward = -0.7

            else:
                _stats["required_keys"] += 1
                batch_required_keys += 1
                raw = fields["confidence_internal"]
                try:
                    p = float(raw)
                    if not (0.0 <= p <= 1.0):
                        raise ValueError(f"out of range: {p}")
                    reward = max(-1.0, min(1.0, 2.0 * p * v - p ** 2 + 0.1))
                    _stats["valid_conf"] += 1
                    batch_valid_conf += 1
                except (TypeError, ValueError):
                    invalid_conf = True
                    reward = -0.5
                    _stats["invalid_conf"] += 1
                    batch_invalid_conf += 1

        rewards.append(reward)
        _stats["n"]          += 1
        _stats["reward_sum"] += reward

        # Langfuse: sample 3%
        if random.random() < _LOG_RATE:
            meta = _prompt_to_meta.get(prompt, {})
            raw_cid = kwargs.get("claim_id", meta.get("claim_id", ""))
            raw_qid = kwargs.get("query_id", meta.get("query_id", ""))
            cid = raw_cid[i] if isinstance(raw_cid, list) and i < len(raw_cid) else str(raw_cid)
            qid = raw_qid[i] if isinstance(raw_qid, list) and i < len(raw_qid) else str(raw_qid)
            _lf_reward_sample(
                prompt=prompt, completion=completion_text,
                v_label=v, p=p, reward=reward,
                claim_id=cid, query_id=qid,
                invalid_json=invalid_json,
                missing_conf=missing_keys,
                invalid_conf=invalid_conf,
            )

    if rewards and _reward_call_count % max(1, _reward_log_every) == 0:
        n = _stats["n"]
        batch_n = len(rewards)
        mean_r = sum(rewards) / batch_n
        cumulative_mean_r = _stats["reward_sum"] / n if n else 0.0
        batch_inv_rate = batch_invalid_json / batch_n
        batch_valid_json_rate = batch_valid_json / batch_n
        batch_required_keys_rate = batch_required_keys / batch_n
        batch_valid_conf_rate = batch_valid_conf / batch_n
        batch_bad_conf_rate = batch_invalid_conf / batch_n
        cumulative_inv_rate = _stats["invalid_json"] / n if n else 0.0
        avg_completion_tokens = (
            sum(batch_completion_token_lengths) / len(batch_completion_token_lengths)
            if batch_completion_token_lengths else 0.0
        )
        print(
            f"[GRPO-REWARD] step={_reward_call_count} completions={batch_n} "
            f"reward_mean={mean_r:+.4f} cumulative_reward_mean={cumulative_mean_r:+.4f} "
            f"valid_json_rate={batch_valid_json_rate:.1%} required_keys_rate={batch_required_keys_rate:.1%} "
            f"valid_confidence_rate={batch_valid_conf_rate:.1%} invalid_json_rate={batch_inv_rate:.1%} "
            f"cumulative_invalid_json_rate={cumulative_inv_rate:.1%} invalid_confidence_rate={batch_bad_conf_rate:.1%} "
            f"avg_completion_tokens={avg_completion_tokens:.1f}"
        )
        _lf_periodic_stats()

    return rewards


# ─── Dataset ───────────────────────────────────────────────────────────────────

def _prepare_prompt(raw_prompt: Any, tokenizer: Any) -> str:
    """Support legacy string prompts and V4MAD chat-message prompt lists."""
    if isinstance(raw_prompt, list):
        return tokenizer.apply_chat_template(
            raw_prompt,
            tokenize=False,
            add_generation_prompt=True,
        ).rstrip()
    return str(raw_prompt).rstrip()


def _truncate_prompt_for_context(
    prompt: str,
    tokenizer: Any,
    *,
    max_prompt_length: int,
    max_completion_length: int,
) -> tuple[str, bool, int]:
    """
    Hard-cap prompt tokens before GRPOTrainer sees them.

    Unsloth/TRL can otherwise pass prompt+completion tensors beyond the loaded
    2048 context, which crashes attention-mask construction. Left truncation
    preserves the generation contract at the end of the prompt.
    """
    safe_prompt_limit = max(256, min(max_prompt_length, MAX_SEQ_LEN - max_completion_length - 32))
    old_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=safe_prompt_limit,
        )
    finally:
        tokenizer.truncation_side = old_side
    input_ids = enc["input_ids"]
    clipped = len(input_ids) >= safe_prompt_limit
    return tokenizer.decode(input_ids, skip_special_tokens=False), clipped, len(input_ids)


def load_dataset(
    agent: str,
    tokenizer: Any,
    data_path: Optional[Path] = None,
    max_prompt_length: int = 1792,
    max_completion_length: int = 192,
):
    from datasets import Dataset

    path = data_path or DATA_FILES[agent]
    if not path.exists():
        sys.exit(f"[ERROR] Data file not found: {path}")

    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = _prepare_prompt(row["prompt"], tokenizer) + "\n\n" + GENERATION_CONTRACT
            prompt, clipped, prompt_tokens = _truncate_prompt_for_context(
                prompt,
                tokenizer,
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
            )
            rows.append({
                "prompt":      prompt,
                "v_label":     float(row["v_label"]),
                "claim_id":    row.get("claim_id",    ""),
                "query_id":    row.get("query_id",    ""),
                "is_critical": bool(row.get("is_critical", False)),
                "agent_role":  row.get("agent_role", row.get("role", agent)),
                "prompt_was_clipped": clipped,
                "prompt_tokens": prompt_tokens,
            })

    # Add to the global v_label lookup used inside brier_reward. Validation
    # prompts also need reward labels if eval is enabled.
    global _prompt_to_vlabel, _prompt_to_meta
    _prompt_to_vlabel.update({r["prompt"]: r["v_label"] for r in rows})
    _prompt_to_meta.update({
        r["prompt"]: {
            "claim_id": r["claim_id"],
            "query_id": r["query_id"],
            "is_critical": r["is_critical"],
            "agent_role": r["agent_role"],
        }
        for r in rows
    })

    counts = {0.0: 0, 0.5: 0, 1.0: 0}
    for r in rows:
        counts[r["v_label"]] = counts.get(r["v_label"], 0) + 1

    print(f"[Dataset] {agent}: {len(rows)} rows from {path}")
    print(
        f"  NOT_SUPPORTED={counts[0.0]}  "
        f"PARTIAL={counts[0.5]}  "
        f"SUPPORTED={counts[1.0]}"
    )
    clipped_count = sum(1 for r in rows if r["prompt_was_clipped"])
    max_prompt_tokens = max((r["prompt_tokens"] for r in rows), default=0)
    print(
        f"  prompt_clipped={clipped_count}/{len(rows)}  "
        f"max_prompt_tokens_after_cap={max_prompt_tokens}"
    )

    return Dataset.from_list(rows)


# ─── Training ──────────────────────────────────────────────────────────────────

def train(
    agent: str,
    data_path: Optional[Path] = None,
    val_data_path: Optional[Path] = None,
    resume_from: Optional[str] = None,
    init_adapter: Optional[Path] = None,
    max_steps: int = -1,
    save_steps: int = 50,
    logging_steps: int = 5,
    max_completion_length: int = 192,
    num_generations: int = 4,
    max_prompt_length: int = 1792,
    use_vllm: bool = False,
) -> None:
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    global _reward_log_every, _reward_tokenizer
    _reward_log_every = max(1, logging_steps)

    output_dir = OUTPUT_DIRS[agent]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GRPO  agent={agent}  model={MODEL_NAME}")
    print(f"output={output_dir}")
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    print(f"{'='*60}\n")

    # ── Model ──────────────────────────────────────────────────────────────────
    print("Loading model (QLoRA 4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,         # auto → bf16 on RTX 5090
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()
    if init_adapter is not None:
        _load_init_adapter(model, init_adapter)
    _reward_tokenizer = tokenizer
    if use_vllm:
        print("[WARN] vLLM is disabled for this GRPO training script; using use_vllm=False.")

    # ── Dataset ────────────────────────────────────────────────────────────────
    global _prompt_to_vlabel, _prompt_to_meta
    _prompt_to_vlabel = {}
    _prompt_to_meta = {}
    dataset = load_dataset(
        agent,
        tokenizer,
        data_path=data_path,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )
    eval_dataset = (
        load_dataset(
            f"{agent}_val",
            tokenizer,
            data_path=val_data_path,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
        )
        if val_data_path is not None else None
    )

    # ── Langfuse run span ──────────────────────────────────────────────────────
    _lf_run_span(
        agent,
        len(dataset),
        num_generations=num_generations,
        max_completion_length=max_completion_length,
    )

    # ── GRPO config ────────────────────────────────────────────────────────────
    # This truncates prompts before generation.
    # The reward_fn still receives the ORIGINAL string (not truncated), so the
    # _prompt_to_vlabel lookup is unaffected.
    grpo_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        temperature=0.7,                # must be > 0 — GRPO needs diverse completions
        bf16=True,
        fp16=False,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
        seed=42,
    )
    config_params = inspect.signature(GRPOConfig.__init__).parameters
    if "use_vllm" in config_params:
        grpo_kwargs["use_vllm"] = False
    if "max_completion_length" in config_params:
        grpo_kwargs["max_completion_length"] = max_completion_length
    else:
        grpo_kwargs["max_new_tokens"] = max_completion_length
    if "do_sample" in config_params:
        grpo_kwargs["do_sample"] = True
    if eval_dataset is not None:
        if "eval_strategy" in config_params:
            grpo_kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in config_params:
            grpo_kwargs["evaluation_strategy"] = "steps"
        if "eval_steps" in config_params:
            grpo_kwargs["eval_steps"] = save_steps
    grpo_config = GRPOConfig(**grpo_kwargs)

    # ── Trainer ────────────────────────────────────────────────────────────────
    # TRL >= 0.15 uses processing_class + args; older uses tokenizer + config.
    # Try newer API first; fall back silently.
    trainer_kwargs = {}
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    try:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=grpo_config,
            train_dataset=dataset,
            reward_funcs=brier_reward,
            **trainer_kwargs,
        )
    except TypeError:
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            config=grpo_config,
            train_dataset=dataset,
            reward_funcs=[brier_reward],
            **trainer_kwargs,
        )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\nStarting GRPO training...")
    print("NOTE: High invalid JSON rate is expected for the first ~100 steps.")
    print("      Monitor invalid_json rate in the periodic stats — it should fall over time.\n")

    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save final adapter ─────────────────────────────────────────────────────
    final_path = output_dir / "final"
    final_path.mkdir(exist_ok=True)
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nFinal GRPO adapter → {final_path}")

    if _langfuse:
        _langfuse.flush()


# ─── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for MAD debate agents")
    parser.add_argument(
        "--agent", required=True, choices=["a", "b", "combined"],
        help="Which agent/data mix to train: a, b, or combined",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        metavar="JSONL",
        help="Optional GRPO train JSONL path. Supports legacy string prompts and V4MAD chat-list prompts.",
    )
    parser.add_argument(
        "--val-data-path",
        default=None,
        metavar="JSONL",
        help="Optional validation JSONL path. Keep test JSONL untouched for final eval.",
    )
    parser.add_argument(
        "--resume-from-checkpoint", default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume from a saved checkpoint directory (e.g. outputs/grpo_agent_a/checkpoint-50)",
    )
    parser.add_argument(
        "--init-adapter",
        default=None,
        metavar="ADAPTER_DIR",
        help="Optional compatible 14B LoRA adapter to initialize the fresh GRPO adapter.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum optimizer steps to run. Use 1 for a smoke test; -1 runs the full epoch.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Checkpoint save interval in optimizer steps.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
        help="Trainer logging interval in optimizer steps.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=192,
        help="Maximum generated completion tokens per rollout.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of completions sampled per prompt for each GRPO group.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=1792,
        help="Maximum prompt tokens retained before generation.",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Deprecated. Ignored for GRPO training; vLLM should be used only for inference/eval.",
    )
    args = parser.parse_args()

    _init_langfuse()

    agent = "combined" if args.agent == "combined" else f"agent_{args.agent}"
    data_path = Path(args.data_path).expanduser().resolve() if args.data_path else None
    val_data_path = Path(args.val_data_path).expanduser().resolve() if args.val_data_path else None
    init_adapter = Path(args.init_adapter).expanduser().resolve() if args.init_adapter else None
    train(
        agent,
        data_path=data_path,
        val_data_path=val_data_path,
        resume_from=args.resume_from_checkpoint,
        init_adapter=init_adapter,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        use_vllm=args.use_vllm,
    )


if __name__ == "__main__":
    main()
