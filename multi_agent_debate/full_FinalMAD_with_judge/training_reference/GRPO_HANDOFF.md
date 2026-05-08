# MAD v3 — GRPO Fine-Tuning Handoff

This document is everything a new session needs to run GRPO fine-tuning on the two MAD debate agents using Unsloth on an RTX 5090 (32GB VRAM).

Read this fully before writing any code.

---

## What This Project Is

Multi-Agent Debate (MAD) v3 — a local LLM guardrail system for detecting hallucinations in regulatory/healthcare QA.

Pipeline:
```
User query → RAG retrieval → Baseline LLM answer → Decomposer (extracts atomic claims)
→ Agent A (strict auditor) + Agent B (adversarial auditor) debate 2 rounds per claim
→ Blind 32B Judge assigns verdict: NOT_SUPPORTED (0.0) / PARTIAL (0.5) / SUPPORTED (1.0)
```

Two debate agents ran on RTX 5090 using `Qwen/Qwen2.5-14B-Instruct-AWQ`.
Judge ran on SJSU H100 using `Qwen/Qwen2.5-32B-Instruct`.
416 claims across 50 healthcare/regulatory queries were fully judged.

---

## What's in This Package

```
grpo_finetune_package/
├── GRPO_HANDOFF.md                  ← this file
│
├── data/
│   ├── agent_a_grpo.jsonl           ← 832 rows: Agent A SFT + GRPO dataset
│   ├── agent_b_grpo.jsonl           ← 832 rows: Agent B SFT + GRPO dataset
│   └── ground_truth_50.json         ← 50 queries with reference labels
│
├── prompts.py                       ← EXACT system + user prompts used in live debate
├── schemas.py                       ← Pydantic schemas (AgentOutputFull, strip_for_peer, etc.)
├── train_sft.py                     ← SFT training script (transformers.Trainer + peft LoRA)
│
├── sft_adapters/
│   ├── sft_agent_a/                 ← Trained LoRA adapter for Agent A (Qwen2.5-7B-Instruct)
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors  (155MB — LoRA r=16)
│   │   └── tokenizer files
│   └── sft_agent_b/                 ← Trained LoRA adapter for Agent B (Qwen2.5-7B-Instruct)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files
│
└── judge_reference/
    └── judge.py                     ← Judge node with Langfuse logging (reference only)
```

---

## JSONL Dataset Format (agent_a_grpo.jsonl / agent_b_grpo.jsonl)

Each row is one (agent, round, claim) training example:

```json
{
  "prompt": "<AGENT_A_SYSTEM>\nUSER QUERY: ...\n\nCLAIM TO VERIFY: ...",
  "completion": "{\"verdict\": \"PARTIAL\", \"reasoning\": \"...\", \"evidence_cited\": [...], \"confidence_internal\": 0.9}",
  "valid_completion": true,
  "v_label": 0.5,
  "role": "agent_a",
  "round": 0,
  "claim_id": "...",
  "query_id": "q_001",
  "is_critical": false,
  "is_material": true,
  "judge_confidence": 0.85
}
```

Key fields for GRPO:
- `prompt` — full prompt fed to model (system + user, already concatenated)
- `v_label` — judge verdict (0.0 / 0.5 / 1.0). This is the reward anchor — fixed.
- `valid_completion` — `false` for 23 truncated rows. Filter these for SFT warm-up; keep all for GRPO (model generates fresh completions anyway).
- `completion` — stored model response from live debate run (use for SFT only)

Stats:
- 832 rows per agent (416 claims × 2 rounds)
- v_label distribution: NOT_SUPPORTED=79 (19%), PARTIAL=327 (79%), SUPPORTED=10 (2%)
- Agent A: 809 valid_completion=true / 23 false
- Agent B: 817 valid_completion=true / 15 false

---

## Agent Roles and System Prompts

Both agents use the same base model and same output schema. Only the system prompt differs.

**Agent A — Strict Regulatory Auditor:**
```
You are a strict regulatory compliance auditor in a debate.
YOUR ROLE: Verify whether the LLM's claim is supported by the retrieved evidence.
Be precise. Find errors, hallucinations, unsupported claims, or misleading statements.
[... see prompts.py AGENT_A_SYSTEM for full text]
```

**Agent B — Adversarial Auditor:**
```
You are an adversarial auditor in a debate.
YOUR ROLE: Challenge the LLM's claim. Look for what's WRONG, MISSING, or UNSUPPORTED.
[... see prompts.py AGENT_B_SYSTEM for full text]
```

Output schema (both agents, both rounds):
```json
{
  "verdict": "SUPPORTED | PARTIAL | NOT_SUPPORTED | IDK",
  "reasoning": "at most 3 short sentences",
  "evidence_cited": [{"chunk_id": "...", "relevant_quote": "..."}],
  "confidence_internal": 0.0
}
```

Round 1 adds the opposing agent's Round 0 verdict/reasoning as extra context (peer view).
`confidence_internal` is stripped before being shown to the peer or judge.

---

## SFT Adapters — What They Are

**Base model trained:** `Qwen/Qwen2.5-7B-Instruct` (NOTE: agents originally used 14B AWQ)

The SFT adapters were trained on the stored debate completions to teach the model the output format. They were trained on H100 using standard PEFT LoRA (r=16, BF16, no quantization).

Results:
- Train loss: 0.1158 (Agent A), 0.1158 (Agent B)
- Eval loss: 0.1193 (Agent A), 0.1193 (Agent B)
- Validation: 5/5 samples produced valid parseable JSON with correct keys

**Important:** For production GRPO, you should use `Qwen/Qwen2.5-14B-Instruct` (the same family as the original debate agents). Download it from HuggingFace first:
```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-14B-Instruct", local_dir="./models/Qwen2.5-14B-Instruct")
```
The provided SFT adapters work with 7B. For 14B, re-run SFT first (train_sft.py, change MODEL_PATH).

---

## GRPO Training — What To Build

### Reward Function

Brier reward: `R = 2 × p × v − p²`

Where:
- `p` = `confidence_internal` parsed from model's generated JSON output (0.0–1.0)
- `v` = `v_label` from dataset row (0.0 / 0.5 / 1.0) — judge's verdict, fixed

Reward properties:
- p=0.9, v=0.0 (overconfident on hallucination): R = 2×0.9×0.0 − 0.81 = **−0.81** (penalized)
- p=0.9, v=1.0 (confident and correct): R = 2×0.9×1.0 − 0.81 = **+0.99** (rewarded)
- p=0.5, v=0.0 (uncertain on hallucination): R = 2×0.5×0.0 − 0.25 = **−0.25** (mild penalty)

The current agents are severely overconfident: avg confidence ~0.9 on claims the judge rated PARTIAL/NOT_SUPPORTED. GRPO should push them to emit lower confidence on non-supported claims.

Additional penalties:
- Invalid JSON: R = −1.0
- Missing `confidence_internal`: R = −1.0
- confidence_internal outside [0, 1]: R = −0.5

Optional: multiply reward by 1.5 for `is_critical=True` claims.

### Unsloth GRPO Setup

Install:
```bash
pip install unsloth
pip install "trl>=0.15.0" peft accelerate datasets
```

Load model for GRPO:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./models/Qwen2.5-14B-Instruct",  # or 7B path
    max_seq_length=2048,
    dtype=None,       # auto (bf16 on modern GPU)
    load_in_4bit=True,  # QLoRA — needed on 32GB for 14B
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

GRPO dataset format (what Unsloth's GRPOTrainer expects):
```python
{
    "prompt": "<full prompt string>",   # system + user, no completion
    # metadata passed to reward_fn via a separate mechanism
}
```

Reward function signature for trl GRPOTrainer:
```python
def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, prompt in zip(completions, prompts):
        # 1. Find v_label for this prompt (match to dataset row)
        # 2. Parse completion JSON
        # 3. Compute Brier reward
        ...
    return rewards
```

**Note on v_label lookup:** The reward function receives the generated `completion` but needs the `v_label` for the corresponding prompt. Pass v_label as part of the prompt metadata, or build a lookup dict keyed by prompt string at dataset load time.

### GRPO Training Parameters for RTX 5090 (32GB)

```python
from trl import GRPOConfig

config = GRPOConfig(
    output_dir="outputs/grpo_agent_a",
    num_train_epochs=1,                 # GRPO is data-hungry; start with 1
    per_device_train_batch_size=1,      # generate K completions per prompt
    gradient_accumulation_steps=8,
    learning_rate=5e-6,                 # lower LR than SFT
    num_generations=4,                  # K=4 completions per prompt (GRPO group)
    max_new_tokens=256,
    temperature=0.7,                    # sampling during rollout
    bf16=True,
    logging_steps=5,
    save_steps=50,
    report_to="none",
    use_vllm=False,                     # set True if you have vllm installed for faster rollouts
)
```

### Dataset Loading for GRPO

Load the JSONL, filter for GRPO (all rows, not just valid_completion):
```python
import json

def load_grpo_dataset(path, agent):
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            # For GRPO: use prompt only (model generates fresh completions)
            rows.append({
                "prompt": row["prompt"],
                "v_label": row["v_label"],
                "claim_id": row["claim_id"],
                "is_critical": row["is_critical"],
            })
    return rows

agent_a_data = load_grpo_dataset("data/agent_a_grpo.jsonl", "agent_a")
agent_b_data = load_grpo_dataset("data/agent_b_grpo.jsonl", "agent_b")
```

---

## Training Sequence

```
Step 1: SFT warm-up on 7B (already done — adapters included)
        OR re-run SFT on 14B: python train_sft.py --agent a && python train_sft.py --agent b

Step 2: Load SFT-adapted model as starting point for GRPO
        (initialize GRPO from SFT checkpoint, not raw base model)

Step 3: GRPO Agent A — train_grpo.py --agent a
        Duration: ~2-4 hours on RTX 5090 for 1 epoch over 832 prompts with K=4

Step 4: GRPO Agent B — train_grpo.py --agent b
        (same duration)

Step 5: Validate — run inference on 10 held-out prompts, check confidence calibration

Step 6: Deploy in debate pipeline via vLLM LoRA adapter
```

---

## How to Deploy Fine-Tuned Agents

After GRPO, you have LoRA adapters. Two deployment paths:

**Option A — vLLM + LoRA (fastest, no re-quantization):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --enable-lora \
  --lora-modules agent_a=./outputs/grpo_agent_a \
                 agent_b=./outputs/grpo_agent_b \
  --max-lora-rank 16 \
  --gpu-memory-utilization 0.90
```
In the debate pipeline, request `model="agent_a"` for Agent A calls, `model="agent_b"` for Agent B.

**Option B — Merge + AWQ re-quantization (same inference speed as original):**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "outputs/grpo_agent_a")
merged = model.merge_and_unload()
merged.save_pretrained("outputs/merged_agent_a_bf16")

# Then quantize to AWQ using autoawq:
# pip install autoawq
# python -c "from awq import AutoAWQForCausalLM; ..."
```

---

## After Fine-Tuning — Eval Plan

1. Replace debate agents with fine-tuned models
2. Change baseline LLM to a different 3B model (e.g., `Qwen2.5-3B-Instruct` or `Llama3.2-3B-Instruct`)
3. Run full debate pipeline on same 50 queries
4. Run judge again on new outputs
5. Compare:
   - Before: avg confidence ~0.9 on PARTIAL/NOT_SUPPORTED claims
   - After: expect confidence to drop toward 0.3–0.5 on unsupported claims
   - Judge verdict distribution should shift (more granular, fewer confident wrong answers)

---

## Key Numbers to Know

| Metric | Value |
|---|---|
| Queries | 50 healthcare/regulatory |
| Claims debated | 416 |
| Agent outputs | 1664 (2 agents × 2 rounds × 416) |
| Judge verdicts | 416: NOT_SUPPORTED=79, PARTIAL=327, SUPPORTED=10 |
| Avg agent confidence | ~0.9 (severely overconfident) |
| SFT train loss | 0.1158 (both agents) |
| SFT eval loss | 0.1193 (both agents) |
| SFT base model | Qwen2.5-7B-Instruct |
| Original debate model | Qwen2.5-14B-Instruct-AWQ |
| Judge model | Qwen2.5-32B-Instruct |

---

## What the Judge Does (Reference)

The judge is a blind 32B model (`Qwen/Qwen2.5-32B-Instruct`) running on SJSU H100 (not on the 5090). It is **not fine-tuned** and not part of GRPO — it is the source of ground truth `v_label` values that are already baked into the JSONL files.

**Judge input:** Both agents' R0 and R1 outputs (stripped via `strip_for_peer` — no `confidence_internal`), the original claim, and RAG chunks.

**Judge output schema:**
```json
{
  "v_label": 0.0,
  "judge_confidence": 0.85,
  "judge_reasoning": "...",
  "evidence_chunk_ids": ["chunk_id_1"]
}
```

`v_label` maps to: `NOT_SUPPORTED=0.0`, `PARTIAL=0.5`, `SUPPORTED=1.0`. This value is fixed in the JSONL as the reward anchor.

**How the judge ran:** Async LangChain calls, results written to SQLite via `insert_judge_verdict`. Langfuse traced every call (optional, keys needed in `.env`). On parse failure, judge defaulted to PARTIAL with `judge_confidence=0.0` — 0 such failures in the final run.

**Judge verdict distribution (416 claims):**
- NOT_SUPPORTED (0.0): 79 claims (19%)
- PARTIAL (0.5): 327 claims (79%)
- SUPPORTED (1.0): 10 claims (2%)

The heavy skew toward PARTIAL means GRPO must learn fine-grained calibration, not just binary correct/wrong.

---

## SFT Phase — What We Did and What Failed

### Goal
SFT warm-up teaches the model the **output format** (valid JSON with the 4 required keys) before GRPO's online reward optimization. Without SFT, the base model frequently produces prose instead of JSON, making the reward function always return -1.0 and giving GRPO no signal to learn from.

### What Worked
- **`transformers.Trainer` + custom `CompletionOnlyDataset`** — completed cleanly on H100, both agents in ~9 minutes each
- **Loss masking on prompt tokens** — `labels = -100` for system+user tokens, real token ids for completion only. This is critical: without it, the model trains to memorize system prompts, not generate correct completions
- **`model.enable_input_require_grads()`** — required when using gradient checkpointing with LoRA. Without this call, PyTorch raises a RuntimeError about leaf tensors not requiring grad
- **SFT results:** train loss 0.1158, eval loss 0.1193 (both agents). 5/5 validation samples produced valid parseable JSON with correct keys

### What Failed — Do NOT Repeat These

**1. Using `trl.SFTTrainer` on HPC**
`trl.trainer.sft_trainer` has a hard import at line 31:
```python
from datasets import Dataset, IterableDataset
```
This is not lazy — importing `SFTTrainer` at all requires the `datasets` package. On SJSU HPC (Python 3.11 environment with restricted pip access), `datasets>=4.x` requires `pyarrow>=21.0.0` which had no pre-built wheel and failed to compile. **Fix: use `transformers.Trainer` directly with a custom PyTorch Dataset.**

**2. `numpy 2.x` on HPC login nodes (GCC 7.3)**
`numpy 2.4.4` requires GCC >= 9.3 to build from source. hpc1 login node has GCC 7.3. Any package that transitively pulls numpy 2.x will fail.
- Fix: pin `numpy==1.26.4` which has a pre-built cp311 wheel
- Doesn't help if the real blocker is pyarrow (see above)

**3. `bitsandbytes` pulling `scipy`**
`bitsandbytes 0.42.0` requires scipy, which also failed to build under GCC 7.3.
- Fix: drop bitsandbytes entirely. H100 has 81GB VRAM — 7B model in BF16 with LoRA fits easily, no 4-bit QLoRA needed for SFT on H100.

**4. Unsloth on HPC**
Unsloth requires a modern pip environment with internet access (it installs triton/xformers kernels). On SJSU HPC, offline `--only-binary --platform` download didn't find a matching wheel.
- Fix: use Unsloth **only on the RTX 5090 at college** where the environment is fresh and internet is available. SFT ran fine with plain `transformers.Trainer`.

**5. `SFTConfig` instead of `TrainingArguments`**
`trl>=0.15` renamed the config class. If you use an older trl, `SFTConfig` doesn't exist; if you use a newer trl, `SFTConfig` has different field names than `TrainingArguments`. Mixing them causes silent errors.
- Fix: use `transformers.TrainingArguments` directly — it's stable across versions.

**6. Python 3.6 `huggingface_hub` on HPC**
The system `python3` on hpc1 is 3.6. Running `from huggingface_hub import snapshot_download` fails with `ModuleNotFoundError: No module named 'dataclasses'` because `dataclasses` is a Python 3.7+ stdlib module.
- Fix: always use the explicit `python3.11` binary or activate a proper venv. Check with `python --version` before running anything.

---

## GRPO — What NOT To Do

These are lessons from building and debugging the SFT phase. They apply directly to GRPO.

### 1. Do NOT use `SFTTrainer` for GRPO either
Same import chain issue. Use `GRPOTrainer` from `trl` directly — it does not have the same hard `datasets` dependency at import time. But verify on the 5090 before assuming.

### 2. Do NOT use `use_vllm=True` unless vLLM is confirmed installed
`GRPOConfig(use_vllm=True)` will crash at trainer initialization if vLLM is not installed. Set `use_vllm=False` first, confirm training works end-to-end, then switch to vLLM for faster rollouts.

### 3. Do NOT load the model twice
Unsloth's `FastLanguageModel` loads the model with its own memory layout. If you load it again with `AutoModelForCausalLM` for the SFT adapter, you'll OOM. The correct flow:
```python
model, tokenizer = FastLanguageModel.from_pretrained(...)  # loads base
model = FastLanguageModel.get_peft_model(model, ...)        # adds LoRA
# Then load SFT weights into the LoRA adapter:
from peft import set_peft_model_state_dict
import safetensors.torch
sft_weights = safetensors.torch.load_file("sft_adapters/sft_agent_a/adapter_model.safetensors")
set_peft_model_state_dict(model, sft_weights)
```

### 4. Do NOT pass the full JSONL row to the reward function directly
`GRPOTrainer` calls your reward function with `completions` and `prompts`. It does **not** pass arbitrary metadata. Build a `prompt → v_label` lookup dict at dataset load time:
```python
prompt_to_vlabel = {row["prompt"]: row["v_label"] for row in rows}
```
Then look up inside the reward function. Keying by prompt string is safe because prompts are unique (each is a specific claim × round × agent combination).

### 5. Do NOT use `temperature=0` during GRPO rollouts
GRPO requires sampling diversity across the K completions in each group. With temperature=0, all K completions are identical, the group reward has zero variance, and the GRPO gradient is zero — the model doesn't learn. Use `temperature=0.7` as specified in the config.

### 6. Do NOT skip the SFT warm-up for GRPO
If you skip SFT and run GRPO on the raw base model:
- ~30–40% of generated completions will be invalid JSON → reward = -1.0
- GRPO can't learn from a reward signal that's almost always -1.0 (no variance)
- The model diverges or produces garbage
The provided SFT adapters are already trained. Use them.

### 7. Do NOT train Agent A and Agent B simultaneously
Both use the same 14B base model. Loading two model instances at once will OOM on 32GB VRAM. Train A first, save adapter, then train B.

### 8. Do NOT use `num_generations` (K) > 6 on 32GB VRAM
Each generation requires a forward+backward pass. K=4 is the tested safe value for 14B + QLoRA on 32GB. K=8 will OOM. If you want more diversity, increase `num_train_epochs` instead.

---

## Files You Do NOT Need (not included)

- `mad_with_verdicts.db` — 12MB SQLite with all raw data (optional, for debugging)
- `export_grpo_dataset.py` — script that built the JSONL files (already done)
- HPC Slurm scripts — HPC-specific, not needed on local 5090
- RAG index files — only needed if re-running the full pipeline from scratch

---

## Quick Start on RTX 5090

```bash
# 1. Unzip and enter directory
unzip mad_grpo_package.zip && cd grpo_finetune_package

# 2. Install dependencies
pip install unsloth "trl>=0.15.0" peft accelerate datasets

# 3. Download 14B model (or use 7B path from sft_adapters)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-14B-Instruct', local_dir='./models/Qwen2.5-14B-Instruct')
"

# 4. (Optional) Re-run SFT on 14B — takes ~20 min on 5090
# Edit MODEL_PATH in train_sft.py to './models/Qwen2.5-14B-Instruct'
python train_sft.py --agent a
python train_sft.py --agent b

# 5. Write train_grpo.py using the spec in this document
#    Ask Claude: "Read GRPO_HANDOFF.md and write train_grpo.py for Unsloth GRPO"

# 6. Run GRPO
python train_grpo.py --agent a
python train_grpo.py --agent b
```
