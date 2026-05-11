# MAD 3 Agents — Multi-Agent Debate Pipeline with GRPO Fine-Tuning

A complete end-to-end pipeline for factual claim verification using Multi-Agent Debate (MAD), a Judge LLM, and GRPO fine-tuning of Qwen2.5-7B-Instruct.

---

## Pipeline Overview

```
Stage 1: MAD Agents          Stage 2: Judge              Stage 3: GRPO Training
─────────────────────       ─────────────────────       ────────────────────────
414 claims from RAG DB  →   Judge reviews debates   →   Fine-tune Qwen2.5-7B
3 agents × 2 rounds         adjudicates verdicts        on GRPO/SFT/DPO datasets
vLLM on A100 40GB           anti-bias design            Unsloth + TRL on A100
```

---

## File Guide

| File | Stage | Description |
|------|-------|-------------|
| `3Agents_v1.ipynb` | Stage 1 | First version — GPTQ model, basic 3-agent debate |
| `3Agents.ipynb` | Stage 1 | v1 with full Colab outputs (414 claims, 139.3 min run) |
| `MAD_v2_Colab_v2.ipynb` | Stage 1 | v2 — AWQ-Marlin, anti-sycophancy, PARTIAL rate 69%→5.7% |
| `MAD_v2_Colab_v2b.ipynb` | Stage 1 | v2 alternate run |
| `MAD_Judge_v1_Colab.ipynb` | Stage 2 | Judge LLM — 414 verdicts, temp=0, anti-bias |
| `MAD_Stage3_GRPO_Colab.ipynb` | Stage 3 | GRPO fine-tuning — both dtype bugs fixed |
| `mad_debate_stage2_judged.db` | All | SQLite DB — 2484 agent outputs + 414 judge verdicts |
| `mad_v2_grpo_v1.jsonl` | Stage 3 | GRPO training data v1 (~13 MB, 2484 records) |
| `mad_v2_grpo_v2.jsonl` | Stage 3 | GRPO training data v2 — fixed symmetric Brier (~13 MB) |
| `mad_v2_sft_v1.jsonl` | Stage 3 | SFT warm-up data v1 (~891 KB) |
| `mad_v2_sft_v2.jsonl` | Stage 3 | SFT warm-up data v2 — 1221 records (~5.7 MB) |
| `mad_v2_dpo_v1.jsonl` | Stage 3 | DPO preference pairs v1 — 60 pairs (~346 KB) |
| `mad_v2_dpo_v2.jsonl` | Stage 3 | DPO preference pairs v2 — 77 pairs (~463 KB) |

---

## Stage 1 — Multi-Agent Debate

**Model:** Qwen2.5-14B-Instruct-AWQ via vLLM on A100 40GB

**3 agents with distinct roles:**
| Agent | Role | Temp R0 | Temp R1 |
|-------|------|---------|---------|
| `agent_a` | Verifier — argues FOR the claim | 0.3 | 0.4 |
| `agent_b` | Adversarial Auditor — challenges the claim | 0.8 | 0.9 |
| `agent_c` | Calibrator — weighs both sides | 0.5 | 0.5 |

**2 rounds per claim:**
- **Round 0:** Each agent independently assesses the claim + evidence
- **Round 1:** Each agent sees the other two agents' reasoning (anonymised) and updates their verdict

**v2 improvements over v1:**
- AWQ-Marlin kernel (faster than AWQ)
- Anti-sycophancy prompt: warns agents when peers agree — forces independent thinking
- Confidence clamped to [0.05, 0.95] — preserves Brier reward signal
- Retry logic: up to 3 attempts with JSON reminder on parse failure
- PARTIAL rate dropped from **69% → 5.7%** (agents now commit to clear verdicts)

**Scale:** 414 claims × 3 agents × 2 rounds = **2,484 LLM calls** in **139.3 minutes**

---

## Stage 2 — Judge LLM

**Model:** Same Qwen2.5-14B-Instruct-AWQ, temperature=0.0 (deterministic)

**Anti-bias design:**
- Agent identities hidden from judge (sees "Debater 1/2/3", not "agent_a/b/c")
- `confidence_internal` never shown to judge
- Agent presentation order randomised per claim (seeded by `claim_id` for reproducibility)
- System prompt explicitly states: *"consensus is NOT evidence"*
- Judge forms own verdict from evidence BEFORE reading the debate transcript

**Outputs:** `v_label` ∈ {0.0 = NOT_SUPPORTED, 0.5 = PARTIAL, 1.0 = SUPPORTED}

**Results:**
- NOT_SUPPORTED: 256 claims (61%) | avg confidence: 0.888
- PARTIAL: 95 claims (22%) | avg confidence: 0.789
- SUPPORTED: 63 claims (15%) | avg confidence: 0.902
- Judge–agent majority agreement: **79%** (moderate independence)

---

## Stage 3 — GRPO Fine-Tuning

**Model:** Qwen2.5-7B-Instruct, full bfloat16, no quantisation

### Reward Function — Symmetric Brier Score

```
R = 1 - 2 × (p_for_brier − v_label)²

p_for_brier(confidence, verdict):
  SUPPORTED     →  p = confidence        # high conf + SUPPORTED = high P(true)
  NOT_SUPPORTED →  p = 1 - confidence    # high conf + NOT_SUPPORTED = low P(true)
  PARTIAL       →  p = 0.5              # always midpoint
  IDK           →  p = 0.5

Range: [−1, +1]
  Perfect correct answer  → R = +1.0
  Uncertain (conf = 0.5)  → R = +0.5
  Parse failure           → R = −1.0 (penalty)
  Perfect wrong answer    → R = −1.0

Examples:
  NOT_SUPPORTED conf=0.85, v_label=0.0  →  p=0.15  →  R = +0.955  ✓
  SUPPORTED     conf=0.90, v_label=1.0  →  p=0.90  →  R = +0.980  ✓
  SUPPORTED     conf=0.90, v_label=0.0  →  p=0.90  →  R = −0.620  ✗ (wrong + confident)
```

### Training Config
| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen2.5-7B-Instruct` (bfloat16) |
| LoRA r / alpha | 16 / 16 |
| Max seq length | 1536 tokens |
| GRPO completions per prompt (G) | 4 |
| Effective batch size | 4 (batch=1 × grad_accum=4) |
| Learning rate | 5e-6 (cosine decay) |
| Training epochs | 2 |
| Reward weights | Brier=1.0, Format bonus=0.1 |

### Dataset Stats
| Split | Records | Avg Brier | Notes |
|-------|---------|-----------|-------|
| GRPO | 2,484 | 0.865 | All claims × 3 agents × 2 rounds |
| SFT | 1,221 | 0.932 | High-quality R1 outputs only (reward > 0.5) |
| DPO | 77 pairs | margin 0.48 | Best vs worst agent per claim |

---

## Critical Bug Fixes in Stage 3

Two separate `RuntimeError: self and mat2 must have the same dtype, got Half and Float` crashes were diagnosed and fixed:

### Bug 1 — bitsandbytes NF4 dequantisation
| | Details |
|--|---------|
| **Cause** | `load_in_4bit=True` — bitsandbytes NF4 dequantises weights to **float32** during GRPO inference (no autocast context active). Inside `matmul_lora`: `out=bfloat16` vs `B.to(float32)=Float` → crash |
| **Dead giveaway** | Model loaded from `*-bnb-4bit` HuggingFace repo |
| **Fix** | `load_in_4bit=False` — load full bfloat16 model |

### Bug 2 — Unsloth custom gradient checkpointing
| | Details |
|--|---------|
| **Cause** | `use_gradient_checkpointing='unsloth'` — Unsloth pre-allocates activation buffers in **float16** to save VRAM. During backward recomputation, input X arrives as float32. `matmul_lora`: `out=Half` (pre-allocated) vs `B.to(X.dtype)=Float` → `addmm_` crash |
| **Dead giveaway** | Log line: `"Unsloth: Will smartly offload gradients to save VRAM!"` |
| **Fix** | `use_gradient_checkpointing=False` |

**Why GC is safe to skip on A100 40GB:**
```
Model weights (7B bfloat16):   ~14.0 GB
LoRA adapters (r=16):           ~0.3 GB
Optimizer states (AdamW 8bit):  ~1.5 GB
GRPO activations (G=4):         ~6.0 GB
──────────────────────────────────────
Total used:                    ~22 GB
VRAM remaining:                ~18 GB  ✓  No GC needed
```

---

## How to Run

### Stage 1 — Run the debate
1. Open `MAD_v2_Colab_v2.ipynb` in Google Colab (A100 GPU)
2. Upload `mad_debate_stage2_judged.db` when prompted
3. Run all cells top to bottom (~2.5 hours)

### Stage 2 — Run the judge
1. Open `MAD_Judge_v1_Colab.ipynb`
2. Upload the DB from Stage 1
3. Run all cells (~30 minutes)
4. Download updated DB + GRPO/SFT/DPO exports

### Stage 3 — Fine-tune with GRPO
1. Open `MAD_Stage3_GRPO_Colab.ipynb` in Colab (**A100 40GB required**)
2. Upload `mad_v2_grpo_v2.jsonl` and `mad_v2_sft_v2.jsonl`
3. Skip Cell 7 (SFT warm-up) — go directly to GRPO
4. **Do NOT change:** `load_in_4bit=False` or `use_gradient_checkpointing=False`

### Dependencies
```bash
# Stage 1 & 2
pip install vllm aiohttp nest_asyncio pydantic

# Stage 3
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes datasets
```

---

## Interview Story

```
Stage 1: Qwen2.5-14B-AWQ agents debate in 3-role MAD format.
         Stochastic (temp 0.3–0.9) to ensure diverse debate signal.
         Anti-sycophancy prompts prevent consensus collapse.

Stage 2: Same model as deterministic judge (temp=0.0).
         Anti-bias: agent identity hidden, order randomised, anti-majority instruction.
         Outputs v_label ∈ {0.0, 0.5, 1.0} as ground truth for Brier reward.

Stage 3: Qwen2.5-7B-Instruct fine-tuned with Unsloth GRPO.
         Reward = symmetric Brier score (corrected p_for_brier for NOT_SUPPORTED).
         GRPO generates 4 completions per prompt, updates toward higher Brier reward.
         Result: model learns calibrated, evidence-grounded verdict reasoning.
```
