# RLHF Implementation — Guardrails Enterprise
## Live Reinforcement Learning from Human Feedback for Healthcare & Regulatory AI

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why RLHF for Healthcare AI](#2-why-rlhf-for-healthcare-ai)
3. [System Architecture](#3-system-architecture)
4. [Multi-Agent Debate Pipeline](#4-multi-agent-debate-pipeline)
5. [Algorithm: GRPO with REINFORCE++](#5-algorithm-grpo-with-reinforce)
6. [Reward Function Design](#6-reward-function-design)
7. [Human Feedback Loop](#7-human-feedback-loop)
8. [Model & Training Configuration](#8-model--training-configuration)
9. [Dataset & Domain Prompts](#9-dataset--domain-prompts)
10. [SDK Architecture](#10-sdk-architecture)
11. [Infrastructure & Hardware](#11-infrastructure--hardware)
12. [Implementation Journey & Decisions](#12-implementation-journey--decisions)
13. [Results & Evaluation](#13-results--evaluation)
14. [File Structure](#14-file-structure)
15. [How to Run](#15-how-to-run)
16. [Next Steps](#16-next-steps)

---

## 1. Project Overview

**Guardrails Enterprise** is an enterprise SDK that wraps any baseline LLM with a healthcare and regulatory AI guardrail layer. The core goal is to ensure that LLM responses in healthcare and regulatory contexts are:

- Factually grounded with citations to specific regulatory sources (HIPAA, CFR, FDA, HL7 FHIR)
- Calibrated in confidence — the model should not be overconfident on uncertain claims
- Free of Protected Health Information (PHI) — HIPAA compliance at inference time
- Compliant with regulatory policy blocks — preventing harmful or legally problematic outputs

To achieve this, we use **Reinforcement Learning from Human Feedback (RLHF)** to fine-tune the base LLM. Specifically, we implement an **online RL training loop** using **GRPO (Group Relative Policy Optimization)** combined with a **human-in-the-loop feedback mechanism**, running on a Colab A100 40GB GPU.

The fine-tuned model is exported as a LoRA adapter that can be loaded on top of any compatible base model, distributed as part of the SDK package.

---

## 2. Why RLHF for Healthcare AI

### The problem with standard fine-tuning

Standard supervised fine-tuning (SFT) teaches a model to mimic a fixed dataset. In healthcare and regulatory domains this is insufficient because:

- Regulatory documents change — FDA guidance, CFR updates, CMS rules evolve constantly
- Calibration matters more than accuracy — an overconfident wrong answer is more dangerous than an uncertain correct one
- PHI leakage cannot be caught by loss functions — it requires explicit reward signals
- Multi-agent debate produces dynamic outputs that cannot be pre-labelled

### Why RLHF solves this

RLHF trains the model to maximize a reward signal rather than mimic labels. This means:

- The reward function encodes domain-specific rules (PHI penalty, citation requirement) that are impossible to express as labels
- The model learns calibration directly — it is rewarded for being right when confident and penalized for being wrong when confident
- Human feedback overrides automatic rewards for ambiguous cases — capturing regulatory nuance that heuristics miss
- The training loop is online — the model generates responses, gets rewards, and updates in real time

### Why GRPO specifically

GRPO (used in DeepSeek-R1 training) is preferred over PPO for our use case because:

- It does not require a separate critic model — saving ~14 GB of GPU memory on a single A100
- It uses group relative rewards — comparing multiple completions for the same prompt, which is natural for the MAD pipeline where Agent A generates multiple candidate claims
- It is more stable than vanilla REINFORCE for long-form text generation
- TRL 1.3.0 ships native GRPO support fully compatible with our torch 2.10 + transformers 5.0 environment

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enterprise Guardrails SDK                     │
│                                                                  │
│  ┌─────────────┐    ┌──────────────────────────────────────┐    │
│  │ Baseline LLM│───▶│         MAD Pipeline                 │    │
│  │ Qwen 2.5-7B │    │  Agent A │ Agent B │ Judge           │    │
│  └─────────────┘    └──────────────────────────────────────┘    │
│          ▲                        │                              │
│          │                        ▼                              │
│  ┌───────────────┐    ┌───────────────────────┐                 │
│  │  LoRA Adapter │    │   Reward Computation   │                 │
│  │  (RLHF output)│    │   Brier + PHI +        │                 │
│  └───────────────┘    │   Citation + Human     │                 │
│          ▲            └───────────────────────┘                 │
│          │                        │                              │
│  ┌───────────────────────────────────────────────┐              │
│  │              GRPO Training Loop               │              │
│  │   Generate → Reward → Advantage → Update      │              │
│  └───────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### Component responsibilities

| Component | Role |
|---|---|
| Baseline LLM | Qwen/Qwen2.5-7B-Instruct — the model being fine-tuned |
| LoRA Adapter | r=16 adapter trained via GRPO, merged for SDK distribution |
| MAD Pipeline | Multi-Agent Debate — Agent A claims, Agent B challenges, Judge verdicts |
| Reward Function | Brier score + PHI penalty + citation bonus + human override |
| GRPO Trainer | TRL GRPOTrainer, online RL loop, no critic model needed |
| Human Feedback UI | ipywidgets interface in Colab for ambiguous reward cases |
| SDK Wrapper | GuardrailsSDK class — policy check, PHI scrub, calibrated output |

---

## 4. Multi-Agent Debate Pipeline

### Overview

The MAD (Multi-Agent Debate) pipeline is the environment in which RLHF training happens. Rather than training on static question-answer pairs, Agent A is trained by debating claims in a structured three-agent system.

### Agents

**Agent A — The Claimant (the model being trained)**
- Generates a healthcare or regulatory claim in response to a prompt
- Must include a calibrated confidence score: `CONFIDENCE: <float 0.0-1.0>`
- Must cite at least one regulatory source (HIPAA section, CFR part, FDA guidance, etc.)
- Must not include any PHI or patient identifiers

**Agent B — The Challenger**
- Generates a counterargument to Agent A's claim
- Identifies weaknesses in citations, overconfident statements, or regulatory inaccuracies
- Not trained via RLHF — uses the base model

**Judge — The Verdict**
- Evaluates Agent A's claim against Agent B's challenge
- Returns a verdict probability `v ∈ [0,1]` — how much the Judge agrees with Agent A
- Verdict feeds directly into the Brier score reward

### Debate format

```
System: You are Agent A in a Multi-Agent Debate pipeline for healthcare 
        regulatory AI. Provide evidence-based claims with calibrated 
        confidence. Never include PHI. Always cite regulatory sources.
        End with: CONFIDENCE: <float 0.0-1.0>

User:   [Healthcare or regulatory query]

Agent A: [Claim with citation and confidence score]
         CONFIDENCE: 0.82

Agent B: [Counterargument challenging Agent A's claim]

Judge:  [Verdict — agrees/disagrees with Agent A, confidence v]
```

### Why MAD for RLHF

Standard RLHF trains a model to satisfy a reward model. MAD adds adversarial pressure — Agent A must produce claims that survive challenge from Agent B. This produces more robust, better-cited, more calibrated responses than single-agent training because the model learns to anticipate counterarguments.

---

## 5. Algorithm: GRPO with REINFORCE++

### GRPO — Group Relative Policy Optimization

GRPO generates `G` completions for each prompt, computes rewards for all of them, then uses the group mean as a baseline to compute advantages. This eliminates the need for a separate critic/value network.

**Advantage computation:**

```
For each prompt i, generate G completions {c_1, c_2, ..., c_G}
Compute rewards {r_1, r_2, ..., r_G}
Baseline = mean(r_1, ..., r_G)
Advantage_j = (r_j - baseline) / std(r_1, ..., r_G)
```

**Policy gradient update:**

```
L_GRPO = E[ min(
    ratio * advantage,
    clip(ratio, 1-ε, 1+ε) * advantage
)] - β * KL(π_θ || π_ref)
```

Where:
- `ratio = π_θ(c|p) / π_old(c|p)` — probability ratio between current and old policy
- `ε = 0.2` — PPO clip range
- `β = 0.05` — KL penalty coefficient (low, allows meaningful divergence from base)
- `π_ref` — frozen reference model (base Qwen 2.5-7B) used to prevent reward hacking

### Why REINFORCE++ baseline

In our implementation we use `num_generations=2` per prompt (reduced from 4 for memory). This is the REINFORCE++ variant — using the group mean as a simple baseline rather than a learned value function. This is appropriate because:

- Our reward function is already well-shaped (Brier score is a proper scoring rule)
- A learned critic would require an extra ~14 GB on the A100
- REINFORCE++ has been shown to be more stable than GRPO for reasoning tasks (Logic-RL, PRIME papers)

### Training configuration

| Parameter | Value | Reason |
|---|---|---|
| Algorithm | GRPO / REINFORCE++ | No critic, memory efficient |
| Steps | 150 | ~22 Colab compute units on A100 |
| num_generations | 2 | Memory constraint (40 GB) |
| per_device_train_batch_size | 1 | A100 40 GB with 7B model |
| gradient_accumulation_steps | 4 | Effective batch = 4 |
| max_completion_length | 256 | Agent A response length |
| learning_rate | 5e-6 | Conservative — fine-tuning not pretraining |
| lr_scheduler | cosine | Standard for RL fine-tuning |
| warmup_steps | 10 | Short warmup, model already pretrained |
| beta (KL) | 0.05 | Low — allow meaningful policy change |
| bf16 | True | A100 native format, faster than fp16 |
| gradient_checkpointing | True | Reduces activation memory |

---

## 6. Reward Function Design

The reward function is the heart of the RLHF system. It encodes everything we care about in a healthcare regulatory AI system as a scalar signal.

### Full reward formula

```
R(prompt, response) =
    Brier(p, v)           # calibration reward
  + verdict_bonus         # judge agreement bonus
  + citation_bonus        # regulatory grounding
  - phi_penalty           # HIPAA compliance
  - overconfidence_penalty # hallucination gate
  - format_penalty        # response structure
```

### Component breakdown

#### Brier score calibration reward
```python
v = judge_verdict(response, prompt)   # ∈ [0, 1]
p = extract_confidence(response)      # ∈ [0, 1]
brier = 2 * p * v - p**2             # proper scoring rule
```

This is the **Brier score** — a proper scoring rule that simultaneously rewards accuracy and calibration. It is maximized when `p = v` — when the model's stated confidence exactly matches the Judge's verdict probability. Key properties:

- If model says `p=0.9` and Judge agrees `v=1.0` → reward = `2(0.9)(1.0) - 0.81 = 0.99`
- If model says `p=0.9` and Judge disagrees `v=0.0` → reward = `0 - 0.81 = -0.81`
- If model says `p=0.5` and Judge partially agrees `v=0.5` → reward = `0.5 - 0.25 = 0.25`

This came directly from PR #2's offline GRPO implementation and is preserved exactly.

#### Verdict bonus
```python
if v > 0.7:
    reward += 0.10
```
Extra reward when the Judge strongly agrees. Encourages the model to make claims the Judge finds compelling, not just calibrated ones.

#### Citation grounding bonus
```python
CITATION_PATTERNS = [
    r'\b(HIPAA|45\s*CFR|42\s*CFR|21\s*CFR)\b',
    r'\b(HL7|FHIR|ICD-?10|CPT|SNOMED)\b',
    r'\b(FDA|DEA|CMS|HHS|NIH)\b',
    r'\b(ISO\s*13485|EU\s*MDR|21st\s*Century\s*Cures)\b',
    r'§\s*\d+\.\d+',   # CFR section numbers e.g. §164.312
    ...
]
if has_citation(response):
    reward += 0.15
```
Rewards the model for grounding claims in specific regulatory sources. Without this, the model can score well on Brier but produce vague, uncited responses.

#### PHI penalty
```python
# Uses Microsoft Presidio for PII detection
results = analyzer.analyze(text=response, language="en",
    entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS",
              "US_SSN", "MEDICAL_LICENSE", "US_BANK_NUMBER"])
if any(r.score > 0.75 for r in results):
    reward -= 0.40
```
Hard penalty for any PHI in the response. Presidio runs named entity recognition across 8 PHI categories. Threshold of 0.75 prevents false positives while catching real PHI leakage.

#### Overconfidence-without-citation penalty
```python
if confidence > 0.85 and not has_citation(response):
    reward -= 0.30
```
High-confidence claims in healthcare without evidence are dangerous. This gate specifically targets hallucinated certainty — the model claiming `CONFIDENCE: 0.95` on an uncited claim.

#### Format compliance penalty
```python
if confidence is None:
    return -0.20   # no CONFIDENCE: line at all
```
Enforces the response format. Agent A must always end with `CONFIDENCE: <float>`. Missing this line returns an immediate `-0.20` and skips all other reward computation.

### Reward range

| Scenario | Approximate reward |
|---|---|
| Perfect: citation + calibrated + no PHI + Judge agrees | +0.99 to +1.35 |
| Good: citation + reasonable confidence | +0.40 to +0.80 |
| Ambiguous: some signals mixed (human review zone) | -0.10 to +0.30 |
| Bad: overconfident, no citation | -0.50 to -0.80 |
| PHI detected | -0.40 additional penalty |
| No CONFIDENCE line | -0.20 flat |

---

## 7. Human Feedback Loop

### Design philosophy

Not all reward signals can be captured by heuristics. A regulatory expert reading a response about vancomycin dosing in CKD patients knows whether it is correct — Presidio and regex do not. The human feedback loop routes ambiguous responses to a human reviewer who can apply domain expertise.

### Triage logic

```
Auto reward computed
        │
        ├── reward < -0.10  →  Clearly bad — use auto reward, skip human
        │
        ├── -0.10 ≤ reward ≤ +0.30  →  Ambiguous — show to human
        │                                Human clicks Good / Bad / Skip
        │
        └── reward > +0.30  →  Clearly good — use auto reward, skip human
```

The ambiguous zone (`-0.10` to `+0.30`) captures responses that have some good signals but are missing others — for example, a response with a citation but an unusual confidence value, or a response that looks medically plausible but uses an informal source.

### Human review interface

When a response falls in the ambiguous zone, a widget appears inline in the Colab notebook:

```
=================================================================
HUMAN REVIEW REQUIRED
Auto reward: +0.240 (ambiguous zone)
=================================================================

PROMPT:
What does 45 CFR §164.524 specify about a patient's right to 
access their PHI, including timeline and fee limitations?

AGENT A RESPONSE:
45 CFR §164.524 outlines the patient's right to access their 
Protected Health Information (PHI)...

=================================================================
[ Good (+0.80) ]  [ Bad (-0.60) ]  [ Skip (+0.24) ]
```

**Good (+0.80):** Human judges the response as correct, well-cited, appropriately calibrated. Reward overridden to `+0.80`.

**Bad (-0.60):** Response is misleading, incorrectly cited, or medically dangerous. Reward overridden to `-0.60`.

**Skip:** Human is unsure or the auto reward seems appropriate. Auto reward used as-is.

**Timeout (60s):** If no click within 60 seconds, auto reward is used and training continues.

### Feedback log

Every human review is logged to `human_feedback_log.json` on Google Drive:

```json
{
  "step": 12,
  "prompt": "What does 45 CFR §164.524 specify...",
  "completion": "45 CFR §164.524 outlines the patient's right...",
  "auto_reward": 0.24,
  "human_reward": 0.80,
  "human_reviewed": true
}
```

This log serves two purposes: audit trail for regulatory compliance, and a dataset for training a future reward model that can learn the human reviewer's preferences.

### Why this is real RLHF

The canonical RLHF pipeline (as used in InstructGPT and similar) has three stages: SFT, reward model training, RL fine-tuning. Our implementation collapses stages 1 and 3 (we use a pretrained model and fine-tune directly with RL) while preserving the human signal via the inline feedback loop. The human reward overrides make the "H" in RLHF real — not just a reward model trained on historical human data, but live human judgment applied per training step.

---

## 8. Model & Training Configuration

### Base model: Qwen/Qwen2.5-7B-Instruct

| Property | Value |
|---|---|
| Parameters | 7 billion |
| Architecture | Transformer decoder, GQA |
| Context length | 32,768 tokens |
| Training | Pretrained + instruction-tuned by Alibaba |
| License | Apache 2.0 |
| Format | bfloat16 |
| Size on disk | ~15.2 GB |

Qwen 2.5-7B was chosen because:
- It was already used in PR #2's offline GRPO experiment with proven results (8.4% Brier improvement)
- Strong performance on instruction following and structured output (the `CONFIDENCE:` format)
- Apache 2.0 license — compatible with enterprise SDK distribution
- 7B size fits on A100 40GB with LoRA fine-tuning headroom

### LoRA configuration

| Parameter | Value | Reason |
|---|---|---|
| r (rank) | 16 | Same as PR #2, sufficient for domain adaptation |
| alpha | 32 | Standard 2× rank scaling |
| target_modules | q_proj, v_proj, k_proj, o_proj | All attention projections |
| lora_dropout | 0.05 | Light regularization |
| bias | none | Standard for instruction-tuned models |
| task_type | CAUSAL_LM | Autoregressive generation |

LoRA adds approximately 40M trainable parameters on top of the 7B frozen base — about 0.57% of total parameters. This is what makes fine-tuning feasible on a single A100.

### Software stack

| Package | Version | Role |
|---|---|---|
| torch | 2.10.0+cu128 | Deep learning framework |
| transformers | 5.0.0 | Model loading and tokenization |
| trl | 1.3.0 | GRPOTrainer, reward function interface |
| peft | 0.19.1 | LoRA implementation |
| accelerate | 1.13.0 | Distributed training utilities |
| deepspeed | 0.18.9 | ZeRO optimization (available, not used in single-GPU mode) |
| presidio-analyzer | latest | PHI/PII detection |
| presidio-anonymizer | latest | PHI scrubbing in SDK |
| spacy | 3.x + en_core_web_lg | NLP backbone for Presidio |
| ray | 2.55.1 | Available for future multi-GPU scaling |
| CUDA | 12.8 | GPU compute |

---

## 9. Dataset & Domain Prompts

### Training prompt construction

Each training prompt is formatted using Qwen's chat template with a MAD-specific system prompt:

```
<|im_start|>system
You are Agent A in a Multi-Agent Debate pipeline for healthcare 
regulatory AI. Your role is to provide a well-reasoned, 
evidence-based claim with a calibrated confidence score.
RULES:
(1) Never include PHI or patient identifiers.
(2) Always cite a specific source: HIPAA section, CFR part, 
    drug label, or peer-reviewed study.
(3) End your response with: CONFIDENCE: <float 0.0-1.0>.
(4) If uncertain, express lower confidence — do not hallucinate certainty.
<|im_end|>
<|im_start|>user
[regulatory query]
<|im_end|>
<|im_start|>assistant
```

### Prompt categories

**HIPAA & Privacy (6 prompts)**
- Technical safeguards under 45 CFR §164.312
- Patient right of access under 45 CFR §164.524
- Breach notification timelines
- Safe Harbor de-identification (18 PHI identifiers)
- De-identified data for AI training
- PHI sharing with employers (GINA Title II)

**FDA & Drug Regulation (4 prompts)**
- Off-label prescribing (metformin for prediabetes)
- Drug-drug interactions (clopidogrel + omeprazole)
- Adverse event reporting under 21 CFR §803
- Off-label drug promotion regulations

**Medical Devices & AI (3 prompts)**
- EU MDR 2017/745 classification for AI diagnostic tools
- FDA 510(k) vs PMA for AI imaging
- ISO 13485 post-market surveillance for SaMD

**Health IT & Interoperability (3 prompts)**
- HL7 FHIR R4 patient consent resources
- CMS Interoperability Rule (CMS-9115-F) FHIR APIs
- 21st Century Cures Act information blocking

**Clinical & Regulatory (4 prompts)**
- Vancomycin dosing in CKD stage 3
- DEA schedule II opioid prescribing violations
- CMS conditions of participation (42 CFR §482.42)
- 21 CFR Part 11 electronic signatures

**Total: 20 unique regulatory prompts**

Note: The repo's `rlhf/data/healthcare_test_queries.json` (50 queries) and `rlhf/data/mad_stress_test_queries.json` (40 queries) from PR #2 are not yet merged to the deploy branch. When merged, the dataset will automatically expand to 110 prompts — the loader handles both sources with deduplication.

---

## 10. SDK Architecture

### GuardrailsSDK class

The fine-tuned LoRA adapter is packaged inside a `GuardrailsSDK` wrapper class that provides a clean API for enterprise callers:

```python
from guardrails_sdk.core import GuardrailsSDK

sdk = GuardrailsSDK("/path/to/guardrails_sdk_model")
response = sdk.query("What does HIPAA §164.312 require for ePHI access controls?")

print(response.response)       # The answer, PHI-scrubbed
print(response.confidence)     # 0.82 — Agent A's calibrated confidence
print(response.has_citation)   # True — regulatory source cited
print(response.phi_detected)   # False — no PHI in response
print(response.blocked)        # False — passed policy check
```

### GuardrailsResponse fields

| Field | Type | Description |
|---|---|---|
| response | str | Agent A's answer, PHI-scrubbed if needed |
| confidence | float or None | Extracted CONFIDENCE: value |
| has_citation | bool | Whether a regulatory source was cited |
| phi_detected | bool | Whether Presidio found PHI (before scrubbing) |
| blocked | bool | Whether the query was blocked by policy |
| block_reason | str or None | Reason for block if blocked=True |

### Policy blocks

The SDK blocks queries matching these patterns before they reach the model:

| Pattern | Block reason |
|---|---|
| Violence-related queries | `violence` |
| PHI in the prompt itself | `phi_in_prompt` |
| Jailbreak attempts | `jailbreak` |

### PHI scrubbing pipeline

If the model produces a response containing PHI (detected by Presidio with score > 0.75), the SDK automatically scrubs it using `presidio-anonymizer` before returning:

```
Model response with PHI → Presidio analyzer → Presidio anonymizer → Clean response
"Patient John Smith (SSN 123-45-6789)..." → "[PERSON] ([US_SSN])..."
```

### Model loading

The SDK loads the merged model (base weights + LoRA adapter combined via `openrlhf.cli.lora_combiner` or PEFT's `merge_and_unload`):

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
```

---

## 11. Infrastructure & Hardware

### Current setup: Colab A100

| Resource | Spec |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |
| VRAM | 40 GB HBM2e |
| CUDA | 12.8 |
| RAM | ~83 GB system |
| Storage | Ephemeral + Google Drive for persistence |
| Compute budget | 100 Colab compute units |

**VRAM allocation during training:**
- Model weights (bfloat16, LoRA only trainable): ~15 GB
- Optimizer states (AdamW, CPU offload): ~0 GB GPU
- Activations + gradient checkpointing: ~8 GB
- Generation buffer (2 completions per prompt): ~6 GB
- Total peak: ~30-32 GB (leaving ~8 GB headroom)

**Compute unit budget:**
- Install + setup: ~2 CU
- 150-step training run: ~22 CU
- Evaluation: ~5 CU
- Total per full run: ~29 CU
- Remaining from 100 CU budget: ~71 CU (allows ~2 more full runs)

### Future setup: RTX 5090 + A100 split

For production training beyond the 2-day sprint, the intended architecture splits work across two machines:

| Component | RTX 5090 (32 GB GDDR7) | A100 (40 GB HBM2e) |
|---|---|---|
| Role | Inference / rollout generation | Training / gradient computation |
| Runs | vLLM server, MAD environment, reward model | Actor training, reference model, checkpointing |
| Advantage | High memory bandwidth (1.8 TB/s) for fast generation | HBM2e optimized for training kernels |

This split enables true async RLHF via OpenRLHF once a wheel compatible with torch 2.10 + CUDA 12.8 is released.

### Checkpoint persistence

All checkpoints are saved to Google Drive at:
```
MyDrive/guardrails_openrlhf/
├── checkpoints/
│   ├── step_50/          # LoRA adapter at step 50
│   ├── step_100/         # LoRA adapter at step 100
│   └── final/            # Final merged model
├── human_feedback_log.json
├── eval_results.json
└── training.log
```

Training automatically resumes from the latest checkpoint if the Colab session disconnects.

---

## 12. Implementation Journey & Decisions

### Original plan: OpenRLHF with Ray + vLLM

The initial design (PR #2 extension) used OpenRLHF's `train_reinforce_ray.py` with:
- Ray for distributed scheduling
- vLLM for fast async rollout generation
- REINFORCE++ without a critic model
- Actor + vLLM sharing the A100 via Hybrid Engine sleep-mode

This would have provided true async live RLHF — generation and training running concurrently.

### Why we pivoted to TRL

The Colab A100 environment has torch 2.10.0 + transformers 5.0.0 + CUDA 12.8 — a very new stack as of May 2026. OpenRLHF's pip package pins older torch versions in its dependencies and pip spent 20+ minutes failing to resolve conflicts. Specific blockers encountered:

| Package | Issue |
|---|---|
| `openrlhf[vllm]` | Dependency resolver loop — never resolved in 20+ min |
| `flash-attn` | No pre-built wheel for torch 2.10 + CUDA 12.8 — fell back to source compile, hung for 17+ min |
| `torchao` | Version 0.10.0 installed but peft requires 0.16.0+; 0.17.0 available but requires torch 2.11+ |

Resolution: uninstall `torchao`, use TRL 1.3.0 which ships native GRPO and is fully compatible with the environment.

### What TRL GRPO vs OpenRLHF REINFORCE++ means

| Aspect | TRL GRPO | OpenRLHF REINFORCE++ |
|---|---|---|
| Generation | Sequential (generate then train) | Async (generate while training) |
| Throughput | ~0.01 it/s on A100 | ~3-5x faster with vLLM async |
| Reward interface | Python function, synchronous | HTTP server or Python, async |
| Human feedback | Inline Colab widget | Would need separate UI |
| Setup complexity | Run one cell | Requires Ray cluster + vLLM |
| Compatibility | torch 2.10 + transformers 5.0 ✅ | Blocked by dependency conflicts ❌ |

TRL GRPO is functionally equivalent for our use case — the LoRA adapter output is identical, the reward function is identical, the training signal is identical. The only difference is throughput.

### GRPOConfig API changes in TRL 1.3.0

TRL 1.3.0 removed `max_prompt_length` from `GRPOConfig` — prompt length is now controlled by the tokenizer's `max_length` during dataset preprocessing. Other renamed parameters discovered via `inspect.signature`:

- `max_prompt_length` → removed (tokenizer handles it)
- `use_vllm` → still present, set to `False`
- `num_generations` → unchanged
- `max_completion_length` → unchanged

---

## 13. Results & Evaluation

### PR #2 baseline (offline GRPO)

The original offline GRPO experiment from PR #2 established the baseline:

| Metric | Base Model | Fine-tuned (PR #2) | Delta |
|---|---|---|---|
| Brier score | 0.2467 | 0.2258 | **8.4% improvement** |
| Confidence calibration | 5/12 | 6/12 | +1 case |
| Training data | 52 examples | — | — |
| Training time | 53 min on A100 | — | — |

### Current run (online GRPO + human feedback)

Live training in progress. Evaluation runs after step 150 using the same 12 regulatory test cases from PR #2's `run_eval.py`.

**Metrics tracked:**
- Average reward per step (logged every 5 steps)
- Training loss
- KL divergence from reference model
- Per-case: reward, confidence, citation rate, PHI violations
- Human review count and override rate

**Expected improvements over PR #2 baseline:**
- Higher citation rate — citation bonus explicitly in reward
- Lower PHI violation rate — Presidio-based penalty
- Better calibration in ambiguous zone — human overrides provide signal that heuristics cannot

### Evaluation prompts (12 regulatory test cases)

1. What does HIPAA §164.312 require for ePHI technical safeguards?
2. Is off-label metformin prescribing for prediabetes supported by FDA?
3. What are the 18 PHI identifiers in the HIPAA Safe Harbor method?
4. Under 21 CFR §803, what is the MDR adverse event reporting timeline?
5. Does CMS-9115-F require FHIR APIs for patient data access?
6. What does GINA Title II prohibit regarding genetic information in employment?
7. What is the standard of care for vancomycin dosing in CKD stage 3?
8. How does EU MDR classify an AI-based diagnostic imaging SaMD?
9. What constitutes information blocking under the 21st Century Cures Act?
10. Can de-identified data be used for AI training without HIPAA consent?
11. What are the 510(k) vs PMA differences for AI diagnostic tools?
12. What are mandatory infection control requirements under 42 CFR §482.42?

---

## 14. File Structure

```
guardrails-enterprise/
├── rlhf/
│   ├── README.md                          # This document
│   ├── data/
│   │   ├── healthcare_test_queries.json   # 50 healthcare queries (PR #2)
│   │   └── mad_stress_test_queries.json   # 40 regulatory stress queries (PR #2)
│   └── pipeline/
│       ├── pipeline.py                    # Brier reward + MAD db loader (PR #2)
│       ├── run_on_real_data.py            # Colab script for real MAD data (PR #2)
│       ├── run_training.py                # GRPO training script (PR #2, offline)
│       ├── run_eval.py                    # 12-case evaluation script (PR #2)
│       └── verify.py                      # Pipeline verification (PR #2)
│
├── openrlhf/                              # New: live RLHF implementation
│   ├── guardrails_openrlhf_colab.ipynb   # Main Colab notebook
│   └── reward_fn/
│       └── healthcare_reward.py          # Reward function (Brier + PHI + citation)
│
└── guardrails_sdk/
    └── core.py                            # SDK wrapper class
```

### Google Drive output structure

```
MyDrive/guardrails_openrlhf/
├── checkpoints/
│   ├── step_50/                   # Checkpoint at step 50
│   ├── step_100/                  # Checkpoint at step 100
│   ├── step_150/                  # Checkpoint at step 150
│   └── final/                     # Merged model (base + LoRA)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── guardrails_sdk/
│           └── core.py
├── human_feedback_log.json        # All human review decisions
├── eval_results.json              # Base vs fine-tuned comparison
├── eval_results.png               # Reward bar chart
├── training_curve.png             # Reward trajectory plot
└── training.log                   # Full training log
```

---

## 15. How to Run

### Prerequisites

- Google Colab with A100 GPU (Runtime → Change runtime type → A100)
- Google Drive mounted
- No HF token required (Qwen 2.5-7B is public)

### Step-by-step execution

```
§1  GPU check          — verify A100, CUDA 12.8
§2  Install            — deepspeed, trl, ray, presidio, matplotlib
§3  Mount Drive        — checkpoint persistence
§4  Clone repo         — deploy branch of guardrails-enterprise
§5  Build dataset      — 20 regulatory prompts → JSONL
§6  Write reward fn    — healthcare_reward.py to /content/reward_fn/
    Self-test          — verify all 4 reward cases
§7  SKIP               — Ray not needed for TRL single-GPU
§8a Load prompts       — import reward fn, load JSONL
§8b Human feedback UI  — define hybrid_reward_fn with widget
§8c Config             — GRPOConfig + LoraConfig
§8d Train              — GRPOTrainer.train() — ~45-70 min
§9  Evaluate           — base vs fine-tuned on 12 test cases
§10 Export             — lora_combiner → merged SDK model
§11 Plot               — training reward curve from log
```

### During training

When a human review widget appears:
1. Read the PROMPT and AGENT A RESPONSE carefully
2. Check: Does it cite a real regulatory source correctly?
3. Check: Is the confidence score appropriate for the claim's certainty?
4. Check: Is the answer medically/legally accurate?
5. Click **Good** if yes to all, **Bad** if clearly wrong, **Skip** if unsure

You have 60 seconds per review. Training continues automatically on timeout.

### Resuming after disconnect

If Colab disconnects mid-training, the session state is lost but checkpoints are on Drive. To resume:
1. Re-run §2 (installs)
2. Re-run §3 (mount Drive)
3. Re-run §8a, §8b, §8c
4. Re-run §8d — it auto-detects the latest checkpoint in `CKPT_DIR` and resumes

### OOM troubleshooting

If you get `OutOfMemoryError`:

```python
# Before re-running trainer:
import torch, gc
try: del trainer
except: pass
gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.mem_get_info()[0]/1024**3, "GB free")
```

Then reduce in GRPOConfig:
- `num_generations`: 2 → 1
- `max_completion_length`: 256 → 128

---

## 16. Next Steps

### Immediate (after current training run)

1. **Merge PR #2 to deploy branch** — adds 90 more training prompts (50 healthcare + 40 MAD stress), immediately improves training data diversity
2. **Run evaluation** (§9) — compare base vs fine-tuned on 12 regulatory test cases
3. **Export SDK model** (§10) — merge LoRA adapter + base weights, package with `guardrails_sdk/core.py`
4. **Second training iteration** — use human feedback log from run 1 to identify weak areas, add targeted prompts

### Short term (week 1-2)

5. **Real MAD Judge integration** — replace the heuristic `_judge_verdict()` in `healthcare_reward.py` with actual Agent B + Judge LLM calls using the `run_on_real_data.py` pipeline from PR #2
6. **Expand prompt dataset** — add MIMIC-III clinical notes (de-identified), PubMed abstracts, and additional CFR sections
7. **Reward model training** — use `human_feedback_log.json` to train a small reward model that approximates the human reviewer, enabling fully automated RLHF at scale
8. **SDK packaging** — `pip install guardrails-enterprise` with the merged model bundled or downloaded on first use

### Medium term (month 1)

9. **OpenRLHF migration** — once a wheel compatible with torch 2.10+ ships, migrate to true async live RLHF with Ray + vLLM for 3-5x throughput improvement
10. **RTX 5090 + A100 split** — implement the two-machine architecture: 5090 serves vLLM rollouts and reward model, A100 handles gradient computation
11. **Multi-agent reward** — extend reward function to Agent B and Judge, not just Agent A — full MAD pipeline fine-tuning
12. **Regulatory corpus RAG** — add retrieval-augmented generation from a live regulatory document store to ground citation checks against actual document content rather than regex patterns

### Long term

13. **Continuous online learning** — deploy the SDK in shadow mode, collect human feedback on production queries, retrain weekly
14. **Multi-domain expansion** — financial regulatory AI (SEC, FINRA), legal AI (court documents, contracts) using the same reward function architecture
15. **Federated RLHF** — train across multiple hospital systems without sharing patient data, using federated gradient aggregation

---

*Document generated: May 2026*
*Model: Qwen/Qwen2.5-7B-Instruct + LoRA r=16*
*Algorithm: GRPO / REINFORCE++ via TRL 1.3.0*
*Hardware: NVIDIA A100-SXM4-40GB, CUDA 12.8*
*Repository: github.com/shashidharbabu/guardrails-enterprise (deploy branch)*
