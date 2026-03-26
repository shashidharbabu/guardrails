# GRPO Feedback Loop — Agent A Fine-tuning

Fine-tunes **Agent A** (the MAD pipeline's fact-checking agent) using GRPO (Group Relative Policy Optimization) with Brier score rewards derived from Judge verdicts.

## Status

- Pipeline: Verified end-to-end on real MAD database (52 training examples from 68 post_cycle2 claims)
- Training: Completed on Colab A100 — Qwen 2.5 7B Instruct with LoRA adapter
- Evaluation: 8.4% Brier score improvement (0.2467 → 0.2258) on 12 regulatory test cases

## Directory Structure

```
rlhf/
├── README.md
├── pipeline/
│   ├── pipeline.py            # Core reward computation + training data generation
│   ├── run_on_real_data.py    # Self-contained Colab script (no imports needed)
│   └── verify.py              # Multi-phase pipeline verification
├── agent_a/
│   ├── run_training.py        # GRPO training script for Colab (T4/A100)
│   └── run_eval.py            # Base vs fine-tuned evaluation (12 test cases)
└── data/
    ├── mad_stress_test_queries.json    # 40 regulatory stress test queries
    └── healthcare_test_queries.json   # 50 healthcare domain queries
```

## How to Run (Google Colab)

### Step 1: Compute rewards from MAD database

Upload `run_on_real_data.py` and your MAD database (`mad_store.db`) to Colab:

```python
!python run_on_real_data.py
```

This reads the MAD pipeline's SQLite database, computes Brier rewards for every Agent A claim, filters out gaslighted and noisy records, and outputs `real_training_data.json`.

### Step 2: Train Agent A

Upload `run_training.py` and `real_training_data.json` to Colab:

```python
!python run_training.py
```

Runs a 1-step smoke test, then full GRPO training. Saves LoRA adapter to `/content/grpo_output/final_adapter/`.

### Step 3: Evaluate

Upload `run_eval.py` to Colab (same session):

```python
!python run_eval.py
```

Runs 12 regulatory test cases comparing base Qwen 2.5 7B vs fine-tuned model. Outputs Brier scores, verdict accuracy, and confidence calibration.

## Model & Training Details

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| Adapter | LoRA r=16, alpha=32, targets: q_proj, v_proj |
| Algorithm | GRPO (TRL) |
| Batch size | 1 (gradient accumulation: 4) |
| Learning rate | 1e-4 (cosine schedule) |
| Generations per prompt | 4 |
| Max completion length | 512 tokens |
| Training data | 52 examples from real MAD pipeline |

## Reward Formula

### Brier Score Reward (calibration)

```
R = 2 * p * v - p^2
```

- `p` — Agent A's final confidence (from `claims.confidence_p` at `post_cycle2`)
- `v` — Judge's ground truth: 1.0 (supported), 0.5 (partial), 0.0 (not supported)

| Scenario | p | v | Reward |
|---|---|---|---|
| Correct + confident | 0.95 | 1.0 | +0.9975 |
| Wrong + overconfident | 0.65 | 0.0 | -0.4225 |
| Wrong + honest | 0.00 | 0.0 | 0.0000 |

### Verdict Bonus

- +0.2 for correct verdict label (SUPPORTED/PARTIAL/NOT_SUPPORTED)
- -0.2 for incorrect verdict label
- Total reward clipped to [-1, +1]

## Data Filters

Records are filtered before training:

1. **Gaslighting filter** (`is_clean`): Excludes records where Agent B wrongly pushed Agent A to doubt a correct claim (`b_reward = -1`)
2. **Double-uncertain filter**: Excludes records where `v_label = 0.5` AND `0.4 <= p_final <= 0.6` (uninformative for learning)

## SQLite Tables Consumed

| Table | What the feedback loop reads |
|---|---|
| `claims` | `agent_a_prompt` (training input), `confidence_p` at `post_cycle2` |
| `attacks` | `p_before_attack`, `p_after_attack` for `b_reward` computation |
| `judge_verdicts` | `v_label` — ground truth for Brier reward |

## Evaluation Results (Real Data)

| Metric | Base Model | Fine-tuned | Change |
|---|---|---|---|
| Brier score | 0.2467 | 0.2258 | 8.4% better |
| Confidence calibration | 5/12 | 6/12 | +1 |
| Verdict accuracy | 5/12 | 4/12 | -1 |

First iteration with 52 training examples. Improvement expected to compound with more data and training iterations.

## Dependencies

```
torch
transformers>=4.40.0
peft>=0.10.0
trl>=0.8.0
datasets
accelerate
bitsandbytes
```

All dependencies are auto-installed by the training and eval scripts when run on Colab.
