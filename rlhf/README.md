# GRPO Feedback Loop

This package reads the four MAD SQLite tables under `multi_agent_debate/` and writes **`attacks.b_reward`** plus a dedicated **`rewards`** table. The MAD pipeline never touches reward columns.

## Status

Implemented: batch scoring, heuristic shaping (optional Presidio), GRPO advantage normalization, FastAPI human-review queue, JSONL export for Colab, standalone Colab reward module under `openrlhf/reward_fn/`.

Heavy ML (`torch`, `trl`, `peft`) stays **out** of this requirements file - use Colab or a GPU image for training.

## Reward formulas

### Agent A - Brier calibration

```text
brier_reward = 2 * p_final * v_label - p_final ** 2
```

- `p_final` - `claims.confidence_p` at `checkpoint = post_cycle2`
- `v_label` - `judge_verdicts.v_label` (0.0 / 0.5 / 1.0)

Composite **`auto_reward`** adds verdict/citation bonuses and PHI/overconfidence/format penalties per `docs/RLHF_IMPLEMENTATION.md` (applied to a synthetic text bundle from `claim_text` + `reasoning` + `verdict`).

### Agent B - precision

```text
b_reward = +1   # weak claim (v < 1) and confidence drop >= 0.2 after attack
b_reward = -1   # v = 1.0 and confidence drop >= 0.2 (gaslighting)
b_reward =  0   # otherwise
```

Written to **`attacks.b_reward`**.

### `is_clean`

Per claim: `is_clean = 0` if **any** attack row for that `(query_id, rollout_id, claim_id)` has `b_reward = -1`. Otherwise `1`. Only clean rows should be exported for Agent A GRPO by default.

### GRPO advantage

For each `query_id`, group all `rollout_id` values. For each rollout, `rollout_total = SUM(final_reward)` over **material** claims with `is_clean = 1` and non-null `final_reward`. Then:

```text
grpo_advantage = rollout_total - mean(rollout_total across rollouts)
```

Stored on every `rewards` row for that rollout (same scalar per row).

## SQLite contract

| Table | Written by | Feedback loop |
|-------|------------|---------------|
| `queries`, `claims`, `attacks`, `judge_verdicts` | MAD | Read |
| `attacks.b_reward` | Feedback loop | `UPDATE` |
| `rewards` | Feedback loop | `CREATE` + `INSERT`/`UPDATE` |

The feedback loop does **not** write scores back onto `queries`.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MAD_DB_PATH` | `<repo>/multi_agent_debate/mad_store.db` | Same DB file as MAD |
| `HUMAN_FEEDBACK_LOG_PATH` | `<repo>/human_feedback_log.jsonl` | Append-only audit log |
| `FEEDBACK_TRIAGE_LOW` / `FEEDBACK_TRIAGE_HIGH` | `-0.10` / `0.30` | Ambiguous band for `/review/next` |
| `FEEDBACK_USE_PRESIDIO` | `1` | Set `0` to skip PHI penalty if Presidio unavailable |
| `FEEDBACK_API_HOST` / `FEEDBACK_API_PORT` | `0.0.0.0` / `8002` | Human review API |

## Install (feedback loop only)

From repository root:

```bash
pip install -r rlhf/requirements-feedback.txt
python -m spacy download en_core_web_lg   # if using Presidio PHI
```

## Commands

**Batch score + GRPO advantage**

```bash
PYTHONPATH=. python -m rlhf.pipeline.run_reward_scoring
PYTHONPATH=. python -m rlhf.pipeline.run_reward_scoring --db /path/to/mad_store.db --no-advantage
```

**Export JSONL for Colab / TRL**

```bash
PYTHONPATH=. python -m rlhf.pipeline.export_grpo_dataset /tmp/grpo_train.jsonl
PYTHONPATH=. python -m rlhf.pipeline.export_grpo_dataset /tmp/all.jsonl --include-unclean
```

**Human review API**

```bash
PYTHONPATH=. python -m rlhf.feedback_loop.api
# POST /rewards/compute - re-run batch scoring
# GET  /review/next - next ambiguous item
# POST /review/{query_id}/{rollout_id}/{claim_id}  body: {"decision":"good"|"bad"|"skip"}
# GET  /review/log?tail=50
```

**After a MAD CLI test (repo root on `PYTHONPATH`)**

```bash
cd multi_agent_debate
PYTHONPATH=.. python -m multi_agent.run_test --verify-storage --run-feedback-loop
```

## Tests

```bash
pytest rlhf/tests/
```

## Colab

See [openrlhf/guardrails_openrlhf_colab.ipynb](../openrlhf/guardrails_openrlhf_colab.ipynb) and [openrlhf/reward_fn/healthcare_reward.py](../openrlhf/reward_fn/healthcare_reward.py) for TRL-oriented completion + judge scoring (copyable to `/content/reward_fn/`).
