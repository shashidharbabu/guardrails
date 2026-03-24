# GRPO Feedback Loop

This directory contains the GRPO (Group Relative Policy Optimization) feedback loop for fine-tuning the MAD pipeline agents using data collected from live pipeline runs.

## Status
🚧 In Development — Phase 3 work. Depends on MAD pipeline (`multi_agent_debate/`) writing correctly to SQLite.

## Purpose

Fine-tune Agent A and Agent B using reward signals derived from judge verdicts and attack effectiveness. The feedback loop reads the 4 SQLite tables written by the MAD pipeline and writes reward values back — the MAD pipeline code never touches reward columns.

## Training targets

| Phase | Agent | Method | Status |
|-------|-------|--------|--------|
| Phase 1 | Agent A | GRPO via TRL (Brier score reward) | 🚧 Pending |
| Phase 2 | Agent B | GRPO via TRL (precision reward) | 🚧 Pending |

## Reward formulas

### Agent A — Brier score reward (calibration)
```python
brier_reward = 2 * p_final * v_label - p_final ** 2
```
- `p_final` — Agent A's final confidence score for the claim (from `claims.confidence_p` at `post_cycle2`)
- `v_label` — Judge's ground truth score: 1.0 / 0.5 / 0.0 (from `judge_verdicts.v_label`)

### Agent B — Precision reward
```python
b_reward = +1  # attacked a wrong claim (v=0 or 0.5) AND delta_p >= 0.2
b_reward = -1  # attacked a correct claim (v=1.0) — gaslighting penalty
b_reward =  0  # attack had no meaningful effect
```

### is_clean filter
```python
is_clean = 1 if b_reward != -1 else 0
```
Only `is_clean = 1` records go into the TRL GRPO trainer. Records where Agent B gaslit Agent A (lowered confidence on a valid claim) are excluded from Agent A's training data.

### GRPO advantage
```python
grpo_advantage = rollout_total - mean_across_rollouts
```
Run the same query multiple times (multiple `rollout_id` values) to compute the mean for normalisation.

## SQLite tables consumed (written by MAD pipeline)

| Table | What the feedback loop uses |
|-------|-----------------------------|
| `claims` | `agent_a_prompt` (GRPO training input), `confidence_p` at `post_cycle2` |
| `attacks` | `p_before_attack`, `p_after_attack` → compute `delta_p` for `b_reward` |
| `judge_verdicts` | `v_label` → ground truth for Brier reward |
| `queries` | join key; feedback loop writes `brier_reward` and `is_clean` back |

## Columns written by feedback loop (NOT by MAD code)

- `attacks.b_reward` — precision reward for Agent B
- `rewards` table — created by feedback loop: `brier_reward`, `is_clean`, `grpo_advantage`

## Planned Components
- Reward computation script (reads SQLite, writes `b_reward` and `rewards` table)
- TRL GRPO trainer integration for Agent A (Phase 1)
- TRL GRPO trainer integration for Agent B (Phase 2)
- Evaluation harness against synthetic eval dataset
