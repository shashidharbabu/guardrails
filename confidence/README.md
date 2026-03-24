# Confidence Scoring Engine (CSE)

Computes a final confidence score for each MAD pipeline run and drives the routing decision (DELIVER / RETRY / HARD_BLOCK / HUMAN_REVIEW).

## Status
🚧 In Development — Phase 2 work. Current implementation is v0.1 (judge-only aggregate).

## Scoring formula (full — Phase 2)

```
final_score = 0.30 × F_llm + 0.25 × (1 − H_llm) + 0.10 × relevancy + 0.35 × judge_eval_score
```

| Component | Weight | Source | Description |
|-----------|--------|--------|-------------|
| `F_llm` | 0.30 | DeepEval | LLM answer faithfulness to retrieved context |
| `H_llm` | 0.25 | DeepEval | LLM hallucination rate (`1 − H_llm` = non-hallucination score) |
| `relevancy` | 0.10 | DeepEval | Context relevancy of retrieved RAG chunks |
| `judge_eval_score` | 0.35 | MAD Judge | Min-aggregated judge score across material claims |

## Current implementation (v0.1)

Uses **judge-only aggregate** — the min of all material claim judge scores from the MAD pipeline:

```python
judge_eval_score = min(jv.score for jv in judge_verdicts if jv.is_material)
final_score = judge_eval_score   # v0.1: full formula pending DeepEval integration
```

The v0.1 score is what drives routing in the current MAD pipeline. When reporting results, document this as "judge-only aggregate (v0.1)" — do not claim the full CSE formula is implemented.

## Routing thresholds

| Score | Routing decision |
|-------|-----------------|
| any material claim with judge score = 0.0 | `HARD_BLOCK` (hard rule, checked first) |
| ≥ 0.8 | `DELIVER` |
| 0.4 – 0.8 | `RETRY` (send correction signal to LLM) |
| < 0.4 | `HUMAN_REVIEW` |

## Integration

The CSE writes the final score and routing decision back to the `queries` table in SQLite (`final_cse_score`, `routing_decision` columns) after each MAD pipeline run.

LangSmith tracing (`@traceable` decorators) is planned for observability — the CSE formula computation stays in Python; LangSmith wraps around it for debugging only.

## Planned Components
- DeepEval Layer 1 integration (faithfulness, hallucination, relevancy metrics)
- Full 4-component formula implementation
- LangSmith tracing decorators
- Score calibration and threshold tuning against the synthetic evaluation dataset
