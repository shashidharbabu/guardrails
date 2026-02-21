# Synthetic Data Validation Report

## Scope

This report validates the latest smoke-test outputs for both synthetic-data pipelines using:
- current validator outputs in the generated artifacts
- RAGAS metric outputs generated post-validation
- manual spot review of sample records

## Inputs Reviewed

- Embedding outputs:
  - `synthetic_data/_embedding_run_ragas_v2/synthetic/embedding/embedding_triplets_raw.jsonl`
  - `synthetic_data/_embedding_run_ragas_v2/synthetic/embedding/embedding_triplets_validated.jsonl`
  - `synthetic_data/_embedding_run_ragas_v2/synthetic/embedding/rejected/embedding_triplets_rejected.jsonl`
  - `synthetic_data/_embedding_run_ragas_v2/synthetic/embedding/eval/ragas_metrics.jsonl`
  - `synthetic_data/_embedding_run_ragas_v2/synthetic/embedding/eval/ragas_summary.json`
- LLM outputs:
  - `synthetic_data/_llm_run_ragas_v2/synthetic/llm/agent_pairs_raw.jsonl`
  - `synthetic_data/_llm_run_ragas_v2/synthetic/llm/agent_pairs_validated.jsonl`
  - `synthetic_data/_llm_run_ragas_v2/synthetic/llm/rejected/agent_pairs_rejected.jsonl`
  - `synthetic_data/_llm_run_ragas_v2/synthetic/llm/eval/ragas_metrics.jsonl`
  - `synthetic_data/_llm_run_ragas_v2/synthetic/llm/eval/ragas_summary.json`

## Summary Metrics

### Embedding pipeline (triplets)
- Raw: `5`
- Validated: `4`
- Rejected: `1`
- Top reject reasons:
  - `negative_not_semantically_related`: `1`

### LLM pipeline (instruction-response pairs)
- Raw: `5`
- Validated: `0`
- Rejected: `5`
- Top reject reasons:
  - `policy_citation_not_in_context`: `5`

### RAGAS aggregate metrics

### Embedding RAGAS
- Samples scored: `4`
- Metrics: `answer_relevancy`
- Mean scores:
  - `answer_relevancy`: `0.3720`

### LLM RAGAS
- Samples scored: `5` (scored on raw because validator kept `0`)
- Metrics: `answer_relevancy`
- Mean scores:
  - `answer_relevancy`: `0.2944`

## Manual Quality Review

### Embedding triplets
- Query/positive alignment looks coherent in sampled records.
- Hard negatives are from different docs and generally related but not identical.
- Validated records include `validation.semantic_similarity`, so acceptance logic is traceable.

Conclusion: **acceptable for smoke-test stage**, but still needs larger-run quality monitoring before production-scale generation.

### LLM pairs
- Generated outputs are structurally complete and decision labels are plausible.
- All records fail validator citation checks because `policy_violations[].policy` strings do not consistently include retrieved source ids (e.g., `t0__...`), which the current validator requires.
- RAGAS scores are available for observability but do not override validator hard-gates.

Conclusion: **not up to mark yet** under current validation criteria.

## Readiness Verdict

- Embedding pipeline: **PASS (smoke-test readiness)**.
- LLM pipeline: **FAIL (validator mismatch blocker; RAGAS monitoring available)**.

## Recommended Next Step

Resolve LLM citation mismatch before larger runs by aligning one of:
- generation format (force policy citations to include `retrieved_context.source` ids), or
- validator rule (accept normalized policy names + source mapping strategy).
