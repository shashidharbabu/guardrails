# Synthetic Evaluation Dataset

This directory contains synthetic data pipelines for both embedding and LLM finetuning, and generates a labelled evaluation dataset for benchmarking the MAD pipeline end-to-end.

## Status
🚧 In Development — build after MAD pipeline is verified running.

## Purpose
Generate high-quality synthetic training data for:
- Embedding retrieval finetuning (triplets)
- LLM safety/decision finetuning (instruction-response pairs)
- Ground-truth evaluation dataset to test MAD pipeline routing accuracy

## Evaluation Dataset Specification

| Property | Value |
|----------|-------|
| Size | 200 examples |
| Domain | Healthcare / Hospital (primary evaluation domain) |
| Generator | Claude API (NOT Qwen2.5 — avoids circular evaluation) |
| Human labels | Required — judge verdicts (v=1.0/0.5/0.0) per claim must be human-validated |

### Error types (50 examples each)

| Error type | Expected MAD routing |
|-----------|---------------------|
| `fully_correct` | `DELIVER` |
| `missing_caveat` | `RETRY` |
| `hallucinated_specific` | `HARD_BLOCK` |
| `jurisdiction_blind` | `HARD_BLOCK` or `RETRY` |

## Split Pipeline Entry Points

### 1) Embedding pipeline
Runs load/filter/enrich + triplet generation + triplet validation.

```bash
python -m synthetic_data.run_embedding_pipeline \
  --chunks-dir rag/chunk_test \
  --triplets 5 \
  --enrich-limit 120 \
  --out-dir synthetic_data/_embedding_run \
  --debug
```

Artifacts are written under:
- `synthetic_data/_embedding_run/synthetic/embedding/`
- rejected records under `synthetic_data/_embedding_run/synthetic/embedding/rejected/`

### 2) LLM pipeline
Runs load/filter/enrich + LLM pair generation + pair validation.

```bash
python -m synthetic_data.run_llm_pipeline \
  --chunks-dir rag/chunk_test \
  --pairs 10 \
  --enrich-limit 120 \
  --out-dir synthetic_data/_llm_run \
  --debug
```

Artifacts are written under:
- `synthetic_data/_llm_run/synthetic/llm/`
- rejected records under `synthetic_data/_llm_run/synthetic/llm/rejected/`

## Notes
- Both pipelines require `ANTHROPIC_API_KEY`.
- LLM-pair schema keeps `retrieved_context` and does not require `reasoning`.
- Triplet validation uses semantic similarity between positive and hard-negative texts.

## Implementation spec (current)
See `synthetic_data/synthetic_data_pipeline_plan.docx` (and `.txt`) for the end-to-end design. The repo implementation follows that document and provides a Colab/Drive-oriented setup in `synthetic_data/colab/`.
