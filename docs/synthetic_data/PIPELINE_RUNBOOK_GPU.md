# Synthetic Data Pipelines Runbook (GPU Lab / Colab)

This guide explains how to run both synthetic-data pipelines at larger scale:
- Pipeline 1: embedding triplets
- Pipeline 2: LLM instruction-response pairs

It also includes post-run validation gates and troubleshooting.

## 1) Prerequisites

- Python 3.10+ environment
- Repo checked out locally
- Input chunks available as JSONL files in one folder
- Anthropic API key

Set API key:

```bash
export ANTHROPIC_API_KEY="your_key_here"
```

Install dependencies (if needed):

```bash
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org anthropic jsonlines tqdm
```

Install RAGAS monitoring dependencies:

```bash
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ragas langchain-anthropic sentence-transformers
```

Run all commands from repo root:

```bash
cd /path/to/guardrails-enterprise
```

## 2) Input Data Expectations

- `chunks_dir` must contain `*.jsonl` files (one JSON object per line).
- Expected chunk fields include: `chunk_id`, `doc_id`, `text`, `tier`, `token_count`.
- More document diversity improves hard-negative quality for embedding triplets.

Example:

```text
rag/chunks/
  t0__...chunks.jsonl
  t1__...chunks.jsonl
  ...
```

## 3) Pipeline 1: Embedding Triplets

## Recommended starter command (larger run)

```bash
python -m synthetic_data.run_embedding_pipeline \
  --chunks-dir rag/chunks \
  --triplets 2000 \
  --enrich-limit 1200 \
  --retry-limit 2 \
  --with-ragas \
  --out-dir synthetic_data/_embedding_run_prod \
  --debug
```

## Outputs

- Raw triplets: `synthetic_data/_embedding_run_prod/synthetic/embedding/embedding_triplets_raw.jsonl`
- Validated triplets: `synthetic_data/_embedding_run_prod/synthetic/embedding/embedding_triplets_validated.jsonl`
- Rejected triplets: `synthetic_data/_embedding_run_prod/synthetic/embedding/rejected/embedding_triplets_rejected.jsonl`
- RAGAS per-sample metrics: `synthetic_data/_embedding_run_prod/synthetic/embedding/eval/ragas_metrics.jsonl`
- RAGAS summary: `synthetic_data/_embedding_run_prod/synthetic/embedding/eval/ragas_summary.json`
- Shared preprocessing artifacts:
  - `filtered_chunks.jsonl`
  - `enriched_chunks.jsonl`
  - `rejected/rejected_chunks.jsonl`

## 4) Pipeline 2: LLM Synthetic Pairs

## Recommended starter command (larger run)

```bash
python -m synthetic_data.run_llm_pipeline \
  --chunks-dir rag/chunks \
  --pairs 2000 \
  --enrich-limit 1200 \
  --retry-limit 2 \
  --label-dist '{"BLOCK": 0.5, "ALLOW": 0.3, "ESCALATE": 0.2}' \
  --with-ragas \
  --out-dir synthetic_data/_llm_run_prod \
  --debug
```

## Outputs

- Raw pairs: `synthetic_data/_llm_run_prod/synthetic/llm/agent_pairs_raw.jsonl`
- Validated pairs: `synthetic_data/_llm_run_prod/synthetic/llm/agent_pairs_validated.jsonl`
- Rejected pairs: `synthetic_data/_llm_run_prod/synthetic/llm/rejected/agent_pairs_rejected.jsonl`
- RAGAS per-sample metrics: `synthetic_data/_llm_run_prod/synthetic/llm/eval/ragas_metrics.jsonl`
- RAGAS summary: `synthetic_data/_llm_run_prod/synthetic/llm/eval/ragas_summary.json`
- Shared preprocessing artifacts:
  - `filtered_chunks.jsonl`
  - `enriched_chunks.jsonl`
  - `rejected/rejected_chunks.jsonl`

## Note on schema

LLM pipeline keeps:
- `instruction.retrieved_context`

LLM pipeline does not require:
- `response.reasoning`

## 5) Post-Run Quality Gates (Required)

Use these checks before accepting a run.

## Embedding gate

- Gate E1: validated/raw ratio >= `0.70`
- Gate E2: top reject reason is not dominated by one systemic issue (>80%)
- Gate E3: sampled validated records include realistic query-to-positive relevance
- Gate E4: validated records include `validation.semantic_similarity`
- Gate E5: RAGAS `answer_relevancy` is non-null for >=80% evaluated samples

## LLM gate

- Gate L1: validated/raw ratio >= `0.60`
- Gate L2: no single reject reason dominates >80%
- Gate L3: sampled responses have coherent `decision`, `threat_category`, and actionability
- Gate L4: policy citations align with validator expectations
- Gate L5: RAGAS `answer_relevancy` is populated for >=80% evaluated samples

## Quick count script

```bash
python - <<'PY'
import jsonlines, collections, os
base="synthetic_data"
paths={
  "emb_raw": f"{base}/_embedding_run_prod/synthetic/embedding/embedding_triplets_raw.jsonl",
  "emb_val": f"{base}/_embedding_run_prod/synthetic/embedding/embedding_triplets_validated.jsonl",
  "emb_rej": f"{base}/_embedding_run_prod/synthetic/embedding/rejected/embedding_triplets_rejected.jsonl",
  "llm_raw": f"{base}/_llm_run_prod/synthetic/llm/agent_pairs_raw.jsonl",
  "llm_val": f"{base}/_llm_run_prod/synthetic/llm/agent_pairs_validated.jsonl",
  "llm_rej": f"{base}/_llm_run_prod/synthetic/llm/rejected/agent_pairs_rejected.jsonl",
}
for k,p in paths.items():
    if not os.path.exists(p):
        print(k, "MISSING")
        continue
    with jsonlines.open(p) as r:
        rows=list(r)
    print(k, len(rows))
    if k.endswith("rej"):
        c=collections.Counter()
        for row in rows:
            for rr in row.get("reject_reasons",[]) or []:
                c[rr]+=1
        print(k+"_reasons", dict(c))
PY
```

## 6) Troubleshooting

## Missing API key

Error:
- `RuntimeError: Missing ANTHROPIC_API_KEY env var.`

Fix:
- export key in same terminal session before running.

## Too few embedding triplets

Symptoms:
- low raw count
- many `insufficient_negative_candidates`

Fixes:
- add more diverse docs to `chunks_dir`
- increase `--enrich-limit`
- keep multiple domains/jurisdictions in input

## LLM all rejected with `policy_citation_not_in_context`

Cause:
- generated `policy` strings not matching expected source-id style in validator.

Fix path:
- enforce `policy` values to include `retrieved_context.source` ids in generation, or
- adjust validator to support normalized policy-name mapping.

## Rate/latency issues on large runs

Fixes:
- reduce `--enrich-limit` and run batches
- lower targets (`--triplets`, `--pairs`) per batch
- use `--retry-limit 1` during dry runs, increase for production runs
