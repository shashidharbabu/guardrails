# FinalMAD with Judge

A clean, shareable Multi-Agent Debate (MAD) pipeline for hallucination detection.

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Set VLLM_AGENTS_URL, VLLM_JUDGE_URL at minimum
```

### 3. Run debate + judge
```bash
python scripts/run_debate_from_inputs.py     --input examples/sample_input.json     --run-id demo-01 --db mad.db --run-judge
```

### 4. Inspect results
```bash
python scripts/inspect_db.py --db mad.db --run-id demo-01
```

### 5. Run judge separately
```bash
python scripts/run_judge_on_db.py --run-id demo-01 --db mad.db
```

### 6. Export GRPO dataset
```bash
python scripts/export_grpo_dataset.py --db mad.db --run-id demo-01 --out-dir grpo_export/
```

## Required Env Vars

| Variable | Required | Default |
|----------|----------|---------|
| VLLM_AGENTS_URL | Yes | http://localhost:8001/v1 |
| VLLM_JUDGE_URL | Yes (judge) | http://localhost:8004/v1 |
| VLLM_DECOMPOSER_URL | If using decomposer | http://localhost:8002/v1 |
| SQLITE_DB_PATH | No | mad.db |
| CLAIM_CONCURRENCY | No | 2 |
| LANGFUSE_PUBLIC_KEY | No | (tracing disabled) |
| LANGFUSE_SECRET_KEY | No | (tracing disabled) |

## What Works
- All Pydantic schemas (AgentOutputFull, JudgeOutput, strip_for_peer)
- All prompts (decomposer, agents A/B, judge) -- exact copies from live run
- SQLite DB schema + CRUD layer (5 tables)
- Agent JSON parsing with fallback repair loop
- Round 0 and Round 1 debate nodes (async, concurrent)
- Decomposer node with coverage check
- Judge node with optional Langfuse tracing
- Full pipeline orchestrator
- CLI scripts: debate, judge, export, inspect
- GRPO export script

## What is TODO / Stubbed
- RAG live retrieval (src/retrieval/rag_client.py): needs Qdrant + BM25
  - Workaround: pass rag_chunks directly in input JSON (recommended)
- GRPO training: see training_reference/GRPO_HANDOFF.md

## Repo Structure
```
configs/config.py          env-based config (vLLM URLs, DB path, concurrency)
src/schemas/schemas.py     Pydantic models
src/utils/prompts.py       All system + user prompts
src/utils/vllm_client.py   vLLM client factories
src/decomposer/node.py     Claim extraction node
src/retrieval/rag_client.py RAG stub (TODO)
src/agents/parser.py       JSON parsing + repair
src/agents/round0.py       Round 0 debate
src/agents/round1.py       Round 1 debate
src/debate/pipeline.py     Full pipeline orchestrator
src/judge/node.py          Judge node + Langfuse
src/db/db.py               SQLite helpers
src/db/schema.sql          DB schema
src/langfuse_log/logger.py Langfuse client factory
scripts/run_debate_from_inputs.py  Main entrypoint
scripts/run_judge_on_db.py         Judge on existing DB
scripts/export_grpo_dataset.py     GRPO export
scripts/inspect_db.py              DB inspector
examples/sample_input.json         GDPR sample input
docs/PIPELINE_OVERVIEW.md
docs/DATABASE_SCHEMA.md
docs/VLLM_SETUP.md
docs/LANGFUSE_SETUP.md
training_reference/GRPO_HANDOFF.md
```
