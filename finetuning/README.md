# Finetuning Pipelines

Fine-tuning scripts for all 4 models used in the Guardrails Gateway.

## Status
🚧 In Development

## The 4 Fine-tuned Models

| # | Model | Task | Key Metrics |
|---|-------|------|-------------|
| 1 | RoBERTa-base | Jailbreak detection (Gateway) | F1: 0.9821, OOD evaluated |
| 2 | RoBERTa-base NER | PII detection — 57 entity types (Gateway) | 98.2% recall |
| 3 | Llama-Prompt-Guard-2-86M | Prompt injection detection (Gateway) | F1: 0.9821, 1.2% missed attack rate |
| 4 | Qwen2.5-3B-Instruct | LLM generator (benchmarking) | Base eval: ROUGE-L 0.2634 |

**RAG embedding model (separate):** Qwen3-4B fine-tuned with LoRA (r=16, α=32, all layers, 3 epochs, 2e-4 lr) on 13,572 query-chunk pairs. Scripts live in `rag/`.

## Training datasets

| Model | Dataset(s) |
|-------|-----------|
| RoBERTa NER | `ai4privacy/pii-masking-200k` (processed by Airflow DAG → GCS) |
| RoBERTa Jailbreak | JailBreakV-28k + Databricks Dolly-15k |
| Llama-Prompt-Guard | 3 combined prompt injection datasets |
| Qwen2.5-3B | Instruction tuning for RAG tasks |
| Qwen3-4B (RAG) | 13,572 query-chunk pairs from regulatory corpus |

## Known model limitations

- **RoBERTa Jailbreak**: weak on fictional framing and indirect multi-step attacks
- **Llama-Prompt-Guard**: `on_fail="noop"` is intentional — the Gateway's DecisionEngine, not guardrails-ai, makes the final block/pass call

## Planned Components
- Training scripts for each of the 4 models
- Evaluation harnesses (F1, recall, ROUGE-L)
- Model checkpointing and versioning
- Hyperparameter configuration files
- Integration tests against Gateway and MAD pipeline
