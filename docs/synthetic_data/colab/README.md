# Synthetic Data Generation (Colab/Drive)

This folder mirrors the **exact execution model** described in `synthetic_data_pipeline_plan.docx`: run a sequence of Colab notebooks that read **pre-chunked JSONL** files and produce two synthetic datasets:

- **Pipeline 1 (Embeddings)**: `(query, positive_chunk, hard_negative)` triplets
- **Pipeline 2 (Safety agent)**: `(instruction, retrieved_context, response)` pairs

## Drive layout (expected by `config.py`)

Create this folder in Google Drive:

`MyDrive/guardrails-synthetic-data/`

Then inside it:

- `config.py` (copy from this folder)
- `requirements.txt` (optional; notebooks pip install directly)
- `data/chunks/` (drop the 22 JSONL chunk files here)
- `data/synthetic/` (pipeline checkpoints + outputs)
- `data/exports/` (final train/val/test jsonl files)

## Secrets

Set `ANTHROPIC_API_KEY` in **Colab Secrets** (do not hardcode keys in notebooks).

## Run order

1. `01_filter_chunks.ipynb`
2. `02_metadata_enrichment.ipynb`
3. `03_embedding_triplets.ipynb`
4. `04_agent_pairs.ipynb`
5. `05_validate_export.ipynb`

