# EnrichedRAG_1

A compact export of the old 39-document regulatory/healthcare RAG pipeline.

This folder contains the pipeline that was used for the original corpus only. It excludes the later new-document chunking and enrichment flow.

## Corpus

The corpus contains about 39 source documents across regulatory, policy, security, legal AI, and healthcare guidance.

The core artifacts are:
- raw re-chunked corpus: `data/rechunked_output/chunks.json`
- enriched corpus: `data/rechunked_output/enriched_chunks.json`
- source JSONL files used to reconstruct/rechunk the docs: `data/chunks_with_links/`

## Why We Rechunked

The original chunks had structural and sentence-boundary problems common in regulatory corpora:
- many chunks started or ended mid-sentence
- some chunks were too small to be useful
- some chunks were too large and mixed multiple ideas
- table-of-contents / junk-like chunks were still present

We rechunked to produce more retrieval-friendly units:
- sentence-aware chunking with spaCy
- structure-aware heading handling
- target chunk sizes with overlap
- removal of very small/junk chunks
- preservation of source metadata like tier and links

The old-pipeline rechunker used here is:
- `pipeline/rechunk.py`

This is the valid old-pipeline rechunker because it writes:
- `rechunked_output/chunks.json`
- `rechunked_output/rechunk_report.txt`

`rechunk_v2.py` was intentionally excluded because it belongs to the later new-document flow and writes `new_chunks.jsonl`.

## Why We Enriched

Dense retrieval on regulatory text often struggles because many clauses are semantically incomplete in isolation.

We enriched each chunk by prepending a short context prefix before embedding. The context prefix was generated using MiniMax M2.7 through OpenRouter. The goal was to make each chunk more self-identifying and improve dense retrieval quality for fragmented legal/policy text.

The old-pipeline enrichment script used here is:
- `pipeline/enrich_chunks.py`

The enriched output used for indexing and retrieval is:
- `data/rechunked_output/enriched_chunks.json`

## Indexing

Indexing used a dual-index setup.

Dense index:
- model: `Qwen/Qwen3-Embedding-4B`
- vectors stored in local Qdrant-compatible artifacts
- embeds `enriched_text`

Sparse index:
- BM25 via `rank_bm25`
- indexes raw `text`

Why both:
- dense search helps semantic retrieval
- sparse search helps exact legal terms, article numbers, named regulations, and lexical matches

Indexing script:
- `pipeline/indexing.py`

Indexed artifacts included here:
- `indexed_data/bm25_index.pkl`
- `indexed_data/meta (1).json`

Excluded from GitHub due to repository file-size limits:
- `indexed_data/embeddings.npy`
- `indexed_data/storage (1).sqlite`

Those large dense-index artifacts can be regenerated with `pipeline/indexing.py`.

## Retrieval

Retrieval uses a hybrid pipeline:
1. dense search in Qdrant
2. sparse BM25 search
3. Reciprocal Rank Fusion (RRF)
4. BGE reranker (`BAAI/bge-reranker-v2-m3`)
5. tier boost for highly authoritative material
6. weak-evidence check
7. top-5 final chunks with citations

Retrieval script:
- `pipeline/retrieval.py`

HyDE support exists in the retrieval code, but it was not selected as the production default after evaluation.

## Pilot Eval Set (15 Questions)

The early pilot eval set was generated locally using Ollama and then used to compare retrieval variants.

Eval generation approach for the pilot:
- grounded question + reference-answer generation from chunk text and metadata
- local model used first: `llama3.1:8b`
- noisy items were filtered out manually and programmatically

Pilot ablation results on 15 questions:

| Variant | Overall |
| --- | ---: |
| hybrid_no_hyde | 0.9220 |
| hybrid_hyde | 0.8727 |
| sparse_only | 0.8510 |
| dense_only | 0.7517 |

Why `nohyde` won:
- HyDE slightly helped in a few cases
- but it reduced relevance/completeness overall on this corpus
- hybrid retrieval without HyDE was the most reliable balance

Final production choice:
- `hybrid_no_hyde`

## Final Evaluation

Final eval set used:
- `evals/evalset_local_85_curated.jsonl`

Custom evaluation used:
- answer generation: `MiniMax M2.7`
- judge: `Claude Sonnet` via OpenRouter
- metrics:
  - faithfulness
  - answer relevance
  - citation grounding
  - completeness
  - overall
  - insufficient-evidence flag

RAGAS was then run on the saved custom-eval outputs.

### Final Results

Custom eval summary:
- most questions scored in the `0.90–1.00` range
- a smaller number of weaker cases were in the high `0.70s` to mid `0.80s`
- insufficient evidence was rare on the curated 85-question set

RAGAS summary:

| Metric | Score |
| --- | ---: |
| faithfulness | 0.9494 |
| answer_relevancy | 0.8254 |
| context_precision | 0.9681 |
| context_recall | 0.9608 |

Interpretation:
- retrieval coverage is strong
- grounding is strong
- the main remaining improvement area is answer directness/completeness, not retrieval integrity

## Evaluation Files Included

Custom eval outputs:
- `evals/full_eval_85_results.json`
- `evals/full_eval_85_results_with_ragas.json`

Eval scripts:
- `pipeline/evals.py`
- `pipeline/evals_ragas_only.py`

## Notes Included

Documentation and analysis notes:
- `notes/chunking_journey.txt`
- `notes/chunk_quality_report.txt`
- `notes/chunking_strategy_report.txt`
- `notes/final_eval_summary.txt`
- `notes/rag_pipeline_handoff.txt`

## Folder Layout

- `pipeline/` — scripts used for the old 39-doc pipeline
- `data/` — re-chunked and enriched corpus plus source chunk JSONLs
- `indexed_data/` — dense/sparse indexing artifacts
- `evals/` — curated eval set and saved evaluation outputs
- `notes/` — reports, handoff docs, and journey notes
