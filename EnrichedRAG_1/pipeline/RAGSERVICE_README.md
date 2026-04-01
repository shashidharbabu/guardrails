# RAG Service

This file documents `rag_service.py`, the thin wrapper that exposes the RAG pipeline to MAD or any other downstream agent workflow.

## What `rag_service.py` does

`rag_service.py` is not a new retriever. It is a wrapper around the existing retrieval stack in:

- `retrieval.py`

Its job is to give MAD one clean entrypoint instead of forcing MAD to manage:
- dense retrieval setup
- BM25 loading
- reranking
- path wiring
- evidence formatting
- fallback behavior

The main callable is:

```python
retrieve_for_mad(question: str, top_k: int = 5, verbose: bool = False) -> dict
```

It returns:
- `query`
- `mode`
- `use_hyde`
- `evidence`
- `insufficient_evidence`
- `chunks`
- `mad_evidence_prompt`

## How it uses the retrieval pipeline

Under the hood, `rag_service.py` imports `retrieval.py` and uses the existing `RetrievalPipeline`.

Preferred path:
- `hybrid_no_hyde`

That means it uses:
1. dense search
2. sparse BM25 search
3. Reciprocal Rank Fusion
4. BGE reranker
5. tier boost
6. weak-evidence assessment

If dense artifacts are not available, it falls back to:
- `sparse_only_fallback`

So the wrapper keeps the same retrieval logic, but exposes it in a MAD-friendly shape.

## What files are required

To run the full hybrid path correctly, the service needs:

1. Enriched chunk corpus
- `data/rechunked_output/enriched_chunks.json`

2. BM25 artifact
- `indexed_data/bm25_index.pkl`

3. Local Qdrant collection / dense storage
- a valid local dense index directory or Qdrant collection

In this export, the GitHub-safe folder includes:
- `indexed_data/bm25_index.pkl`

But it does **not** include the full local Qdrant dense store because those files were too large for GitHub.

That means:
- full hybrid retrieval requires regenerating or supplying the dense index locally
- otherwise the wrapper will fall back to sparse-only mode

## What is needed for full hybrid retrieval

For the best validated mode (`hybrid_no_hyde`), you need all of these:

- `enriched_chunks.json`
- `bm25_index.pkl`
- local Qdrant collection or local Qdrant disk artifacts
- embedding model support for `Qwen/Qwen3-Embedding-4B`
- reranker model support for `BAAI/bge-reranker-v2-m3`

Practical requirement:
- your Python environment must support the Qwen3 embedding model
- your environment must have:
  - `torch`
  - `transformers`
  - `qdrant-client`
  - `rank-bm25`

## Why the fallback exists

The export was designed to be GitHub-pushable.

Large dense-index artifacts were excluded from the repo:
- `embeddings.npy`
- local SQLite/Qdrant storage file

So `rag_service.py` was designed to fail gracefully:
- if full hybrid artifacts exist, use hybrid retrieval
- if not, use sparse-only fallback

This makes the exported pipeline usable in more environments without breaking import-time execution.

## Expected runtime behavior

On a properly provisioned local or GPU machine:
- `rag_service.py` should initialize the retrieval stack
- run `hybrid_no_hyde`
- return top retrieved chunks plus a ready-made evidence prompt

On a lighter or incomplete environment:
- it may fall back to sparse-only mode

## How MAD should use it

MAD should not import `retrieval.py` directly.

MAD should import:
- `rag_service.py`

And call:

```python
from rag_service import retrieve_for_mad

result = retrieve_for_mad(
    "What rights do data subjects have under GDPR?",
    top_k=5,
    verbose=True,
)
```

Then MAD should use:
- `result["chunks"]` for structured retrieved evidence
- `result["mad_evidence_prompt"]` as the grounded evidence block for agent reasoning

This keeps MAD independent from retrieval internals.

## Why this is the right interface for MAD

MAD is a downstream reasoning layer.

It should receive:
- grounded evidence
- source citations
- evidence sufficiency signal

It should **not** need to know:
- how dense search works
- how BM25 works
- how reranking works
- how Qdrant is wired

That separation makes the architecture cleaner and easier to maintain.

## Output structure

The returned `chunks` list contains fields like:
- `rank`
- `chunk_id`
- `rerank_score`
- `rrf_score`
- `sources`
- `tier`
- `doc_name`
- `article_number`
- `text`
- `enriched_text`
- `context_prefix`
- `source_url`
- `citation`

The returned `mad_evidence_prompt` includes:
- role instructions
- evidence-use rules
- citation expectations
- few-shot examples
- the retrieved evidence itself

## Current best mode

From the evaluation work done on this corpus, the best validated mode is:

- `hybrid_no_hyde`

Why:
- it outperformed `hybrid_hyde`
- it outperformed `dense_only`
- it outperformed `sparse_only`

Pilot ablation results:

| Variant | Overall |
| --- | ---: |
| hybrid_no_hyde | 0.9220 |
| hybrid_hyde | 0.8727 |
| sparse_only | 0.8510 |
| dense_only | 0.7517 |

So MAD should use this wrapper in `hybrid_no_hyde` mode whenever the full artifacts are available.

## If you want full portability

If you want `rag_service.py` to always run hybrid retrieval from the exported folder alone, you must also package:
- a local Qdrant-compatible dense store

Without that, the wrapper can still work, but only via sparse-only fallback.
