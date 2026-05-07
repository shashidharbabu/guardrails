"""
rag/eval/evaluate.py — RAGAS evaluation harness for the RAG pipeline.
======================================================================

Evaluates the full RAG pipeline (Qdrant + SLM verifier) using RAGAS metrics
with Ollama as the judge LLM and embedding model. No OpenAI/Anthropic API key
is required.

METRICS (all reference-free — no ground-truth answers needed)
-------------------------------------------------------------
  ContextPrecision   — are the retrieved chunks relevant to the query?
  Faithfulness       — does the generated answer stay grounded in the chunks?
  AnswerRelevancy    — is the answer relevant to the question?

HOW IT WORKS
------------
For each test query in rag/eval/queries.py:
  1. run_rag_pipeline(query)       → top_chunks (verified by SLM)
  2. _generate_answer(query, chunks) → LLM answer using Ollama qwen2.5:7b
  3. Build RAGAS SingleTurnSample   → {user_input, response, retrieved_contexts}
  4. Score with RAGAS evaluate()

OUTPUTS
-------
  rag/eval/results/eval_results.json   — per-query scores + aggregate
  rag/eval/results/eval_results.csv    — tabular format for paper tables

USAGE
-----
Run from repo root (rag_folder/ must be on PYTHONPATH):

    cd /path/to/guardrails-enterprise
    export PYTHONPATH="$PYTHONPATH:rag_folder"
    export QDRANT_API_KEY="your-key"
    python -m rag.eval.evaluate

Optional flags:
    --queries hipaa,gdpr        # run only named domains (comma-separated)
    --top-k 7                   # qdrant candidate count (default: 7)
    --no-verify                 # skip SLM verifier, use raw Qdrant results
    --dry-run                   # print dataset rows without running RAGAS

DEPENDENCIES
------------
    pip install ragas langchain-ollama langchain-core datasets
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# ── RAGAS imports ──────────────────────────────────────────────────────────────
try:
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    # Use the non-deprecated collections import path (ragas >= 0.2)
    try:
        from ragas.metrics.collections import (
            ContextPrecision,
            Faithfulness,
            AnswerRelevancy,
        )
    except ImportError:
        # Fallback for older ragas installs
        from ragas.metrics import (  # type: ignore[no-redef]
            ContextPrecision,
            Faithfulness,
            AnswerRelevancy,
        )
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from datasets import Dataset
    _RAGAS_AVAILABLE = True
except ImportError as _ragas_err:
    _RAGAS_AVAILABLE = False
    _RAGAS_IMPORT_ERROR = str(_ragas_err)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rag.eval")

# ── Paths ──────────────────────────────────────────────────────────────────────
_EVAL_DIR    = Path(__file__).resolve().parent
_RESULTS_DIR = _EVAL_DIR / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Public entry point ─────────────────────────────────────────────────────────

def run_evaluation(
    domain_filter: Optional[list[str]] = None,
    top_k: int = 7,
    use_verifier: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Run the full RAGAS evaluation pipeline.

    Args:
        domain_filter:  If set, only evaluate queries from these domains.
        top_k:          Qdrant candidate count before SLM filtering.
        use_verifier:   If False, skip SLM verifier (raw Qdrant retrieval).
        dry_run:        Print dataset rows without calling RAGAS.

    Returns:
        Result dict with per-query scores and aggregate metrics.
    """
    if not _RAGAS_AVAILABLE:
        raise ImportError(
            f"RAGAS / langchain-ollama not installed. Run:\n"
            f"  pip install ragas langchain-ollama langchain-core datasets\n"
            f"Original error: {_RAGAS_IMPORT_ERROR}"
        )

    from rag.eval.queries import get_queries

    queries = get_queries()
    if domain_filter:
        dl = [d.lower() for d in domain_filter]
        queries = [q for q in queries if any(q["domain"].lower().startswith(d) for d in dl)]

    if not queries:
        raise ValueError(f"No queries found for domain filter: {domain_filter}")

    logger.info("Evaluating %d queries (verifier=%s, top_k=%d)", len(queries), use_verifier, top_k)

    # ── Build dataset ──────────────────────────────────────────────────────────
    rows: list[dict] = []
    per_query_meta: list[dict] = []

    for i, q in enumerate(queries, 1):
        logger.info("[%d/%d] %s — %s", i, len(queries), q["id"], q["query"][:80])
        t0 = time.time()

        try:
            if use_verifier:
                # Prefer full pipeline; fall back gracefully to TF-IDF when
                # Qdrant / embedder is unreachable (e.g. no QDRANT_API_KEY set).
                try:
                    from rag.pipeline import run_rag_pipeline
                    rag_result = run_rag_pipeline(q["query"], top_k=top_k)
                    top_chunks = rag_result["verified_output"].get("top_chunks", [])
                    contexts   = [c["text"] for c in top_chunks if c.get("text")]
                    confidence = rag_result["verified_output"].get("confidence", 0.0)
                    sufficient = rag_result["verified_output"].get("sufficient_context", False)
                    if not contexts:
                        # Verifier rejected all → try raw candidates
                        contexts = [
                            c["text"] for c in rag_result.get("retrieved_candidates", [])
                            if c.get("text")
                        ]
                except Exception as qdrant_exc:
                    logger.warning(
                        "  Qdrant pipeline failed (%s) — using TF-IDF fallback for %s",
                        qdrant_exc, q["id"],
                    )
                    from rag.pipeline import _tfidf_fallback
                    chunks    = _tfidf_fallback(q["query"], top_k)
                    contexts  = [c.text for c in chunks]
                    confidence = None
                    sufficient = None
            else:
                # --no-verify: direct retriever path (also falls back via retrieve_verified)
                try:
                    from rag.retriever import retrieve
                    chunks = retrieve(q["query"], top_k=top_k)
                except Exception:
                    from rag.pipeline import _tfidf_fallback
                    chunks = _tfidf_fallback(q["query"], top_k)
                contexts   = [c.text for c in chunks]
                confidence = None
                sufficient = None

            if not contexts:
                logger.warning("  No context retrieved — skipping query %s", q["id"])
                continue

            answer = _generate_answer(q["query"], contexts)
            elapsed = round(time.time() - t0, 2)

            rows.append({
                "user_input":          q["query"],
                "retrieved_contexts":  contexts,
                "response":            answer,
            })

            per_query_meta.append({
                "id":         q["id"],
                "domain":     q["domain"],
                "query":      q["query"],
                "n_contexts": len(contexts),
                "confidence": confidence,
                "sufficient": sufficient,
                "answer":     answer,
                "elapsed_s":  elapsed,
            })

            logger.info(
                "  contexts=%d  confidence=%s  elapsed=%.1fs",
                len(contexts), confidence, elapsed
            )

        except Exception as exc:
            logger.error("  FAILED %s: %s", q["id"], exc, exc_info=True)

    if not rows:
        raise RuntimeError(
            "No rows collected — every query failed or returned empty context.\n"
            "Quick fixes:\n"
            "  1. Set QDRANT_API_KEY in your .env file (Qdrant Cloud auth)\n"
            "  2. Set HF_TOKEN in your .env file (faster HuggingFace downloads)\n"
            "  3. Ensure Ollama is running: `ollama serve`\n"
            "  4. Run with TF-IDF only (no embedder/Qdrant): --no-verify\n"
            "     PYTHONPATH='rag_folder:multi_agent_debate' python -m rag.eval.evaluate --no-verify"
        )

    if dry_run:
        logger.info("DRY RUN — dataset collected, skipping RAGAS scoring.")
        _save_dry_run(per_query_meta)
        return {"dry_run": True, "rows_collected": len(rows)}

    # ── RAGAS scoring ──────────────────────────────────────────────────────────
    logger.info("Running RAGAS evaluation on %d samples …", len(rows))

    from rag.config import OLLAMA_BASE_URL, VERIFIER_MODEL

    ollama_base = OLLAMA_BASE_URL.rstrip("/v1").rstrip("/")
    judge_llm   = LangchainLLMWrapper(ChatOllama(
        model=VERIFIER_MODEL,
        base_url=ollama_base,
        temperature=0,
    ))
    eval_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(
        model=VERIFIER_MODEL,
        base_url=ollama_base,
    ))

    metrics = [
        ContextPrecision(llm=judge_llm),
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=eval_emb),
    ]

    dataset = Dataset.from_list(rows)
    ragas_result = evaluate(dataset=dataset, metrics=metrics)
    scores_df    = ragas_result.to_pandas()

    # ── Merge per-query metadata with RAGAS scores ─────────────────────────────
    full_results: list[dict] = []
    for i, meta in enumerate(per_query_meta):
        if i < len(scores_df):
            row_scores = scores_df.iloc[i].to_dict()
        else:
            row_scores = {}
        full_results.append({**meta, **row_scores})

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    metric_cols = ["context_precision", "faithfulness", "answer_relevancy"]
    aggregate: dict[str, float] = {}
    for col in metric_cols:
        vals = [r[col] for r in full_results if col in r and r[col] is not None]
        aggregate[col] = round(sum(vals) / len(vals), 4) if vals else None

    result = {
        "run_at":    datetime.now(timezone.utc).isoformat(),
        "n_queries": len(full_results),
        "top_k":     top_k,
        "verifier":  use_verifier,
        "model":     VERIFIER_MODEL,
        "aggregate": aggregate,
        "per_query": full_results,
    }

    _save_results(result)
    _print_summary(result)
    return result


# ── Answer generation ──────────────────────────────────────────────────────────

def _generate_answer(query: str, contexts: list[str]) -> str:
    """
    Generate an answer from retrieved contexts using Ollama (qwen2.5:7b).

    Builds a grounded RAG prompt: system + context + question.
    This is the answer RAGAS will evaluate for Faithfulness and AnswerRelevancy.
    """
    from rag.config import OLLAMA_BASE_URL, VERIFIER_MODEL

    context_block = "\n\n---\n\n".join(
        f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )

    system = (
        "You are an enterprise compliance assistant. "
        "Answer the question using ONLY the provided regulatory context. "
        "If the context does not contain enough information to answer, say so explicitly. "
        "Be precise and cite the relevant regulation when possible. "
        "Keep your answer to 3-5 sentences."
    )
    user = (
        f"Regulatory context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer based on the context above:"
    )

    url = OLLAMA_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model":       VERIFIER_MODEL,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.0,
        "stream":      False,
        "max_tokens":  512,
    }

    with httpx.Client(timeout=120) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


# ── Output helpers ─────────────────────────────────────────────────────────────

def _save_results(result: dict) -> None:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = _RESULTS_DIR / f"eval_results_{ts}.json"
    csv_path  = _RESULTS_DIR / f"eval_results_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info("JSON results → %s", json_path)

    if result.get("per_query"):
        _write_csv(result["per_query"], csv_path)
        logger.info("CSV  results → %s", csv_path)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(v) if v is not None else "" for k, v in row.items()})


def _save_dry_run(meta: list[dict]) -> None:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _RESULTS_DIR / f"dry_run_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Dry-run data → %s", path)


def _print_summary(result: dict) -> None:
    agg = result.get("aggregate", {})
    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Queries evaluated : {result['n_queries']}")
    print(f"  Qdrant top-k      : {result['top_k']}")
    print(f"  SLM verifier      : {result['verifier']}")
    print(f"  Judge model       : {result['model']}")
    print()
    print(f"  Context Precision : {agg.get('context_precision', 'N/A')}")
    print(f"  Faithfulness      : {agg.get('faithfulness', 'N/A')}")
    print(f"  Answer Relevancy  : {agg.get('answer_relevancy', 'N/A')}")
    print("=" * 60)
    print(f"  Results saved to  : {_RESULTS_DIR}/")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline with RAGAS + Ollama."
    )
    p.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Comma-separated domain names to filter (e.g. 'hipaa,gdpr'). Default: all.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=7,
        help="Number of Qdrant candidates before SLM filtering (default: 7).",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SLM verifier — evaluate raw Qdrant retrieval.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect dataset rows and save them without running RAGAS scoring.",
    )
    return p.parse_args()


if __name__ == "__main__":
    # Ensure rag_folder is on PYTHONPATH when run as a module
    _repo_root = Path(__file__).resolve().parent.parent.parent.parent
    _rag_folder = _repo_root / "rag_folder"
    if str(_rag_folder) not in sys.path:
        sys.path.insert(0, str(_rag_folder))

    # Load .env from repo root
    try:
        from dotenv import load_dotenv
        load_dotenv(_repo_root / ".env")
    except ImportError:
        pass

    args = _parse_args()
    domains = [d.strip() for d in args.queries.split(",")] if args.queries else None

    run_evaluation(
        domain_filter=domains,
        top_k=args.top_k,
        use_verifier=not args.no_verify,
        dry_run=args.dry_run,
    )
