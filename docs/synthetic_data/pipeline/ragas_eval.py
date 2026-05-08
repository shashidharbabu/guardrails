from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics._answer_relevance import AnswerRelevancy
from langchain_community.embeddings import HuggingFaceEmbeddings

from .io import atomic_write_text, write_jsonl


def _mean_ignore_nan(vals: Iterable[Any]) -> float | None:
    nums: List[float] = []
    for v in vals:
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv):
            continue
        nums.append(fv)
    if not nums:
        return None
    return sum(nums) / len(nums)


def _prepare_ragas_primitives() -> Tuple[Any, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY for RAGAS evaluation.")

    # Use the same Claude family for eval prompts + local HF embeddings.
    import anthropic  # local import to keep import side effects minimal

    llm = llm_factory(
        "claude-sonnet-4-20250514",
        provider="anthropic",
        client=anthropic.Anthropic(api_key=api_key),
    )
    # Use a LangChain embedding implementation wrapped for ragas legacy metrics.
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    return llm, embeddings


def _build_embedding_dataset(triplets: List[dict], sample_limit: int | None) -> Tuple[Dataset, List[dict]]:
    rows = triplets if sample_limit is None else triplets[: max(0, int(sample_limit))]
    user_input = [str(r.get("query", "") or "") for r in rows]
    response = [str((r.get("positive", {}) or {}).get("text", "") or "")[:1200] for r in rows]
    retrieved_contexts = [[str((r.get("positive", {}) or {}).get("text", "") or "")[:1200]] for r in rows]
    ds = Dataset.from_dict(
        {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        }
    )
    return ds, rows


def _build_llm_dataset(pairs: List[dict], sample_limit: int | None) -> Tuple[Dataset, List[dict]]:
    rows = pairs if sample_limit is None else pairs[: max(0, int(sample_limit))]
    user_input = [str((r.get("instruction", {}) or {}).get("user_query", "") or "") for r in rows]
    response = [json.dumps(r.get("response", {}), ensure_ascii=False)[:1200] for r in rows]
    retrieved_contexts = [
        [
            str(c.get("text", "") or "")[:1200]
            for c in ((r.get("instruction", {}) or {}).get("retrieved_context") or [])
            if isinstance(c, dict)
        ]
        for r in rows
    ]
    ds = Dataset.from_dict(
        {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        }
    )
    return ds, rows


def _persist_eval_outputs(*, eval_dir: str, rows: List[dict], summary: dict) -> None:
    os.makedirs(eval_dir, exist_ok=True)
    write_jsonl(os.path.join(eval_dir, "ragas_metrics.jsonl"), rows)
    atomic_write_text(os.path.join(eval_dir, "ragas_summary.json"), json.dumps(summary, indent=2))


def evaluate_embedding_with_ragas(
    *,
    triplets_for_eval: List[dict],
    eval_dir: str,
    sample_limit: int | None = None,
    debug: bool = False,
) -> Dict[str, Any]:
    def dbg(msg: str) -> None:
        if debug:
            print(f"[DEBUG][ragas][embedding] {msg}")

    if not triplets_for_eval:
        summary = {"status": "skipped", "reason": "no_samples", "num_samples": 0}
        _persist_eval_outputs(eval_dir=eval_dir, rows=[], summary=summary)
        return summary

    llm, embeddings = _prepare_ragas_primitives()
    ds, source_rows = _build_embedding_dataset(triplets_for_eval, sample_limit)
    metrics = [
        AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
    ]
    dbg(f"running evaluate() samples={len(source_rows)} metrics={[m.name for m in metrics]}")
    result = evaluate(ds, metrics=metrics, raise_exceptions=False, show_progress=debug)
    scores = result.scores if hasattr(result, "scores") else []

    out_rows: List[dict] = []
    for src, sc in zip(source_rows, scores):
        out_rows.append(
            {
                "id": src.get("id"),
                "query": src.get("query"),
                "metric_scores": sc,
            }
        )

    metrics_found: List[str] = sorted({k for sc in scores for k in sc.keys()}) if scores else []
    metric_means = {m: _mean_ignore_nan(sc.get(m) for sc in scores) for m in metrics_found}
    summary = {
        "status": "ok",
        "dataset": "embedding",
        "num_samples": len(source_rows),
        "metrics": metrics_found,
        "metric_means": metric_means,
    }
    _persist_eval_outputs(eval_dir=eval_dir, rows=out_rows, summary=summary)
    return summary


def evaluate_llm_with_ragas(
    *,
    pairs_for_eval: List[dict],
    eval_dir: str,
    sample_limit: int | None = None,
    debug: bool = False,
) -> Dict[str, Any]:
    def dbg(msg: str) -> None:
        if debug:
            print(f"[DEBUG][ragas][llm] {msg}")

    if not pairs_for_eval:
        summary = {"status": "skipped", "reason": "no_samples", "num_samples": 0}
        _persist_eval_outputs(eval_dir=eval_dir, rows=[], summary=summary)
        return summary

    llm, embeddings = _prepare_ragas_primitives()
    ds, source_rows = _build_llm_dataset(pairs_for_eval, sample_limit)
    metrics = [
        AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
    ]
    dbg(f"running evaluate() samples={len(source_rows)} metrics={[m.name for m in metrics]}")
    result = evaluate(ds, metrics=metrics, raise_exceptions=False, show_progress=debug)
    scores = result.scores if hasattr(result, "scores") else []

    out_rows: List[dict] = []
    for src, sc in zip(source_rows, scores):
        out_rows.append(
            {
                "id": src.get("id"),
                "user_query": (src.get("instruction", {}) or {}).get("user_query"),
                "metric_scores": sc,
            }
        )

    metrics_found: List[str] = sorted({k for sc in scores for k in sc.keys()}) if scores else []
    metric_means = {m: _mean_ignore_nan(sc.get(m) for sc in scores) for m in metrics_found}
    summary = {
        "status": "ok",
        "dataset": "llm",
        "num_samples": len(source_rows),
        "metrics": metrics_found,
        "metric_means": metric_means,
    }
    _persist_eval_outputs(eval_dir=eval_dir, rows=out_rows, summary=summary)
    return summary

