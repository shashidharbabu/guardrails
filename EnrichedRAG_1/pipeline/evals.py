#!/usr/bin/env python3
"""
evals.py

Full evaluation runner for the curated regulatory RAG eval set.

Default mode:
  - Reads the curated 85-question eval set
  - Runs the production retrieval path: hybrid_no_hyde
  - Uses MiniMax M2.7 via OpenRouter for answer generation
  - Uses Claude Sonnet via OpenRouter for judging
  - Writes detailed JSON and markdown summaries

RAGAS mode:
  - RAGAS is required and runs on top of the same frozen eval run.
  - The OpenRouter judge and RAGAS both execute; neither replaces the other.
"""

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import requests

from retrieval import BM25_PATH, RetrievalPipeline

BASE_DIR = Path(os.environ.get("RAG_BASE_DIR", Path(__file__).parent))
EVALS_DIR = BASE_DIR / "evals_output"
EVALSET_PATH = Path(os.environ.get("EVALSET_PATH", EVALS_DIR / "evalset_local_85_curated.jsonl"))
RESULTS_PATH = Path(os.environ.get("EVAL_RESULTS_PATH", EVALS_DIR / "full_eval_85_results.json"))
SUMMARY_PATH = Path(os.environ.get("EVAL_SUMMARY_PATH", EVALS_DIR / "full_eval_85_summary.md"))
RAGAS_PATH = Path(os.environ.get("EVAL_RAGAS_PATH", EVALS_DIR / "full_eval_85_ragas.json"))
EXISTING_RESULTS_PATH = Path(os.environ.get("EXISTING_RESULTS_PATH", EVALS_DIR / "full_eval_85_results.json"))

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ANSWER_MODEL = os.environ.get("EVAL_ANSWER_MODEL", "minimax/minimax-m2.7")
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "anthropic/claude-sonnet-4")
OPENROUTER_MAX_RETRIES = int(os.environ.get("OPENROUTER_MAX_RETRIES", "3"))
OPENROUTER_TIMEOUT = int(os.environ.get("OPENROUTER_TIMEOUT", "120"))
MAX_ITEMS = int(os.environ.get("EVAL_MAX_ITEMS", "0"))
RAGAS_ONLY = os.environ.get("RAGAS_ONLY", "false").lower() == "true"

ANSWER_SYSTEM_PROMPT = """You are the answer-writing model for a strict regulatory RAG benchmark.

Role:
- Write the best possible answer using only the retrieved sources.
- Behave like a careful compliance analyst, not a creative assistant.

Hard requirements:
- Use only the retrieved evidence. No outside knowledge.
- If the evidence is insufficient, reply with exactly: Insufficient evidence.
- Use inline source references like [Source 1], [Source 2].
- Do not cite a source for a claim it does not support.
- Do not invent article numbers, entities, thresholds, deadlines, or penalties.
- Prefer precise, compact, policy-style wording.

Target answer quality:
- Directly answer the question.
- Include supported conditions, exceptions, duties, thresholds, or scope limits when present.
- Synthesize across sources only when the synthesis is explicitly supported.
- Avoid generic filler, hedging, or motivational language.

Few-shot example 1:
Question: What are the minimum elements of a DPIA?
Sources:
[Source 1] GDPR | Article 35 | A DPIA shall contain a systematic description of the envisaged processing operations and purposes, an assessment of necessity and proportionality, and an assessment of the risks to the rights and freedoms of data subjects.
Answer:
A DPIA must include a systematic description of the envisaged processing operations and their purposes, an assessment of necessity and proportionality, and an assessment of the risks to the rights and freedoms of data subjects [Source 1].

Few-shot example 2:
Question: What fine applies for this violation?
Sources:
[Source 1] This excerpt discusses transparency duties but contains no penalty provision.
Answer:
Insufficient evidence.
"""

JUDGE_SYSTEM_PROMPT = """You are the judge for a strict regulatory RAG benchmark.

Your task:
- Score a candidate answer using only:
  1. the user question
  2. the reference answer
  3. the retrieved evidence
  4. the candidate answer

Scoring dimensions from 0.0 to 1.0:
- faithfulness: every material claim is supported by retrieved evidence only
- answer_relevance: directly and usefully answers the user question
- citation_grounding: citations are present when needed and match the supported claims
- completeness: covers the main supported points without major omission
- overall: overall quality given the four dimensions above

Judging rules:
- Penalize unsupported or invented details very heavily.
- Penalize citation misuse heavily.
- Penalize answers that are elegant but not grounded.
- If 'Insufficient evidence.' is the correct behavior, score faithfulness highly.
- If the answer says 'Insufficient evidence.' when the retrieved evidence is actually enough, penalize relevance and completeness.
- Prefer strictness over generosity.

Return strict JSON only with keys:
faithfulness, answer_relevance, citation_grounding, completeness, overall, verdict, rationale

Where:
- verdict is one of: excellent, good, borderline, poor
- rationale is one short sentence

Few-shot example 1:
Question: What are the reporting duties?
Reference answer: Entities must report major incidents to the authority within the required timeframe.
Retrieved evidence: [Source 1] report major incidents within 24 hours.
Candidate answer: Entities must report major incidents within the required timeframe [Source 1].
Output:
{"faithfulness": 0.98, "answer_relevance": 0.95, "citation_grounding": 0.98, "completeness": 0.90, "overall": 0.96, "verdict": "excellent", "rationale": "Grounded, direct, and correctly cited."}

Few-shot example 2:
Question: What penalties apply?
Reference answer: Fines may reach 2 percent of annual turnover.
Retrieved evidence: [Source 1] no penalty information is present.
Candidate answer: Fines may reach 2 percent of annual turnover [Source 1].
Output:
{"faithfulness": 0.05, "answer_relevance": 0.20, "citation_grounding": 0.05, "completeness": 0.10, "overall": 0.08, "verdict": "poor", "rationale": "The answer invents penalty details not supported by the evidence."}
"""


def call_openrouter(system_prompt: str, user_prompt: str, model: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://rag-pipeline.local",
        "X-Title": "RAG Full Eval",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1400,
    }

    last_error = None
    for attempt in range(1, OPENROUTER_MAX_RETRIES + 1):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=OPENROUTER_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"No choices returned from {model}: {data}")

            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        text_parts.append(part["text"])
                merged = "\n".join(text_parts).strip()
                if merged:
                    return merged
            raise RuntimeError(f"Empty or unsupported response content from {model}: {message}")
        except Exception as exc:
            last_error = exc
            if attempt < OPENROUTER_MAX_RETRIES:
                time.sleep(attempt * 2)
            else:
                raise RuntimeError(
                    f"OpenRouter call failed after {OPENROUTER_MAX_RETRIES} attempts for {model}: {exc}"
                ) from exc
    raise RuntimeError(str(last_error))


def parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def load_evalset() -> list[dict]:
    if not EVALSET_PATH.exists():
        raise FileNotFoundError(f"Missing evalset: {EVALSET_PATH}")
    rows = []
    with EVALSET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if MAX_ITEMS > 0:
        return rows[:MAX_ITEMS]
    return rows


def load_existing_results() -> dict[str, Any]:
    if not EXISTING_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing existing results file: {EXISTING_RESULTS_PATH}")
    data = json.loads(EXISTING_RESULTS_PATH.read_text(encoding="utf-8"))
    if "results" not in data or not isinstance(data["results"], list):
        raise RuntimeError(f"Existing results file has unexpected format: {EXISTING_RESULTS_PATH}")
    if MAX_ITEMS > 0:
        data["results"] = data["results"][:MAX_ITEMS]
    return data


def format_sources(chunks: list[dict]) -> str:
    blocks = []
    for chunk in chunks:
        text = chunk.get("enriched_text") or chunk.get("text") or ""
        blocks.append(f"[Source {chunk['rank']}] {chunk['citation']}\n{text[:2000]}")
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks: list[dict]) -> str:
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved sources:\n{format_sources(chunks)}\n\n"
        "Write the answer now."
    )
    try:
        return call_openrouter(ANSWER_SYSTEM_PROMPT, user_prompt, ANSWER_MODEL)
    except Exception:
        return "Insufficient evidence."


def judge_answer(question: str, reference_answer: str, answer: str, chunks: list[dict]) -> dict[str, Any]:
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Retrieved evidence:\n{format_sources(chunks)}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Return strict JSON only."
    )
    raw = call_openrouter(JUDGE_SYSTEM_PROMPT, user_prompt, JUDGE_MODEL)
    data = parse_json_object(raw)
    required = ["faithfulness", "answer_relevance", "citation_grounding", "completeness", "overall"]
    for key in required:
        data[key] = float(data[key])
    data.setdefault("verdict", "borderline")
    data.setdefault("rationale", "")
    return data


def mean_metric(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows if key in row]
    return round(statistics.mean(values), 4) if values else 0.0


def build_summary_markdown(summary: dict[str, Any], ragas_summary: dict[str, Any] | None) -> str:
    lines = [
        "| Variant | Faithfulness | Relevance | Citation | Completeness | Overall | Insufficient Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['variant']} | {summary['faithfulness']:.4f} | {summary['answer_relevance']:.4f} | "
            f"{summary['citation_grounding']:.4f} | {summary['completeness']:.4f} | {summary['overall']:.4f} | "
            f"{summary['insufficient_evidence_rate']:.4f} |"
        ),
        "",
        f"Answer model: `{ANSWER_MODEL}`",
        f"Judge model: `{JUDGE_MODEL}`",
        f"Eval set: `{EVALSET_PATH.name}` ({summary['num_items']} items)",
    ]
    if ragas_summary:
        lines.extend(
            [
                "",
                "**RAGAS**",
                "",
                f"- faithfulness: {ragas_summary.get('faithfulness', 0.0):.4f}",
                f"- answer_relevancy: {ragas_summary.get('answer_relevancy', 0.0):.4f}",
                f"- context_precision: {ragas_summary.get('context_precision', 0.0):.4f}",
                f"- context_recall: {ragas_summary.get('context_recall', 0.0):.4f}",
            ]
        )
    return "\n".join(lines) + "\n"


def run_ragas(rows: list[dict]) -> dict[str, float]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics.collections import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as exc:
        raise RuntimeError(
            "RAGAS is required for this eval run. Install dependencies first: pip install ragas datasets"
        ) from exc

    # RAGAS may instantiate OpenAI-compatible clients internally. Point them to
    # OpenRouter when a separate OpenAI key is not provided.
    if OPENROUTER_API_KEY and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ.setdefault("OPENAI_BASE_URL", OPENROUTER_URL)
    os.environ.setdefault("OPENAI_API_BASE", OPENROUTER_URL)

    records = {
        "question": [],
        "answer": [],
        "ground_truth": [],
        "contexts": [],
    }
    for row in rows:
        records["question"].append(row["question"])
        records["answer"].append(row["answer"])
        records["ground_truth"].append(row["reference_answer"])
        records["contexts"].append(
            [chunk.get("enriched_text") or chunk.get("text") or "" for chunk in row["retrieval"]["chunks"]]
        )

    metric_objects = [
        faithfulness(),
        answer_relevancy(),
        context_precision(),
        context_recall(),
    ]
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    # Newer/other ragas versions may expose hallucination in different namespaces.
    hallucination_metric = None
    try:
        from ragas.metrics.collections import hallucination as hallucination_factory
        hallucination_metric = hallucination_factory()
    except Exception:
        try:
            from ragas.metrics import hallucination as hallucination_factory
            hallucination_metric = hallucination_factory()
        except Exception:
            hallucination_metric = None

    if hallucination_metric is not None:
        metric_objects.append(hallucination_metric)
        metric_names.append("hallucination")

    dataset = Dataset.from_dict(records)
    result = evaluate(dataset, metrics=metric_objects)
    raw = result.to_pandas().mean(numeric_only=True).to_dict()
    summary = {k: float(v) for k, v in raw.items()}
    for name in metric_names:
        summary.setdefault(name, 0.0)
    return summary


def main() -> None:
    if not BM25_PATH.exists():
        raise FileNotFoundError(f"Missing BM25 index: {BM25_PATH}")

    EVALS_DIR.mkdir(parents=True, exist_ok=True)

    if RAGAS_ONLY:
        payload = load_existing_results()
        rows = payload["results"]
        ragas_summary = run_ragas(rows)
        payload["ragas"] = ragas_summary
        RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = payload.get("summary") or {
            "variant": "hybrid_no_hyde",
            "num_items": len(rows),
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "citation_grounding": 0.0,
            "completeness": 0.0,
            "overall": 0.0,
            "insufficient_evidence_rate": 0.0,
        }
        SUMMARY_PATH.write_text(build_summary_markdown(summary, ragas_summary), encoding="utf-8")
        RAGAS_PATH.write_text(json.dumps(ragas_summary, indent=2), encoding="utf-8")
        print(f"Loaded existing results from {EXISTING_RESULTS_PATH}")
        print(f"Saved updated results to {RESULTS_PATH}")
        print(f"Saved summary to {SUMMARY_PATH}")
        print(f"Saved RAGAS summary to {RAGAS_PATH}")
        print()
        print(build_summary_markdown(summary, ragas_summary))
        return

    evalset = load_evalset()
    pipeline = RetrievalPipeline()

    rows = []
    total = len(evalset)
    for idx, item in enumerate(evalset, 1):
        retrieval = pipeline.retrieve(item["question"], use_hyde=False, verbose=False)
        answer = generate_answer(item["question"], retrieval["chunks"])
        judged = judge_answer(item["question"], item["reference_answer"], answer, retrieval["chunks"])
        row = {
            "eval_id": item.get("eval_id", idx),
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "reference_doc_name": item.get("reference_doc_name", ""),
            "answer": answer,
            "retrieval": retrieval,
            "insufficient_evidence": retrieval.get("insufficient_evidence", False),
            **judged,
        }
        rows.append(row)
        print(f"[{idx}/{total}] overall={row['overall']:.2f} insufficient={row['insufficient_evidence']}")

    summary = {
        "variant": "hybrid_no_hyde",
        "num_items": len(rows),
        "faithfulness": mean_metric(rows, "faithfulness"),
        "answer_relevance": mean_metric(rows, "answer_relevance"),
        "citation_grounding": mean_metric(rows, "citation_grounding"),
        "completeness": mean_metric(rows, "completeness"),
        "overall": mean_metric(rows, "overall"),
        "insufficient_evidence_rate": round(
            sum(1 for row in rows if row["insufficient_evidence"]) / max(1, len(rows)),
            4,
        ),
    }

    payload = {
        "evalset_path": str(EVALSET_PATH),
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "summary": summary,
        "results": rows,
        "ragas": None,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    ragas_summary = run_ragas(rows)
    payload["ragas"] = ragas_summary
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    SUMMARY_PATH.write_text(build_summary_markdown(summary, ragas_summary), encoding="utf-8")
    RAGAS_PATH.write_text(json.dumps(ragas_summary, indent=2), encoding="utf-8")

    print(f"Saved results to {RESULTS_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved RAGAS summary to {RAGAS_PATH}")
    print()
    print(build_summary_markdown(summary, ragas_summary))


if __name__ == "__main__":
    main()
