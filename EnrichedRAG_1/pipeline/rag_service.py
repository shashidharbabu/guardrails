#!/usr/bin/env python3
"""
rag_service.py

Single entrypoint for using the EnrichedRAG_1 retrieval stack from MAD or any
other agent workflow.

Why this exists:
- MAD should not know retrieval internals.
- MAD should call one function and get grounded evidence back.
- This wrapper hides path wiring, fallback behavior, and prompt assembly.

Behavior:
- Preferred path: full hybrid retrieval via retrieval.py
- Fallback path: sparse-only retrieval when dense/Qdrant artifacts are missing

The export folder intentionally excludes the large local Qdrant storage from
GitHub, so sparse-only fallback is important for portability.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


EXPORT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = EXPORT_ROOT / "data" / "rechunked_output"
INDEX_ROOT = EXPORT_ROOT / "indexed_data"
ORIGINAL_ROOT = Path(os.environ.get("RAG_SOURCE_ROOT", "/Users/vineethrayadurgam/Desktop/RagforMAD11"))


def _resolve_runtime_paths() -> dict[str, Path]:
    """
    Prefer local full old-pipeline artifacts when available so this export can
    be tested in full hybrid mode on the original machine.
    """
    export_bm25 = INDEX_ROOT / "bm25_index.pkl"
    export_enriched = DATA_ROOT / "enriched_chunks.json"
    export_qdrant_dir = INDEX_ROOT / "qdrant_data"

    original_bm25 = ORIGINAL_ROOT / "Indexeddata" / "bm25_index.pkl"
    original_enriched = ORIGINAL_ROOT / "rechunked_output" / "enriched_chunks.json"
    original_qdrant_dir = ORIGINAL_ROOT / "Indexeddata"

    use_original_dense = (
        original_qdrant_dir.exists()
        and (original_qdrant_dir / "storage (1).sqlite").exists()
        and (original_qdrant_dir / "meta (1).json").exists()
    )

    return {
        "bm25_path": original_bm25 if original_bm25.exists() else export_bm25,
        "enriched_path": original_enriched if original_enriched.exists() else export_enriched,
        "qdrant_dir": original_qdrant_dir if use_original_dense else export_qdrant_dir,
        "using_original_dense": use_original_dense,
    }


RUNTIME_PATHS = _resolve_runtime_paths()


def _patch_retrieval_paths() -> Any:
    """
    Import retrieval.py and rewrite its path globals so it works from the
    EnrichedRAG_1 export layout instead of the original RagforMAD11 layout.
    """
    os.environ.setdefault("RAG_BASE_DIR", str(EXPORT_ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import retrieval as retrieval_module

    retrieval_module.BASE_DIR = EXPORT_ROOT
    retrieval_module.BM25_PATH = RUNTIME_PATHS["bm25_path"]
    retrieval_module.ENRICHED_CHUNKS_PATH = RUNTIME_PATHS["enriched_path"]

    qdrant_dir = RUNTIME_PATHS["qdrant_dir"]
    if qdrant_dir.exists():
        retrieval_module.QDRANT_PATH = str(qdrant_dir)
        os.environ["QDRANT_PATH"] = str(qdrant_dir)

    return retrieval_module


RETRIEVAL = _patch_retrieval_paths()


def _load_chunks() -> list[dict]:
    enriched_path = RUNTIME_PATHS["enriched_path"]
    if not enriched_path.exists():
        raise FileNotFoundError(f"Missing enriched chunks file: {enriched_path}")
    return json.loads(enriched_path.read_text(encoding="utf-8"))


def _load_bm25():
    import pickle

    bm25_path = RUNTIME_PATHS["bm25_path"]
    if not bm25_path.exists():
        raise FileNotFoundError(f"Missing BM25 index file: {bm25_path}")
    with bm25_path.open("rb") as f:
        return pickle.load(f)


def _dense_artifacts_present() -> bool:
    """
    The export does not currently include the full Qdrant local storage.
    If the user regenerates it later under indexed_data/qdrant_data, hybrid
    retrieval will start working automatically.
    """
    qdrant_dir = RUNTIME_PATHS["qdrant_dir"]
    if not qdrant_dir.exists():
        return False
    return (
        (qdrant_dir / "storage.sqlite").exists()
        or (qdrant_dir / "storage (1).sqlite").exists()
    )


def build_mad_evidence_prompt(question: str, chunks: list[dict]) -> str:
    """
    Create a structured evidence block for downstream MAD agents.

    This is intentionally detailed and includes a few-shot section because
    debate-style agents tend to drift unless the retrieval contract is explicit.
    """
    source_blocks = []
    for chunk in chunks:
        source_blocks.append(
            f"[Source {chunk['rank']}]\n"
            f"Citation: {chunk['citation']}\n"
            f"Tier: {chunk.get('tier', '')}\n"
            f"Text:\n{(chunk.get('enriched_text') or chunk.get('text') or '')[:2200]}"
        )

    joined_sources = "\n\n".join(source_blocks)
    return f"""You are a downstream MAD agent consuming retrieved regulatory evidence.

Role:
- Use only the provided evidence.
- Make claims only when supported by one or more sources.
- Prefer precise regulatory wording over generic summaries.
- If the evidence is not enough, explicitly say so.

Task:
- Answer or reason about the question below using the retrieved evidence only.
- Cite sources inline using [Source N].
- Preserve important qualifiers such as conditions, exceptions, scope, thresholds, or duties.

Output expectations:
- Be concise but complete.
- Do not invent article numbers, deadlines, penalties, or definitions.
- Do not merge unsupported claims across documents.

Few-shot example 1:
Question:
What are the minimum elements of a DPIA?

Evidence:
[Source 1]
Citation: GDPR | Article 35
Text:
A DPIA shall contain a systematic description of the envisaged processing operations and purposes, an assessment of necessity and proportionality, and an assessment of risks to the rights and freedoms of data subjects.

Good answer:
A DPIA must include a systematic description of the envisaged processing operations and their purposes, an assessment of necessity and proportionality, and an assessment of risks to the rights and freedoms of data subjects [Source 1].

Few-shot example 2:
Question:
What fine applies for this violation?

Evidence:
[Source 1]
Citation: Unknown
Text:
This passage discusses transparency duties but contains no penalty provision.

Good answer:
Insufficient evidence.

Question:
{question}

Retrieved evidence:
{joined_sources}
"""


def _build_sparse_only_service():
    """
    Portable fallback when dense/Qdrant artifacts are not available in the
    GitHub export.
    """
    chunks = _load_chunks()
    sparse = RETRIEVAL.SparseRetriever(chunks)
    reranker = RETRIEVAL.BGEReranker()

    def retrieve_for_mad(question: str, top_k: int = 5, verbose: bool = False) -> dict[str, Any]:
        sparse_results = sparse.retrieve(question, top_k=RETRIEVAL.SPARSE_TOP_K)
        fused = [
            {
                "chunk_id": r["chunk_id"],
                "rrf_score": r["score"],
                "sources": {"sparse"},
                "payload": r["payload"],
            }
            for r in sparse_results[: max(RETRIEVAL.RERANK_CANDIDATES, top_k)]
        ]
        reranked = reranker.rerank(question, fused, top_k=max(RETRIEVAL.RERANK_CANDIDATES, top_k))
        final = RETRIEVAL.apply_tier_boost(reranked, final_k=top_k)
        strength = RETRIEVAL.assess_retrieval_strength(final)

        output_chunks = []
        for rank, c in enumerate(final, 1):
            output_chunks.append(
                {
                    "rank": rank,
                    "chunk_id": c["chunk_id"],
                    "rerank_score": c.get("rerank_score", 0.0),
                    "rrf_score": c.get("rrf_score", 0.0),
                    "sources": sorted(list(c.get("sources", {"sparse"}))),
                    "tier": c["payload"].get("tier", ""),
                    "doc_name": c["payload"].get("doc_name", ""),
                    "article_number": c["payload"].get("article_number", ""),
                    "text": c["payload"].get("text", ""),
                    "enriched_text": c["payload"].get("enriched_text", ""),
                    "context_prefix": c["payload"].get("context_prefix", ""),
                    "source_url": c["payload"].get("source_url", ""),
                    "citation": RETRIEVAL.build_citation(c["payload"]),
                }
            )

        result = {
            "query": question,
            "mode": "sparse_only_fallback",
            "use_hyde": False,
            "evidence": strength,
            "insufficient_evidence": not strength["is_sufficient"],
            "chunks": output_chunks,
            "mad_evidence_prompt": build_mad_evidence_prompt(question, output_chunks),
        }
        if verbose:
            print(
                f"[Sparse fallback] top_k={len(output_chunks)} "
                f"sufficient={strength['is_sufficient']} "
                f"top_rerank={strength['top_rerank_score']:.4f}"
            )
        return result

    return retrieve_for_mad


def _build_hybrid_service():
    """
    Preferred path when dense artifacts are available.
    """
    pipeline = RETRIEVAL.RetrievalPipeline()

    def retrieve_for_mad(question: str, top_k: int = 5, verbose: bool = False) -> dict[str, Any]:
        result = pipeline.retrieve(question, use_hyde=False, verbose=verbose)
        result["mode"] = "hybrid_no_hyde"
        result["chunks"] = result["chunks"][:top_k]
        result["mad_evidence_prompt"] = build_mad_evidence_prompt(question, result["chunks"])
        return result

    return retrieve_for_mad


if _dense_artifacts_present():
    try:
        retrieve_for_mad = _build_hybrid_service()
        SERVICE_MODE = "hybrid_no_hyde"
        SERVICE_INIT_ERROR = ""
    except Exception as exc:
        # Local environments with older transformers/tokenizers can fail to
        # load Qwen3-Embedding-4B even when the dense artifacts exist.
        retrieve_for_mad = _build_sparse_only_service()
        SERVICE_MODE = "sparse_only_fallback"
        SERVICE_INIT_ERROR = str(exc)
else:
    retrieve_for_mad = _build_sparse_only_service()
    SERVICE_MODE = "sparse_only_fallback"
    SERVICE_INIT_ERROR = ""


def main() -> None:
    print("=" * 60)
    print("  RAG SERVICE — EnrichedRAG_1")
    print("=" * 60)
    print(f"  Service mode: {SERVICE_MODE}")
    print(f"  Export root:  {EXPORT_ROOT}")
    print(f"  Data root:    {RUNTIME_PATHS['enriched_path'].parent}")
    print(f"  BM25 path:    {RUNTIME_PATHS['bm25_path']}")
    print(f"  Qdrant path:  {RUNTIME_PATHS['qdrant_dir']}")
    print(f"  Using local original dense artifacts: {RUNTIME_PATHS['using_original_dense']}")
    if SERVICE_INIT_ERROR:
        print(f"  Hybrid init fallback reason: {SERVICE_INIT_ERROR}")
    print()

    test_queries = [
        "What are the requirements for information security management under ISO 27001?",
        "What rights do data subjects have under GDPR?",
        "How should AI systems be assessed for bias and fairness?",
    ]

    for q in test_queries:
        print(f"Query: {q}")
        result = retrieve_for_mad(q, top_k=5, verbose=True)
        print(f"Mode: {result['mode']}")
        print(f"Insufficient evidence: {result['insufficient_evidence']}")
        for chunk in result["chunks"]:
            print(f"  #{chunk['rank']} {chunk['doc_name']} | {chunk['citation']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
