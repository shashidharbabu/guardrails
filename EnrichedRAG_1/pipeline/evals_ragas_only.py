import json
import os
from pathlib import Path
from typing import Any

# Define paths
BASE_DIR = Path(os.environ.get("RAG_BASE_DIR", Path(__file__).parent))
OUTPUT_DIR = BASE_DIR / "evals_output"
EXISTING_RESULTS_PATH = OUTPUT_DIR / "full_eval_85_results.json"
UPDATED_RESULTS_PATH = OUTPUT_DIR / "full_eval_85_results_with_ragas.json"
RAGAS_JSON_PATH = OUTPUT_DIR / "full_eval_85_ragas.json"
RAGAS_SUMMARY_MD_PATH = OUTPUT_DIR / "full_eval_85_ragas_summary.md"

def load_existing_results() -> dict[str, Any]:
    if not EXISTING_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing existing results file: {EXISTING_RESULTS_PATH}")
    return json.loads(EXISTING_RESULTS_PATH.read_text(encoding="utf-8"))

def build_ragas_dataset(rows: list[dict]):
    from datasets import Dataset
    records = {"question": [], "answer": [], "ground_truth": [], "contexts": []}
    for row in rows:
        # Contexts must be a list of strings
        contexts = [chunk.get("enriched_text") or chunk.get("text") or "" for chunk in row.get("retrieval", {}).get("chunks", [])]
        records["question"].append(row["question"])
        records["answer"].append(row["answer"])
        records["ground_truth"].append(row["reference_answer"])
        records["contexts"].append(contexts)
    return Dataset.from_dict(records)

def resolve_ragas_metrics():
    # Use capitalized class names from the collections module
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    metric_objects = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    return metric_objects

def run_ragas(rows: list[dict]) -> dict[str, float]:
    from ragas import evaluate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    # Judge LLM (gpt-4o-mini is great for this)
    ragas_llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_API_BASE"],
        max_tokens=4096 
    )
    
    # Dedicated Embedding Model (Necessary to avoid the 'modality' error)
    ragas_embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small", 
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_API_BASE"]
    )

    dataset = build_ragas_dataset(rows)
    metric_objects = resolve_ragas_metrics()
    
    # Run evaluation
    result = evaluate(
        dataset, 
        metrics=metric_objects,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    
    return {k: float(v) for k, v in result.to_pandas().mean(numeric_only=True).to_dict().items()}

def main() -> None:
    print("🚀 Loading existing results...")
    payload = load_existing_results()
    rows = payload["results"]
    
    print(f"📊 Starting RAGAS evaluation on {len(rows)} items...")
    ragas_summary = run_ragas(rows)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload["ragas"] = ragas_summary
    
    # Save files
    UPDATED_RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    RAGAS_JSON_PATH.write_text(json.dumps(ragas_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print("\n✅ DONE! FINAL RAGAS SCORES:")
    for metric, score in ragas_summary.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()
