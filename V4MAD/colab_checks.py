"""
Colab environment check — run this cell FIRST before any step.

Verifies: GPU, VRAM, required packages, Qdrant connectivity, Langfuse connectivity.
Prints install commands for anything missing.
"""

import subprocess
import sys


def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ── 1. GPU ────────────────────────────────────────────────────────────────────
section("GPU / CUDA")
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"  torch       : {torch.__version__}")
    print(f"  CUDA        : {cuda}")
    if cuda:
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU         : {name}")
        print(f"  VRAM        : {vram:.1f} GB")
        if vram < 35:
            print("  ⚠ Less than 35GB VRAM — use 4-bit QLoRA for 14B models")
        else:
            print("  ✓ VRAM sufficient for 14B BF16 + 14B judge")
    else:
        print("  ✗ No CUDA GPU — vLLM will fail")
except ImportError:
    print("  ✗ torch not installed")


# ── 2. Required packages ───────────────────────────────────────────────────────
section("Required Packages")

packages = {
    "vllm":          "vllm",
    "langgraph":     "langgraph",
    "langfuse":      "langfuse",
    "openai":        "openai",
    "transformers":  "transformers",
    "pydantic":      "pydantic",
    "sklearn":       "scikit-learn",
    "qdrant_client": "qdrant-client",
    "rank_bm25":     "rank-bm25",
    "requests":      "requests",
    "numpy":         "numpy",
}

missing = []
for import_name, pip_name in packages.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  ✓ {import_name:<20} {ver}")
    except ImportError:
        print(f"  ✗ {import_name:<20} MISSING  →  pip install {pip_name}")
        missing.append(pip_name)

if missing:
    print(f"\n  Run to fix:\n  pip install {' '.join(missing)}")
else:
    print("\n  All packages present ✓")


# ── 3. LangGraph version check ────────────────────────────────────────────────
section("LangGraph Async Checkpointer")
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    print("  ✓ AsyncSqliteSaver available")
except ImportError:
    print("  ✗ AsyncSqliteSaver not found")
    print("    pip install 'langgraph>=0.2.0'")


# ── 4. Qdrant ─────────────────────────────────────────────────────────────────
section("Qdrant (optional — used for live per-claim retrieval)")
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(host="localhost", port=6333, timeout=3)
    colls  = client.get_collections().collections
    print(f"  ✓ Qdrant running — {len(colls)} collections")
    for c in colls:
        print(f"      {c.name}")
except Exception as e:
    print(f"  ⚠ Qdrant not reachable ({e})")
    print("    Step 3 will use stored query chunks + TF-IDF reranking (no Qdrant needed)")


# ── 5. Langfuse ───────────────────────────────────────────────────────────────
section("Langfuse Tracing")
try:
    from langfuse import Langfuse
    lf = Langfuse(
        secret_key="sk-lf-9a48abf7-0969-4974-8dfd-b5d8108b5022",
        public_key="pk-lf-2efefb74-7c5a-415e-9bb7-062ae2edc2d5",
        host="https://us.cloud.langfuse.com",
    )
    lf.auth_check()
    print("  ✓ Langfuse connected")
except Exception as e:
    print(f"  ⚠ Langfuse not reachable ({e})")
    print("    All nodes will disable tracing gracefully — pipeline still runs")


# ── 6. Data files ─────────────────────────────────────────────────────────────
section("Data Files")
from pathlib import Path

ROOT  = Path(__file__).resolve().parent
files = {
    "data/queries_50.json":      "Run utils/extract_query_chunks.py first",
    "data/query_chunks_50.json": "Run utils/extract_query_chunks.py first",
}
for fname, hint in files.items():
    p = ROOT / fname
    if p.exists():
        size = p.stat().st_size // 1024
        print(f"  ✓ {fname:<35} ({size} KB)")
    else:
        print(f"  ✗ {fname:<35} MISSING — {hint}")


# ── 7. Summary ────────────────────────────────────────────────────────────────
section("Run Order (once all checks pass)")
print("""
  python utils/extract_query_chunks.py   # one-time: extract chunks from old DB

  python steps/step1_baseline.py         # 3B baseline LLM, no RAG
  python steps/step2_decompose.py        # 7B decomposer → atomic claims
  python steps/step3_claim_rag.py        # per-claim chunk assignment (no GPU)
  python steps/step4_debate.py           # 14B debate (R0 + R1, shared server)
  python steps/step5_judge.py            # 14B (or 32B) judge verdicts
""")
