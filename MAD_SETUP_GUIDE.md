# Guardrails Gateway — MAD Pipeline
## Complete Setup & Implementation Guide

> **What this covers:** Everything from a blank terminal to a fully running  
> Multi-Agent Debate (MAD) verification pipeline — Ollama setup, model pull,  
> environment config, JSONL chunks, CLI testing, FastAPI service, and Qdrant swap.

---

## Table of Contents

1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Repository Setup](#2-repository-setup)
3. [Ollama Setup](#3-ollama-setup)
4. [Install Python Dependencies](#4-install-python-dependencies)
5. [Environment Variables](#5-environment-variables)
6. [Configure Your Chunk Data](#6-configure-your-chunk-data)
7. [Run Your First Test](#7-run-your-first-test)
8. [Understand the Output](#8-understand-the-output)
9. [Run the FastAPI Service](#9-run-the-fastapi-service)
10. [Test the API Endpoint](#10-test-the-api-endpoint)
11. [Swap in Real Qdrant RAG](#11-swap-in-real-qdrant-rag)
12. [Troubleshooting](#12-troubleshooting)
13. [Architecture Quick Reference](#13-architecture-quick-reference)
14. [File Reference](#14-file-reference)

---

## 1. Prerequisites & System Requirements

### What you need installed before starting

| Tool | Version | Check | Install |
|------|---------|-------|---------|
| Python | 3.10+ | `python3 --version` | [python.org](https://python.org) |
| pip | latest | `pip --version` | bundled with Python |
| Ollama | latest | `ollama --version` | [ollama.com](https://ollama.com) |
| Git | any | `git --version` | bundled with Xcode CLT |

### Hardware

- **MacBook with Apple Silicon (M1/M2/M3/M4):** ✅ Ollama uses Metal GPU automatically  
- **RAM:** 16 GB minimum for `qwen2.5:7b` (recommended). 8 GB will work but slower.  
- **Disk:** ~5 GB free for the model weights  

### Accounts / API Keys you need

| Service | Why | Get it |
|---------|-----|--------|
| Tavily | Agent B web search | [app.tavily.com](https://app.tavily.com) — free tier available |

---

## 2. Repository Setup

### 2a. Clone or navigate to your repo

```bash
# If you haven't cloned yet:
git clone https://github.com/shashidharbabu/guardrails-enterprise.git
cd guardrails-enterprise

# If already cloned:
cd guardrails-enterprise
git checkout deploy
git pull origin deploy
```

### 2b. Drop the MAD codebase into the repo

You received `multi_agent.zip`. Unzip it directly into the repo root.  
This will **replace the empty `multi_agent/` folder** that already exists.

```bash
# From the repo root:
unzip /path/to/multi_agent.zip -d .

# Verify the files landed correctly:
ls multi_agent/
```

You should see:
```
__init__.py       agent_a.py      agent_b.py    api.py
claim_extractor.py  config.py     debate_engine.py  judge.py
mad_pipeline.py   models.py     rag_stub.py   requirements.txt
run_test.py       README.md
```

### 2c. Verify your repo structure

```
guardrails-enterprise/
├── gateway/          ← your existing gateway (classifiers)
├── rag/              ← your existing RAG pipeline
├── multi_agent/      ← MAD pipeline (just added)
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── rag_stub.py
│   ├── claim_extractor.py
│   ├── agent_a.py
│   ├── agent_b.py
│   ├── judge.py
│   ├── debate_engine.py
│   ├── mad_pipeline.py
│   ├── api.py
│   ├── run_test.py
│   └── requirements.txt
├── confidence/
├── rlhf/
└── requirements.txt
```

---

## 3. Ollama Setup

Ollama runs the `qwen2.5:7b` model locally for Agent A, Agent B, and the Judge.  
It exposes an OpenAI-compatible API at `http://localhost:11434`.

### 3a. Verify Ollama is installed

```bash
ollama --version
# Expected: ollama version 0.x.x
```

If not installed:
```bash
# macOS (Homebrew):
brew install ollama

# Or download directly:
# https://ollama.com/download
```

### 3b. Pull the Qwen2.5 7B model

```bash
ollama pull qwen2.5:7b
```

This downloads ~4.7 GB. Takes 5–15 min depending on your connection.  
You'll see a progress bar. Wait until it completes fully.

```
pulling manifest
pulling qwen2.5:7b... ████████████████ 100% 4.7 GB
verifying sha256 digest
writing manifest
success
```

### 3c. Start the Ollama server

```bash
ollama serve
```

**Keep this terminal open and running.** Ollama must be active whenever you run the MAD pipeline.  
Open a new terminal tab for all subsequent steps.

### 3d. Verify the model works

```bash
# Quick smoke test:
ollama run qwen2.5:7b "Say: MAD pipeline ready."
```

Expected: a short response like `MAD pipeline ready.`  
Type `/bye` to exit the chat.

### 3e. Verify the API endpoint

```bash
curl http://localhost:11434/v1/models
```

Expected JSON response listing `qwen2.5:7b`. If you get a connection error, Ollama isn't running — go back to step 3c.

---

## 4. Install Python Dependencies

### 4a. Create a virtual environment (recommended)

```bash
# From repo root:
python3 -m venv .venv
source .venv/bin/activate

# You should now see (.venv) in your prompt
```

### 4b. Install MAD-specific dependencies

```bash
pip install -r multi_agent/requirements.txt
```

This installs:
- `openai` — Ollama client (OpenAI-compatible)
- `tavily-python` — Agent B web search
- `fastapi` + `uvicorn` — API server
- `pydantic>=2.0` — data schemas
- `httpx` — async HTTP

### 4c. Verify key packages

```bash
python3 -c "import openai; import tavily; import fastapi; print('All dependencies OK')"
```

Expected: `All dependencies OK`

---

## 5. Environment Variables

The pipeline is fully configurable via environment variables.  
**Only `TAVILY_API_KEY` is required.** Everything else has working defaults.

### 5a. Create a `.env` file in the repo root

```bash
# From repo root:
touch .env
```

Open `.env` in your editor and add:

```bash
# ── REQUIRED ──────────────────────────────────────────────────────────────────
# Agent B's web search tool (get your key at https://app.tavily.com)
TAVILY_API_KEY=tvly-your-actual-key-here

# ── YOUR CHUNK DATA (set this if you have a JSONL file) ───────────────────────
# Path to your regulatory chunks JSONL file.
# If not set or file not found → falls back to built-in GDPR/HIPAA sample chunks.
CHUNKS_JSONL_PATH=/absolute/path/to/rag/chunks.jsonl

# ── OPTIONAL OVERRIDES (defaults shown) ───────────────────────────────────────
AGENT_MODEL=qwen2.5:7b
JUDGE_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434/v1
MAX_CYCLES=2
CONFIDENCE_THRESHOLD_HIGH=0.8
CONFIDENCE_THRESHOLD_LOW=0.4
MAD_API_PORT=8001
```

### 5b. Load the env file

```bash
# Option A — export from .env manually:
export $(grep -v '^#' .env | xargs)

# Option B — use python-dotenv (add to your shell profile for persistence):
pip install python-dotenv
```

### 5c. Verify your Tavily key works

```bash
python3 -c "
from tavily import TavilyClient
import os
client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY', ''))
r = client.search('GDPR Article 32 encryption requirements')
print(f'Tavily OK — got {len(r[\"results\"])} results')
"
```

Expected: `Tavily OK — got N results`

---

## 6. Configure Your Chunk Data

The MAD pipeline uses a stub RAG retriever backed by your local JSONL chunk file.  
This is the same data already in your Qdrant DB — just the flat JSONL version.

### Option A — Point at your existing JSONL file (recommended)

If your RAG pipeline already produced a JSONL file:

```bash
# Find it:
find . -name "*.jsonl" | head -10

# Set the path:
export CHUNKS_JSONL_PATH=/absolute/path/to/your/chunks.jsonl
```

Your JSONL must have one JSON object per line. Supported field names:

```json
{"chunk_id": "gdpr_art32_001", "text": "GDPR Article 32 requires...", "source": "GDPR_2016_679", "tier": 1}
```

| Your field name | Aliases accepted |
|----------------|-----------------|
| `chunk_id` | `id` |
| `text` | `content`, `chunk_text`, `body` |
| `source` | `doc_id`, `filename` |
| `tier` | `authority_tier` |

The loader accepts all common field name variants — no need to reformat your existing data.

### Option B — Export from Qdrant to JSONL

If your chunks are already in Qdrant but you don't have a JSONL file:

```bash
python3 - <<'EOF'
from qdrant_client import QdrantClient
import json

client = QdrantClient(host="localhost", port=6333)
collection = "regulatory_chunks"   # change to your collection name

offset = None
all_chunks = []

while True:
    records, offset = client.scroll(
        collection_name=collection,
        limit=100,
        offset=offset,
        with_payload=True,
    )
    for r in records:
        all_chunks.append({
            "chunk_id": str(r.id),
            "text":     r.payload.get("text", ""),
            "source":   r.payload.get("source", "unknown"),
            "tier":     r.payload.get("tier", 1),
        })
    if offset is None:
        break

out_path = "rag/chunks.jsonl"
with open(out_path, "w") as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk) + "\n")

print(f"Exported {len(all_chunks)} chunks → {out_path}")
EOF
```

Then set: `export CHUNKS_JSONL_PATH=rag/chunks.jsonl`

### Option C — Use built-in sample chunks (zero setup)

If you skip `CHUNKS_JSONL_PATH`, the pipeline automatically falls back to 10 built-in  
GDPR/HIPAA/NIST sample chunks. These are realistic enough for initial testing.

```
[RAG stub] JSONL not found — using built-in regulatory sample chunks
[RAG stub] Loaded 10 chunks
```

This works immediately. Use it to verify the pipeline runs before plugging in your real data.

---

## 7. Run Your First Test

Everything should be in place now. Let's run it.

### 7a. Confirm Ollama is still running

```bash
# In a separate terminal tab:
ollama serve
```

### 7b. Run the built-in GDPR example

```bash
# From repo root, with .venv active:
python3 -m multi_agent.run_test
```

This runs:
- **Query:** "Does GDPR require us to encrypt customer data at rest?"
- **LLM Answer:** A pre-written answer with 1 accurate claim, 1 partial claim, and 1 hallucinated standard (AES-256 mandate)

You'll see the full pipeline run in your terminal — claim extraction, 2 debate cycles, judge verdicts, routing decision.

**Expected final output:**

```
══════════════════════════════════════════════
  FINAL RESULTS
══════════════════════════════════════════════
  Routing decision    : HARD_BLOCK  (or RETRY)
  Aggregate confidence: 0.35xx
  Debate cycles run   : 2
  Evidence pool size  : 18 chunks

  Judge verdicts:
    ✅ [⚠ material] C1: score=0.5   "GDPR Article 32 mandates encryption at rest"
    ❌ [⚠ material] C2: score=0.0   "AES-256 is the required standard"
    ✅ [⚠ material] C3: score=0.5   "Fines up to 4% of turnover"
```

C2 (AES-256 fabrication) should score 0.0 and trigger HARD_BLOCK.

### 7c. Run the HIPAA example

```bash
python3 -m multi_agent.run_test --example hipaa_phi
```

### 7d. Run the CCPA example

```bash
python3 -m multi_agent.run_test --example ccpa_deletion
```

### 7e. Run a custom query

```bash
python3 -m multi_agent.run_test \
  --query "Does HIPAA allow sharing patient data with third parties for marketing?" \
  --answer "HIPAA permits sharing PHI with third parties for any business purpose as long as a BAA is signed."
```

### 7f. Save full JSON output

```bash
python3 -m multi_agent.run_test --output-json results/gdpr_test.json
```

The JSON file contains the complete `MADOutput` — all claims, all debate cycles, all verdicts, the full transcript. Use this for paper results and evaluation.

---

## 8. Understand the Output

### 8a. What each section means

```
[Step 1/4] Extracting atomic claims...
           Extracted 3 claims (3 material, 0 contextual)
```
The LLM read the answer and broke it into individual verifiable statements.  
Each claim has `is_material=True/False` — material claims trigger hard-block if judge scores 0.0.

---

```
[C1·A] Agent A — initial verification
  [⚠ mat] C1: PARTIAL         p=0.61  "GDPR Art 32 mandates encryption at rest"
  [⚠ mat] C2: NOT_SUPPORTED   p=0.08  "AES-256 is the required standard"
  [⚠ mat] C3: SUPPORTED       p=0.91  "Fines up to 4% of turnover"
```
Agent A's first pass. C2 immediately flagged NOT_SUPPORTED — good.

---

```
[C1·B] Agent B — adversarial challenges
  C1 [CHUNK_CURRENCY      ] → PARTIAL
       EDPB 2024 guidance confirms Art 32 does not mandate encryption specifically...
  C3 [EXCEPTION_EXISTENCE ] → PARTIAL
       4% fine applies specifically to Art 83(5) infringements, not all non-compliance...
```
Agent B applies the True→Skeptic rule to C1 (SUPPORTED) and C3 (SUPPORTED), finds real issues.

---

```
[C1·C] Agent A — revising verdicts
  [⚠ mat] C1: PARTIAL         p=0.52  (revised from PARTIAL after B's EDPB chunk)
  [⚠ mat] C3: PARTIAL         p=0.71  (revised — B's Art 83(5) challenge valid)
```
Agent A updates confidence and verdicts. Claim text never changes.

---

```
  JUDGE VERDICTS
  ⚠ C1: score=0.5  "GDPR Article 32 mandates encryption at rest"
       Art 32 confirmed but encryption is one option, not mandatory...
  ❌ C2: score=0.0  "AES-256 is the required standard"
       No evidence for AES-256 in GDPR corpus. NIST recommendation, not law.
  ⚠ C3: score=0.5  "Fines up to 4% of turnover"
       Percentage correct but applies to Art 83(5) serious infringements only...
```
Judge sees clean evidence pool + final verdicts. Strips confidence to avoid anchoring.

---

```
  Routing decision    : HARD_BLOCK
  Correction signal   : The answer incorrectly claims AES-256 is mandated by GDPR...
```
`HARD_BLOCK` because C2 is `is_material=True` and `score=0.0`.  
The correction signal would be sent to the LLM on retry.

### 8b. Routing decision meanings

| Decision | When | What happens |
|----------|------|-------------|
| `DELIVER` | aggregate > 0.8, all material v=1.0 | Answer is verified — deliver to user |
| `RETRY` | aggregate 0.4–0.8 | Send correction signal to LLM, run MAD again (max 1 retry) |
| `HUMAN_REVIEW` | aggregate < 0.4 | Queue for human expert review |
| `HARD_BLOCK` | any material claim with judge score 0.0 | Block regardless of aggregate — hallucinated regulation |

---

## 9. Run the FastAPI Service

The API lets the gateway call MAD as a microservice.

### 9a. Start the server

```bash
# Make sure Ollama is still running in another tab, then:
uvicorn multi_agent.api:app --host 0.0.0.0 --port 8001 --reload
```

You should see:

```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Application startup complete.
```

### 9b. Check it's alive

```bash
curl http://localhost:8001/mad/health
```

Expected: `{"status":"ok","service":"MAD pipeline"}`

### 9c. Check config

```bash
curl http://localhost:8001/mad/info | python3 -m json.tool
```

Expected JSON showing your model, thresholds, base URL.

### 9d. Open Swagger UI

Open in browser: **http://localhost:8001/docs**

You can test the `/mad/verify` endpoint interactively here without writing any curl commands.

---

## 10. Test the API Endpoint

### 10a. POST /mad/verify — basic test

```bash
curl -X POST http://localhost:8001/mad/verify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Does GDPR require us to encrypt customer data at rest?",
    "llm_answer": "Yes. GDPR Article 32 explicitly mandates encryption of personal data at rest. The regulation specifies AES-256 as the required encryption standard. Organizations that fail to encrypt data at rest face fines up to 4% of global annual turnover."
  }' | python3 -m json.tool
```

### 10b. Expected response structure

```json
{
  "routing_decision": "HARD_BLOCK",
  "aggregate_confidence": 0.35,
  "correction_signal": "The answer incorrectly states GDPR Art 32 mandates AES-256...",
  "claims": [
    {
      "claim_id": 1,
      "claim_text": "GDPR Article 32 mandates encryption of personal data at rest",
      "is_material": true,
      "confidence": 0.52,
      "verdict": "PARTIAL",
      "evidence_chunks": ["gdpr_art32_p1", "edpb_guidelines_2024_encryption"],
      "reasoning": "Art 32 lists encryption as one appropriate measure..."
    }
  ],
  "judge_verdicts": [
    { "claim_id": 1, "score": 0.5, "is_material": true, "reasoning": "..." },
    { "claim_id": 2, "score": 0.0, "is_material": true, "reasoning": "AES-256 not in GDPR..." },
    { "claim_id": 3, "score": 0.5, "is_material": true, "reasoning": "..." }
  ],
  "debate_transcript": "====...full transcript...===="
}
```

### 10c. Python client example

```python
import requests

response = requests.post(
    "http://localhost:8001/mad/verify",
    json={
        "query": "Does HIPAA require encryption of ePHI?",
        "llm_answer": "HIPAA mandates AES-256 encryption for all ePHI at rest."
    }
)

result = response.json()
print(f"Routing: {result['routing_decision']}")
print(f"Confidence: {result['aggregate_confidence']:.3f}")

if result['correction_signal']:
    print(f"Correction: {result['correction_signal']}")
```

---

## 11. Swap in Real Qdrant RAG

When you're ready to connect your actual Qdrant vector database, the change is **one function** in one file.

### 11a. Open `multi_agent/rag_stub.py`

Find the `retrieve()` function (around line 40).

### 11b. Replace the function body

```python
# Replace the entire body of retrieve() with this:

from qdrant_client import QdrantClient
from your_rag_module import get_embedding_model   # your actual embed function

_qdrant = QdrantClient(host="localhost", port=6333)
_embed  = get_embedding_model()                   # your Qwen3-4B embedder

def retrieve(query: str, top_k: int = TOP_K_CHUNKS) -> List[EvidenceChunk]:
    vector = _embed.encode(query).tolist()
    hits   = _qdrant.search(
        collection_name="regulatory_chunks",       # your actual collection name
        query_vector=vector,
        limit=top_k,
    )
    return [
        EvidenceChunk(
            chunk_id=str(h.id),
            text=h.payload["text"],
            source=h.payload.get("source", "unknown"),
            tier=h.payload.get("tier", 1),
        )
        for h in hits
    ]
```

### 11c. Nothing else changes

`agent_a.py`, `agent_b.py`, `judge.py`, `debate_engine.py`, `mad_pipeline.py` — all call `retrieve()` with the same interface. Zero changes needed anywhere else.

### 11d. Test with real RAG

```bash
python3 -m multi_agent.run_test --output-json results/qdrant_test.json
```

Compare evidence chunks in the output JSON — you should now see chunk IDs and source names from your actual Qdrant collection.

---

## 12. Troubleshooting

### ❌ `Connection refused` on Ollama

```
openai.APIConnectionError: Connection error.
```

**Fix:** Ollama isn't running. Open a new terminal and run:
```bash
ollama serve
```

---

### ❌ `model not found` error

```
ollama._types.ResponseError: model "qwen2.5:7b" not found
```

**Fix:** Pull the model first:
```bash
ollama pull qwen2.5:7b
```

---

### ❌ JSON parse error from LLM

```
ValueError: [claim_extractor] Failed to parse LLM response as JSON.
```

**Cause:** The LLM wrapped its response in markdown fences (` ```json `) or added explanation text.  
**Fix:** This is handled automatically by `_clean_json_array()`. If it still fails, the model may be producing a completely malformed response. Try:

```bash
# Test Ollama directly:
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b","messages":[{"role":"user","content":"Return: [1,2,3]"}]}'
```

If that works, increase temperature tolerance or retry.

---

### ❌ `CHUNKS_JSONL_PATH` not found warning

```
[RAG stub] JSONL not found at /path/chunks.jsonl — using built-in regulatory sample chunks
```

**This is not an error** — the pipeline uses sample chunks. To use your real data:
```bash
export CHUNKS_JSONL_PATH=/correct/absolute/path/to/chunks.jsonl
python3 -m multi_agent.run_test   # re-run
```

---

### ❌ Tavily search failing

```
[Agent B] Tavily web search failed: ...
```

**The pipeline continues without web search.** Agent B falls back to RAG-only challenges.  
To fix: check your `TAVILY_API_KEY` is set and correct.

---

### ❌ `ModuleNotFoundError: No module named 'multi_agent'`

```
ModuleNotFoundError: No module named 'multi_agent'
```

**Fix:** You must run from the repo root, not from inside the `multi_agent/` folder:
```bash
# Wrong:
cd multi_agent && python3 run_test.py

# Correct:
cd guardrails-enterprise   # repo root
python3 -m multi_agent.run_test
```

---

### ❌ Out of memory / Ollama crashes

The 7B model needs ~5.5 GB RAM. If your Mac has other apps open:
```bash
# Close other applications, then:
ollama serve

# Or try the 3B model for development:
ollama pull qwen2.5:3b
export AGENT_MODEL=qwen2.5:3b
export JUDGE_MODEL=qwen2.5:3b
```

---

### ❌ Slow performance

- **First run is always slower** — model weights load into memory. Subsequent runs are fast.
- **Expected runtimes on M2 MacBook (16GB):**
  - Claim extraction: ~10–20s
  - Each debate cycle: ~30–60s
  - Judge: ~15–25s
  - **Total per query: ~2–4 minutes for 2 cycles**

---

## 13. Architecture Quick Reference

```
Enterprise Employee Query
        │
        ▼
┌─────────────────────────────────────┐
│  Phase 1 — Input Guardrail          │
│  Prompt Injection │ PII │ Jailbreak  │  → BLOCK
│         Decision Engine              │
└─────────────────┬───────────────────┘
                  │ PASS
                  ▼
┌─────────────────────────────────────┐
│  Phase 2 — Enterprise LLM           │
│  Generates answer (no RAG context)  │
└─────────────────┬───────────────────┘
                  │ raw answer (unverified)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3 — MAD Output Guardrail                             │
│                                                             │
│  ┌──────────────┐    ┌────────────────────────────────────┐ │
│  │ Layer 1      │    │ MAD Debate Pipeline                │ │
│  │ (parallel)   │    │                                    │ │
│  │ Faithfulness │    │  Claim Extraction (single pass)    │ │
│  │ Hallucination│    │         ↓                          │ │
│  │ Relevancy    │    │  ┌── CYCLE 1 ──────────────────┐   │ │
│  │     ↓        │    │  │ A: verify  ↔  B: challenge  │   │ │
│  │  → CSE async │    │  │ A: revise verdicts           │   │ │
│  └──────────────┘    │  └──────────────────────────────┘   │ │
│                      │         ↓                          │ │
│                      │  ┌── CYCLE 2 ──────────────────┐   │ │
│                      │  │ B: follow-up ↔ A: final     │   │ │
│                      │  └──────────────────────────────┘   │ │
│                      │         ↓                          │ │
│                      │  Evidence Pool (both agents)       │ │
│                      │         ↓                          │ │
│                      │  Judge (partially blind)           │ │
│                      │  v = 1.0 / 0.5 / 0.0              │ │
│                      │         ↓                          │ │
│                      │  G-Eval Layer 2                    │ │
│                      └──────────────────────┬─────────────┘ │
└─────────────────────────────────────────────┼───────────────┘
                                              │
                                              ▼
                                 Confidence Scoring Engine
                                 0.30·F + 0.25·(1-H) + 0.10·R + 0.35·J
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │       Routing Decision        │
                              │  Hard rule: is_material+v=0.0 │
                              │  → HARD_BLOCK (human review)  │
                              │  score > 0.8 → DELIVER        │
                              │  score 0.4-0.8 → RETRY        │
                              │  score < 0.4 → HUMAN_REVIEW   │
                              └───────────────────────────────┘
```

---

## 14. File Reference

| File | What it does | When to edit |
|------|-------------|--------------|
| `config.py` | All settings + thresholds | Change model, ports, thresholds |
| `models.py` | Pydantic schemas for all data | Adding new fields to claims/verdicts |
| `rag_stub.py` | JSONL loader + keyword retrieval | **Swap in Qdrant here** |
| `claim_extractor.py` | LLM call to extract atomic claims | Tune extraction prompt |
| `agent_a.py` | Ground Truth Verifier | Tune verification or revision prompts |
| `agent_b.py` | Adversarial Auditor + Tavily | Tune challenge types or web search |
| `judge.py` | Partially blind Judge | Tune scoring criteria |
| `debate_engine.py` | 2-cycle orchestrator + transcript | Change cycle logic or early exit |
| `mad_pipeline.py` | Top-level: extract→debate→judge→route | Change routing thresholds |
| `api.py` | FastAPI service | Add endpoints or auth |
| `run_test.py` | CLI test runner | Add new example queries |

---

## Quick Command Reference

```bash
# ── Setup ─────────────────────────────────────────────────
ollama pull qwen2.5:7b
pip install -r multi_agent/requirements.txt
export TAVILY_API_KEY=tvly-your-key

# ── Every time you work ───────────────────────────────────
ollama serve                          # terminal 1 — keep open
source .venv/bin/activate             # terminal 2 — your working terminal
export $(grep -v '^#' .env | xargs)  # load env vars

# ── Test ──────────────────────────────────────────────────
python3 -m multi_agent.run_test
python3 -m multi_agent.run_test --example hipaa_phi
python3 -m multi_agent.run_test --output-json results.json
python3 -m multi_agent.run_test --list-examples

# ── API ───────────────────────────────────────────────────
uvicorn multi_agent.api:app --port 8001 --reload
curl http://localhost:8001/mad/health
# Swagger: http://localhost:8001/docs

# ── Custom query ──────────────────────────────────────────
python3 -m multi_agent.run_test \
  --query "Your query here" \
  --answer "LLM answer to verify"
```

---

*Guardrails Gateway · Shashidhar Babu PVD et al. · SJSU CS298B · 2025–26*  
*MAD pipeline version 0.1.0 — stub RAG, Ollama backend, FastAPI service*
