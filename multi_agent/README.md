# multi_agent — MAD Verification Pipeline

Multi-Agent Debate (MAD) output guardrail for the Guardrails Gateway.  
Verifies enterprise LLM answers against a regulatory evidence corpus.

---

## Setup

### 1. Pull the model into Ollama

```bash
ollama pull qwen2.5:7b
ollama serve          # keep running in a separate terminal
```

### 2. Install dependencies

```bash
pip install -r multi_agent/requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the repo root (or export directly):

```bash
# Required — your Tavily key for Agent B web search
export TAVILY_API_KEY="tvly-xxxxxxxxxxxx"

# Optional — override defaults
export AGENT_MODEL="qwen2.5:7b"
export JUDGE_MODEL="qwen2.5:7b"
export OLLAMA_BASE_URL="http://localhost:11434/v1"
export MAX_CYCLES=2

# Point at your actual JSONL chunks file
export CHUNKS_JSONL_PATH="/path/to/rag/chunks.jsonl"
```

Your JSONL file should have one chunk per line:
```json
{"chunk_id": "gdpr_art32_001", "text": "GDPR Article 32...", "source": "GDPR_2016_679", "tier": 1}
```
Supported field aliases: `text/content/chunk_text`, `source/doc_id/filename`, `tier/authority_tier`, `chunk_id/id`.  
If the file is not found, the pipeline falls back to built-in regulatory sample chunks.

---

## Running

### CLI test (quickest way to verify everything works)

```bash
# Built-in GDPR encryption example
python -m multi_agent.run_test

# Built-in HIPAA example
python -m multi_agent.run_test --example hipaa_phi

# Custom query
python -m multi_agent.run_test \
  --query "Does CCPA require us to delete user data on request?" \
  --answer "Yes, CCPA gives users an absolute right to deletion within 24 hours."

# Save full JSON output
python -m multi_agent.run_test --output-json results.json

# List all built-in examples
python -m multi_agent.run_test --list-examples
```

### FastAPI service

```bash
uvicorn multi_agent.api:app --host 0.0.0.0 --port 8001 --reload
```

Endpoints:
- `POST /mad/verify`  — run MAD pipeline
- `GET  /mad/health`  — liveness
- `GET  /mad/info`    — current config
- `GET  /docs`        — Swagger UI

Example request:
```bash
curl -X POST http://localhost:8001/mad/verify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Does GDPR require us to encrypt customer data at rest?",
    "llm_answer": "Yes. GDPR Article 32 explicitly mandates AES-256 encryption. Fines up to 4% of turnover."
  }'
```

---

## File structure

```
multi_agent/
├── __init__.py          # Package init
├── config.py            # All settings (env-var overridable)
├── models.py            # Pydantic schemas: Claim, Challenge, MADOutput, etc.
├── rag_stub.py          # JSONL-backed retriever (swap for Qdrant here)
├── claim_extractor.py   # LLM-based atomic claim extraction
├── agent_a.py           # Ground Truth Verifier (verify + revise)
├── agent_b.py           # Adversarial Auditor (True→Skeptic + Tavily)
├── judge.py             # Partially blind Judge Agent
├── debate_engine.py     # 2-cycle orchestrator + transcript builder
├── mad_pipeline.py      # Top-level: extraction → debate → judge → routing
├── api.py               # FastAPI endpoints
├── run_test.py          # CLI test runner
└── requirements.txt     # Dependencies
```

---

## Swapping stub RAG for real Qdrant

In `rag_stub.py`, replace the body of `retrieve()` with:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

def retrieve(query: str, top_k: int = TOP_K_CHUNKS) -> List[EvidenceChunk]:
    vector = embed_model.encode(query).tolist()
    hits: List[ScoredPoint] = qdrant_client.search(
        collection_name="regulatory_chunks",
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

Everything else (agent_a, agent_b, judge, debate_engine) calls `retrieve()` unchanged.

---

## Architecture decisions reflected in code

| Decision | Implementation |
|---|---|
| 2 structured cycles | `debate_engine.run_debate(max_cycles=2)` |
| Agent A revises verdicts, not claim text | `agent_a.revise_verdicts()` — claim_text never mutated |
| True→Skeptic rule | `agent_b` challenges ALL SUPPORTED claims across 4 types |
| Partially blind Judge | `judge.py` strips confidence scores from claims JSON |
| Evidence pool unlabelled | `debate_engine` merges chunks from both agents with no agent labels |
| Hard block on v=0.0 material | `mad_pipeline._compute_routing()` checks hard rule before aggregate |
| Min aggregation for material claims | `debate_engine._confidence_signal()`, `mad_pipeline._aggregate_score()` |
| Qdrant-swap-ready interface | `rag_stub.retrieve()` — single function, identical signature |
