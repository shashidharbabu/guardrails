# Guardrails Gateway — Project Context for Claude Code
## SJSU CS298B · Team 3 · 2025–26

This file gives Claude Code the full context of the Guardrails Gateway project.
Read this before touching any file in this repository.

---

## What this project is

**Guardrails Gateway** is a model-agnostic enterprise AI security system that wraps
any enterprise LLM (banking, healthcare, legal, HR) with two guardrail layers:

1. **Input Guardrail (Gateway)** — blocks threats before they reach the LLM
2. **Output Guardrail (MAD)** — verifies the LLM's answer against a regulatory corpus
   before it reaches the user

The system is domain-agnostic. The same pipeline works for a hospital system,
a bank, a law firm, or an HR department. The only thing that changes is the
regulatory corpus in the RAG database.

**Primary evaluation domain for the paper: Healthcare / Hospital.**

---

## Repository structure

```
guardrails-enterprise/
├── CLAUDE.md                  ← you are here — read first
├── gateway/                   ← Phase 1: input guardrail (3 classifiers)
├── rag/                       ← RAG pipeline (Qdrant, Nemotron-8B embedder)
├── multi_agent/               ← Phase 2: MAD output guardrail (active dev)
├── confidence/                ← Phase 3: Confidence Scoring Engine (pending)
├── rlhf/                      ← Phase 4: GRPO feedback loop (pending)
├── finetuning/                ← Fine-tuning scripts for all 4 models
├── synthetic_data/            ← Synthetic dataset generation
├── dags/                      ← Airflow DAGs
├── docker/                    ← Docker / docker-compose
├── config/                    ← Shared config
├── docs/                      ← Documentation
├── mad_store.db               ← SQLite DB written by MAD pipeline (auto-created)
└── requirements.txt           ← Root-level shared dependencies
```

---

## The four fine-tuned models

This project fine-tunes exactly 4 models. Do not suggest replacing them.

| # | Model | Task | Status | Key metrics |
|---|-------|------|--------|-------------|
| 1 | RoBERTa-base | Jailbreak detection | ✅ Done | F1 0.9821, OOD evaluated |
| 2 | RoBERTa-base NER | PII detection | ✅ Done | 98.2% recall, 57 entity types |
| 3 | Llama-Prompt-Guard-2-86M | Prompt injection | ✅ Done | F1 0.9821, 1.2% missed attack rate |
| 4 | Qwen2.5-3B-Instruct | LLM generator (benchmarking) | ✅ Done | Base eval: ROUGE-L 0.2634 |
| + | Qwen3-4B | Embedding model (RAG) | ✅ Done | Recall@1 94.9%, nDCG@10 0.891 |

The RAG embedding model (Qwen3-4B) is fine-tuned separately with LoRA
(r=16, α=32, all layers, 3 epochs, 2e-4 lr) on 13,572 query-chunk pairs.

---

## Phase 1 — Gateway (gateway/)

**What it does:** Runs 3 classifiers in parallel on every incoming query.
Blocks threats before they reach the LLM.

**Stack:** FastAPI + guardrails-ai (`on_fail="noop"`) + SQLite logging.
Runs on CPU on a MacBook.

**Decision engine:**
```
Gateway Score = (PI_weight × PI_score) + (PII_weight × PII_score) + (JB_weight × JB_score)
Weights: PI=0.3, PII=0.3, JB=0.4

Score > 0.7  → BLOCK  (return error, log)
0.3–0.7      → ESCALATE (flag + log + pass through with warning)
Score < 0.3  → PASS (continue to LLM)
```

**The 3 classifiers:**
- `Llama-Prompt-Guard-2-86M` — prompt injection, fine-tuned on 3 combined datasets
- `RoBERTa-base NER` — PII detection, 57 entity types, fine-tuned on ai4privacy/pii-masking-200k
- `RoBERTa-base` — jailbreak detection, fine-tuned on JailBreakV-28k + Databricks Dolly-15k

**Known limitations (document in paper):**
- Jailbreak model: weak on fictional framing and indirect multi-step attacks
- `on_fail="noop"` is intentional — DecisionEngine (not guardrails-ai) makes final call

---

## Phase 2 — RAG Pipeline (rag/)

**What it does:** Retrieves relevant regulatory document chunks to ground the
MAD pipeline's debate. NOT used to answer the user's query — used to verify
the LLM's answer after the fact.

**Stack:** Qdrant (GCP cloud) + `nvidia/llama-embed-nemotron-8b` embedding model.
No BM25 hybrid. No cross-encoder reranker. Semantic vector search only.

**Corpus (confirmed):**
- Collection: `ai_governance_chunks_nemotron8b`
- Total chunks: 4,664 across ~40 regulatory documents
- Documents include: ISO 27001, GDPR, OWASP LLM cheatsheets, NIST, HIPAA (partial)
- Vector size: 4096 (Nemotron-8B output dimension)
- Distance metric: cosine

**Chunk payload fields (confirmed from ingest notebook):**
```
chunk_id     — string e.g. "t0__iso__27001_2022_infosec__chunk_0000"
doc_id       — string e.g. "t0__iso__27001_2022_infosec"
tier         — string "T0" / "T1" (batch/collection label — NOT authority ranking)
text         — string (chunk content, ingested without prefix)
source       — dict {"document_name": "...", "pdf_url": "..."}
source_url   — string
chunk_index  — int
token_count  — int
chunk_method — string
```

**Critical note on tier:** The `tier` field stores batch labels (`"T0"`, `"T1"`),
NOT document authority rankings. OWASP and ISO 27001 are both `"T0"`. GDPR is
`"T1"`. These labels have no authority meaning and are NOT used for filtering in
MAD. Do not use `tier` for any authority-based logic — it will give wrong results.

**Embedding at query time:** Queries MUST be prefixed with the instruction prefix
before embedding. Chunks were ingested WITHOUT any prefix.
```
Query prefix: "Instruct: Retrieve relevant regulatory passage to answer the query\nQuery: "
```

**rag/ folder files:**
```
rag/
├── __init__.py
├── config.py      — Qdrant URL, collection, model name, query prefix, HF token
├── embedder.py    — Loads Nemotron-8B (lazy singleton), embed_query() / embed_texts()
├── retriever.py   — retrieve(query, top_k) → List[EvidenceChunk], health_check()
└── README.md      — Setup, activation steps, test commands
```

**To activate real Qdrant in MAD** — one change in `multi_agent/rag_stub.py`:
```python
def retrieve(query: str, top_k: int = TOP_K_CHUNKS) -> List[EvidenceChunk]:
    from rag.retriever import retrieve as _real
    return _real(query, top_k)
```

**To verify connection before running MAD:**
```bash
export QDRANT_API_KEY="your-key"
export HF_TOKEN="your-hf-token"
python3 -c "from rag.retriever import health_check; print(health_check())"
# Expected: {'status': 'ok', 'points_count': 4664}
```

**When NOT using Qdrant:** `multi_agent/rag_stub.py` falls back to built-in
Healthcare sample chunks (HIPAA, ADA, HITECH, GDPR) automatically if Qdrant
is not activated. This is the default state for local development.

**CHUNKS_JSONL_PATH** — set this env var if you have a local JSONL export of
the corpus. The stub loads it instead of using the built-in samples.

---

## Phase 3 — MAD Output Guardrail (multi_agent/) ← ACTIVE DEVELOPMENT

**What it does:** After the LLM generates an answer, MAD verifies every claim
in that answer against the regulatory corpus. It flags hallucinations, missing
caveats, jurisdiction errors, and unsupported regulatory statements.

**LLM runtime:** Ollama with `qwen2.5:7b`. Uses OpenAI-compatible API.
Apple Metal GPU used automatically on MacBook.

**See:** `multi_agent/IMPLEMENTATION.md` for the complete implementation guide.

### The user query vs regulatory corpus distinction

The user NEVER asks about regulations. They ask operational questions:
- "Can we share this patient's records with our billing partner?"
- "Does HIPAA require us to encrypt ePHI on our servers?"
- "How quickly must we respond to a patient records request?"

The enterprise LLM answers. MAD then verifies: does the answer accurately
represent what the regulations actually say? The corpus is MAD's reference
library, not a query-answering tool.

### MAD file structure

```
multi_agent/
├── __init__.py
├── config.py            ← all env-var-overridable settings
├── models.py            ← Pydantic schemas (Claim, Challenge, JudgeVerdict, MADOutput)
├── rag_stub.py          ← JSONL-backed retriever (swap body for Qdrant here)
├── claim_extractor.py   ← LLM extracts atomic claims from LLM answer
├── agent_a.py           ← Ground Truth Verifier (verify + revise)
├── agent_b.py           ← Adversarial Auditor (asymmetric RAG, no web search)
├── judge.py             ← Partially blind Judge (confidence stripped)
├── debate_engine.py     ← 2-cycle orchestrator + SQLite storage writes
├── mad_pipeline.py      ← Top-level entry point
├── storage.py           ← SQLite write functions for all 4 GRPO tables
├── api.py               ← FastAPI POST /mad/verify
├── run_test.py          ← CLI test runner with Healthcare examples
└── IMPLEMENTATION.md    ← Full implementation guide
```

### Key design decisions (do not change without understanding why)

| Decision | Value | Why |
|----------|-------|-----|
| Agent B has RAG access | ✅ Yes — asymmetric queries | Without RAG, B uses parametric knowledge = unreliable for compliance |
| Agent B uses web search | ❌ No — removed | Uncontrolled sources, non-reproducible, paper reviewers will challenge |
| Agent B challenge types | 3: CHUNK_CURRENCY, JURISDICTION_SCOPE, EXCEPTION_EXISTENCE + GAP_FINDING | TIER_OVERRIDE removed — corpus T0/T1 labels are not authority rankings |
| Judge sees confidence scores | ❌ No — stripped | Prevents anchoring on Agent A's confidence |
| Judge sees B's challenge text | ❌ No — only evidence pool | Prevents argumentative bias |
| Agent A can change claim text | ❌ No — only verdict/confidence | LLM's claim is the ground truth being evaluated |
| Max debate cycles | 2 | Research: marginal returns plateau after 3 rounds |
| Routing hard rule | is_material + v=0.0 → HARD_BLOCK | Fabricated regulatory standards must never reach users |
| Aggregate: material claims | min() | Weakest link — safety critical |

---

## Phase 4 — Confidence Scoring Engine (confidence/)

**Status:** Pending — Phase 2 work.

**Formula (full):**
```
final_score = 0.30 × F_llm + 0.25 × (1 − H_llm) + 0.10 × relevancy + 0.35 × judge_eval_score
```

**Current implementation (v0.1):** Uses judge-only aggregate (min of material
claim judge scores). DeepEval Layer 1 integration pending.

**IMPORTANT for the paper:** Report results as "judge-only aggregate (v0.1)"
— do not claim the full CSE formula is implemented.

**LangSmith note:** LangSmith is for tracing/observability only. The CSE
formula computation stays in Python. LangSmith wraps around it with
`@traceable` decorators for debugging.

---

## Phase 5 — GRPO Feedback Loop (rlhf/)

**Status:** Pending — Phase 3 work. Depends on MAD storage being correct.

**Training target:** Agent A only (Phase 1). Agent B fine-tuning is Phase 2.

**What it reads from SQLite:** The 4 tables written by `multi_agent/storage.py`:
- `queries` — one row per pipeline run
- `claims` — 3 rows per claim (post_step_A, post_cycle1, post_cycle2)
  - `agent_a_prompt` column = GRPO training input (system + query + chunks + claim)
- `attacks` — one row per Agent B challenge
  - `b_reward` column = written by feedback loop (NOT by MAD)
- `judge_verdicts` — one row per claim after Judge runs
  - `v_label` (1.0/0.5/0.0) = ground truth for Brier reward

**Reward formulas:**
```python
# Agent A — Brier score reward (calibration)
brier_reward = 2 * p_final * v_label - p_final ** 2

# Agent B — Precision score
b_reward = +1  # if attacked wrong claim (v=0/0.5) AND delta_p >= 0.2
b_reward = -1  # if attacked correct claim (v=1.0) — gaslighting penalty
b_reward =  0  # if attack had no meaningful effect
```

**is_clean filter:**
```python
is_clean = 1 if b_reward != -1 else 0
# b_reward = -1 means B gaslit A — A lowered confidence on a valid claim.
# That corrupted confidence must not be used to train A.
# Only is_clean = 1 records go into the TRL GRPO trainer.
```

**GRPO advantage:**
```python
grpo_advantage = rollout_total - mean_across_rollouts
# rollout_total = sum of brier_reward for clean claims in this rollout
# Run same query multiple times (multiple rollout_ids) to get the mean
```

**Critical rule:** MAD code NEVER writes `b_reward`, `brier_reward`, or the
`rewards` table. Those are exclusively the feedback loop's responsibility.

---

## Synthetic evaluation dataset (pending)

**Status:** Pending — build after MAD pipeline is verified running.

**Spec:**
- 200 examples
- Healthcare domain only
- Generated via Claude API (NOT Qwen2.5 — avoids circular evaluation)
- 4 error types per example:
  - `fully_correct` → expected routing: DELIVER
  - `missing_caveat` → expected routing: RETRY
  - `hallucinated_specific` → expected routing: HARD_BLOCK
  - `jurisdiction_blind` → expected routing: HARD_BLOCK or RETRY
- Schema: `{query, domain, llm_answer, error_type, claims:[{claim, ground_truth_verdict}], expected_routing}`
- Human-labelled judge verdicts (v=1.0/0.5/0.0) required per claim

---

## Environment variables

```bash
# LLM (MAD agents)
AGENT_MODEL=qwen2.5:7b
JUDGE_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434/v1

# RAG — Qdrant (required to use real retriever)
QDRANT_URL=https://e2e7b7d2-4927-4c61-a78d-61f9c4e024bb.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=                          # set this — never commit to git
QDRANT_COLLECTION=ai_governance_chunks_nemotron8b
HF_TOKEN=                                # required if nvidia/llama-embed-nemotron-8b is gated

# RAG — local JSONL fallback (optional — only if not using Qdrant)
CHUNKS_JSONL_PATH=/path/to/chunks.jsonl  # falls back to built-in Healthcare samples

# Debate
MAX_CYCLES=2
TOP_K_CHUNKS=5
TOP_K_CHALLENGE_CHUNKS=3

# Routing
CONFIDENCE_THRESHOLD_HIGH=0.8
CONFIDENCE_THRESHOLD_LOW=0.4

# Storage
MAD_DB_PATH=/path/to/mad_store.db        # default: repo_root/mad_store.db

# API
MAD_API_HOST=0.0.0.0
MAD_API_PORT=8001
```

Note: `TAVILY_API_KEY` was removed. Agent B uses asymmetric RAG only — no web search.

---

## How to run (quick reference)

```bash
# 1. Pull model and start Ollama
ollama pull qwen2.5:7b
ollama serve                            # keep running in separate terminal

# 2. Install deps
pip install -r multi_agent/requirements.txt

# 3. Run Healthcare test (HIPAA encryption example)
python -m multi_agent.run_test

# 4. Verify SQLite storage was written correctly
python -m multi_agent.run_test --verify-storage

# 5. Run specific Healthcare example
python -m multi_agent.run_test --example hipaa_phi_sharing
python -m multi_agent.run_test --list    # see all examples

# 6. Start FastAPI service
uvicorn multi_agent.api:app --port 8001 --reload
# Swagger UI: http://localhost:8001/docs

# 7. Activate real Qdrant (when ready)
export QDRANT_API_KEY="your-key"
export HF_TOKEN="your-hf-token"
python3 -c "from rag.retriever import health_check; print(health_check())"
# Then in multi_agent/rag_stub.py replace retrieve() body with:
# from rag.retriever import retrieve as _real; return _real(query, top_k)
```

---

## Paper contributions (novel — no prior work found)

1. MAD as output guardrail for enterprise AI (not answer generation)
2. Two-phase input + output guardrail pipeline (model-agnostic, domain-agnostic)
3. Asymmetric RAG access between debate agents (DRAG paper pattern)
4. Gap-finding obligation search combined with CFMAD critic pattern
5. Brier score reward for calibrated confidence training via GRPO

Note: TIER_OVERRIDE challenge type was designed but removed in v0.1 because
corpus tier metadata (T0/T1) are batch labels, not authority rankings. Will be
re-added as a contribution once corpus has reliable authority_level per document.

---

## What is NOT implemented yet (be honest in paper)

- DeepEval Layer 1 (F_llm, H_llm, relevancy) — full CSE formula is Phase 2
- GRPO feedback loop — Phase 3
- Agent B fine-tuning — Phase 2
- Synthetic evaluation dataset — pending MAD pipeline verification
- LangSmith tracing decorators — optional observability, not blocking

---

## Research papers informing the design

| Paper | What we took |
|-------|-------------|
| CFMAD (COLING 2025) | Agent A verifier role, Agent B skeptic, third-party Judge |
| DPD (ICIC 2025) | True→Skeptic transition rule, 4 challenge types |
| DRAG paper | Asymmetric RAG between agents |
| MAD-Sherlock (ICWSM 2025) | RAG-grounded output verification |
| TTS/Snell et al. (ICLR 2025) | Claim-level confidence scoring |
| Behaviorally Calibrated RL (ByteDance/CMU 2025) | Brier score reward, verbalized confidence |
| HalluLens (arXiv 2504) | Llama-3.1-70B as G-Eval judge |
| DeepSeek-R1 (arXiv 2501) | TRL/GRPO for RL training |
