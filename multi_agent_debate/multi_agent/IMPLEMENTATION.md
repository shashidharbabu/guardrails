# MAD Pipeline — Complete Implementation Guide
## multi_agent/ · Guardrails Gateway · SJSU CS298B · 2025–26

---

## Table of Contents

1. [What MAD does — the one-paragraph version](#1-what-mad-does)
2. [The user query vs regulatory corpus distinction](#2-user-query-vs-corpus)
3. [Full pipeline flow](#3-full-pipeline-flow)
4. [What each file does](#4-what-each-file-does)
5. [What each agent does — step by step](#5-what-each-agent-does)
6. [Asymmetric RAG — the core design decision](#6-asymmetric-rag)
7. [Storage layer — how data is written for GRPO](#7-storage-layer)
8. [Routing decision logic](#8-routing-decision-logic)
9. [Data flow for GRPO feedback loop](#9-grpo-data-flow)
10. [Key design decisions and why](#10-key-design-decisions)
11. [Connecting real Qdrant](#11-connecting-qdrant)
12. [Running and testing](#12-running-and-testing)
13. [What is not implemented yet](#13-not-implemented-yet)

---

## 1. What MAD does

After the enterprise LLM generates an answer to a user query, MAD verifies every
claim in that answer against a curated regulatory corpus. It flags hallucinations,
missing caveats, wrong jurisdiction assumptions, and fabricated regulatory standards
before the answer reaches the user.

MAD does NOT answer the user's query. It verifies the LLM's answer.

---

## 2. User query vs regulatory corpus

The user never asks about regulations. They ask operational questions:

```
User: "Does HIPAA require us to encrypt ePHI on our servers?"
LLM:  "Yes. HIPAA mandates AES-256 encryption for all ePHI at rest."

MAD checks this answer against HIPAA corpus:
  Claim 1: "HIPAA mandates encryption of ePHI"
           → PARTIAL (encryption is ADDRESSABLE, not mandatory under 45 CFR 164.312)
  Claim 2: "AES-256 is the required standard"
           → NOT_SUPPORTED (AES-256 not named in HIPAA — NIST recommendation only)

  Judge scores:
  C1: v=0.5  (partially correct but overstates obligation)
  C2: v=0.0  (fabricated — AES-256 mandate does not exist in HIPAA)

  Routing: HARD_BLOCK (C2 is is_material + v=0.0)
  Correction signal: "HIPAA 45 CFR 164.312 lists encryption as an
  addressable specification — entities must assess whether it is
  reasonable and appropriate, not implement it mandatorily..."
```

The regulatory corpus is MAD's reference library. It is retrieved during the
debate to ground challenges and verdicts. It is not used to answer queries.

---

## 3. Full pipeline flow

```
Enterprise Employee Query
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  GATEWAY (gateway/)                                      │
│  Prompt Injection │ PII Detection │ Jailbreak Detection  │
│  Decision Engine: Score = PI + PII + JB                  │
│  > 0.7 → BLOCK    0.3-0.7 → ESCALATE    < 0.3 → PASS    │
└─────────────────────────┬────────────────────────────────┘
                          │ PASS / ESCALATE
                          ▼
┌──────────────────────────────────────────────────────────┐
│  ENTERPRISE LLM (model-agnostic)                         │
│  Generates answer cold — no RAG context                  │
│  Qwen2.5-3B for benchmarking                             │
└─────────────────────────┬────────────────────────────────┘
                          │ raw answer (unverified)
                          ▼
┌──────────────────────────────────────────────────────────┐
│  MAD OUTPUT GUARDRAIL (multi_agent/)                     │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 1. Claim extraction — one LLM call               │    │
│  │    Each claim: {claim_id, claim_text, is_material}│    │
│  └─────────────────────────────────────────────────┘    │
│                          │                              │
│  ┌──────────────┐        │                              │
│  │ Layer 1      │        │ Runs in parallel →           │
│  │ (async)      │        ▼                              │
│  │ Faithfulness │  ┌──────────────────────────────┐    │
│  │ Hallucination│  │ CYCLE 1                       │    │
│  │ Relevancy    │  │  Step A: Agent A verifies     │    │
│  │   → CSE      │  │         (symmetric RAG)       │    │
│  └──────────────┘  │  Step B: Agent B challenges   │    │
│                    │         (asymmetric RAG)       │    │
│                    │  Step C: Agent A revises       │    │
│                    │         (verdicts only)        │    │
│                    │                               │    │
│                    │ CYCLE 2                       │    │
│                    │  B: follow-up challenges      │    │
│                    │  A: final revised report      │    │
│                    └──────────────────────────────┘    │
│                          │                              │
│                          ▼                              │
│            Evidence Pool (both agents, unlabelled)      │
│                          │                              │
│                          ▼                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Judge Agent (partially blind)                    │  │
│  │ SEES: final verdicts + evidence pool + query     │  │
│  │ DOES NOT SEE: confidence scores, B's challenges  │  │
│  │ Scores: v = 1.0 / 0.5 / 0.0 per material claim  │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│  Layer 2: G-Eval (Llama-3.1-70B) → judge_eval_score    │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  CONFIDENCE SCORING ENGINE (confidence/) [Phase 2]       │
│  final = 0.30×F_llm + 0.25×(1−H_llm) + 0.10×R          │
│        + 0.35×judge_eval_score                           │
│  Current v0.1: judge-only aggregate                      │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  ROUTING DECISION                                        │
│                                                          │
│  Hard rule (checked FIRST):                              │
│  is_material + v=0.0 → HARD_BLOCK → human review        │
│                                                          │
│  Soft rules:                                             │
│  score > 0.8  → DELIVER to user                         │
│  score 0.4-0.8 → RETRY with correction signal           │
│  score < 0.4  → HUMAN_REVIEW                            │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  SQLite Storage (mad_store.db)                           │
│  All debate data → GRPO feedback loop (rlhf/)            │
└──────────────────────────────────────────────────────────┘
```

---

## 4. What each file does

| File | Role | Key function |
|------|------|-------------|
| `config.py` | All settings | env-var overridable, no secrets hardcoded |
| `models.py` | Pydantic schemas | `Claim`, `Challenge`, `JudgeVerdict`, `MADOutput` |
| `rag_stub.py` | RAG retriever | `retrieve(query, top_k, min_tier, max_tier)` |
| `claim_extractor.py` | Extracts atomic claims | `extract_claims(query, llm_answer)` |
| `agent_a.py` | Ground Truth Verifier | `verify_claims()`, `revise_verdicts()` |
| `agent_b.py` | Adversarial Auditor | `challenge_claims()` |
| `judge.py` | Partially blind Judge | `judge_claims()` |
| `debate_engine.py` | 2-cycle orchestrator | `run_debate()`, `build_transcript()` |
| `storage.py` | SQLite writer | 4 tables for GRPO training data |
| `mad_pipeline.py` | Entry point | `run_mad(query, llm_answer) → MADOutput` |
| `api.py` | FastAPI service | `POST /mad/verify` |
| `run_test.py` | CLI test runner | 5 Healthcare examples + storage verification |

---

## 5. What each agent does — step by step

### Claim Extractor (`claim_extractor.py`)

Called once before the debate starts. LLM reads the enterprise LLM's answer and
breaks it into atomic verifiable statements. Each statement should express exactly
one fact — one regulation, one number, one obligation.

```
Input:  query="Does HIPAA require encryption of ePHI?"
        llm_answer="Yes. HIPAA mandates AES-256 encryption..."

Output: [
  Claim(id=1, text="HIPAA mandates encryption of ePHI",           is_material=True,  p=0.70)
  Claim(id=2, text="AES-256 is the required encryption standard",  is_material=True,  p=0.70)
  Claim(id=3, text="Non-compliance faces automatic $50k penalties", is_material=True, p=0.70)
]
```

`is_material=True` means: this claim involves a regulatory obligation, penalty,
threshold, or specific standard. These trigger hard-block if the Judge scores 0.0.

---

### Agent A — Ground Truth Verifier (`agent_a.py`)

**Step A — initial verification (cycle 1 only):**

1. Calls `rag_stub.retrieve(claim_text)` for each claim — confirmatory retrieval
   (the query IS the claim, so RAG returns supporting chunks)
2. Passes all claims + retrieved chunks to `qwen2.5:7b` in one batched LLM call
3. Gets back: `verdict`, `confidence p`, `reasoning`, `evidence_chunk_ids` per claim
4. Builds a per-claim prompt string for GRPO storage — the exact text Agent A saw

```
Temperature: 0.1 (near-deterministic — verification must be consistent across rollouts)
```

**Step C — revision (every cycle):**

1. Receives Agent B's challenges + chunks B retrieved
2. Calls LLM: "here are your verdicts, here are B's challenges, should you revise?"
3. Updates: `verdict`, `confidence p`, `reasoning`, `evidence_chunks` only
4. CANNOT change `claim_text` — the LLM's original claim is the ground truth

Key constraint: the per-claim prompts at post_cycle1 = base_prompt + B's challenge
text appended. At post_cycle2 = post_cycle1 prompt + cycle 2 challenge appended.
This growing prompt is what TRL GRPO reads as training input.

---

### Agent B — Adversarial Auditor (`agent_b.py`)

Called once per cycle after Agent A's step A. Does NOT verify claims — attacks them.

**Asymmetric RAG retrieval (per SUPPORTED claim, 3 queries):**
```python
retrieve(f"amendment update superseded revision {claim.claim_text}")     # CHUNK_CURRENCY
retrieve(f"jurisdiction scope does not apply {claim.claim_text}")        # JURISDICTION_SCOPE
retrieve(f"exception carve-out safe harbour alternative {claim.claim_text}")  # EXCEPTION_EXISTENCE
```

**Tier-aware retrieval (per SUPPORTED claim where A cited tier-2/3 evidence):**
```python
retrieve(f"primary legislation statute {claim.claim_text}", min_tier=1, max_tier=1)
# Fetches ONLY tier-1 primary law to check if it contradicts A's guidance citation
```

**Gap-finding retrieval (topic-level, always runs):**
```python
retrieve(f"also required additionally must {query}")
retrieve(f"prerequisite condition before permitted {query}")
```

**Challenge types Agent B can raise:**
- `CHUNK_CURRENCY` — cited evidence is outdated / superseded
- `JURISDICTION_SCOPE` — regulation doesn't apply to this context
- `EXCEPTION_EXISTENCE` — a carve-out or safe harbour applies
- `TIER_OVERRIDE` — primary law (tier 1) contradicts what A cited from guidance (tier 2/3)
- `GAP_FINDING` — LLM answer missed an entire regulatory obligation (claim_id=0)

**True→Skeptic rule:** When Agent A says SUPPORTED on any claim, Agent B is
required to attempt all 4 challenge types before accepting it. This prevents
early collapse into agreement.

**What Agent B does NOT do:**
- No web search (removed — uncontrolled, non-reproducible)
- No changing claim text
- No fabricating chunk IDs
```
Temperature: 0.2 (slightly exploratory — B needs to find adversarial angles)
```

---

### Judge Agent (`judge.py`)

Runs after both debate cycles complete. Evaluates claims independently against
the evidence pool.

**What the Judge SEES:**
- User query
- Agent A's final revised verdicts (text + reasoning)
- Evidence pool — all chunks from both agents, merged, with NO agent labels

**What the Judge DOES NOT SEE:**
- Confidence scores (stripped before building the prompt — prevents anchoring)
- Agent B's challenge text or argumentative framing
- Which agent retrieved which chunk

**Output:**
- `v_label` per material claim: `1.0` (fully supported) / `0.5` (partial) / `0.0` (unsupported)
- `correction_signal`: a self-contained explanation of what was wrong and what
  the correct regulatory position is — sent to the LLM on retry

```
Temperature: 0.0 (fully deterministic — scoring must be reproducible across runs)
```

---

## 6. Asymmetric RAG — the core design decision

Both agents query the same corpus. Different query intent. That creates genuine debate.

```
SAME QUERY:   "Does HIPAA mandate encryption of ePHI at rest?"

AGENT A                                AGENT B
Query: "HIPAA encryption ePHI"         Query: "exception to HIPAA encryption addressable"
→ Returns: chunks confirming           → Returns: chunks showing it's addressable

RESULT:                                RESULT:
hipaa_164_312_technical_safeguards     hhs_ocr_encryption_guidance_2023
  "implement a mechanism to encrypt"     "HIPAA Security Rule does NOT mandate
   (supports encryption as a measure)     encryption. Addressable specification."

Same corpus. Different intent. Real tension.
```

Without asymmetric RAG:
- Both agents see the same chunks
- Both agents agree on the same verdict
- You have self-consistency checking, not a debate
- You CANNOT catch the most dangerous failure: a claim that is
  internally consistent but factually wrong relative to the regulation

---

## 7. Storage layer — what is written and when

The storage layer writes 4 SQLite tables at every stage of the pipeline.
This data is what the GRPO feedback loop (rlhf/) reads to train Agent A.

### Write sequence (exact order per spec)

```
1. Query enters MAD
   → storage.write_query(query_id, rollout_id, query_text, llm_answer, [])
   → queries table: routing_decision=NULL, final_cse_score=NULL

2. Agent A Step A (initial verify)
   → storage.write_claims_checkpoint(checkpoint="post_step_A", per_claim_prompts={...})
   → claims table: 1 row per claim
   → agent_a_prompt = "system + query + RAG chunks + claim text"

3. Agent B Cycle 1 challenges
   → storage.write_attacks(cycle=1, p_before_attack=A's current p, p_after=NULL)
   → attacks table: 1 row per challenge

4. Agent A Step C revision (Cycle 1)
   → storage.write_claims_checkpoint(checkpoint="post_cycle1", per_claim_prompts={...})
   → claims table: 1 NEW row per claim (never update existing rows)
   → agent_a_prompt = "step A prompt + B's cycle 1 challenge"
   → storage.update_attack_p_after(cycle=1)
   → attacks table: UPDATE p_after_attack (only time MAD updates existing rows)

5. Agent B Cycle 2 challenges
   → storage.write_attacks(cycle=2, p_before=A's revised p, p_after=NULL)

6. Agent A Cycle 2 final
   → storage.write_claims_checkpoint(checkpoint="post_cycle2", per_claim_prompts={...})
   → agent_a_prompt = "post_cycle1 prompt + B's cycle 2 challenge"
   → storage.update_attack_p_after(cycle=2)

7. Judge runs
   → storage.write_judge_verdicts(v_label per claim)
   → judge_verdicts table: last thing MAD writes

8. Routing computed
   → storage.update_query_cse(final_cse_score, routing_decision)
   → queries table updated with final values

9. [FEEDBACK LOOP — NOT MAD]
   → attacks.b_reward computed and written
   → rewards table created and written
```

### The agent_a_prompt column — why it's critical

TRL GRPO training needs two things per training example:
- **INPUT**: the exact prompt Agent A received at this checkpoint
- **REWARD**: the Brier score that resulted from Agent A's output

Without `agent_a_prompt`, TRL has no input to train on. It cannot reconstruct
what Agent A was responding to from the confidence number alone.

```python
# post_step_A prompt (stored in claims.agent_a_prompt):
"System: You are Agent A — a regulatory compliance expert...
 User query: Does HIPAA require us to encrypt ePHI?
 Retrieved evidence:
 [hipaa_164_312_technical_safeguards] (tier 1): 45 CFR 164.312 — encryption
 and decryption (Addressable): implement a mechanism to encrypt...
 Claim to verify: HIPAA mandates AES-256 encryption
 Output verdict and confidence:"

# post_cycle1 prompt (= above + B's challenge appended):
"[...above...]
 --- Agent B Cycle 1 Challenge ---
 [CHUNK_CURRENCY]: hhs_ocr_encryption_guidance_2023 shows 2023 HHS OCR
 guidance clarifies HIPAA encryption is addressable — AES-256 not mandated.
 Review B's challenge. If B found a genuine gap, lower confidence.
 Output your revised verdict and confidence:"
```

### SQLite table schemas

```sql
-- TABLE 1: one row per pipeline run
CREATE TABLE queries (
    query_id         TEXT,
    rollout_id       TEXT,
    query_text       TEXT,
    llm_answer       TEXT,
    rag_chunk_ids    TEXT,        -- JSON array
    timestamp        TEXT,
    final_cse_score  REAL,        -- NULL until routing computed
    routing_decision TEXT,        -- NULL until routing computed
    PRIMARY KEY (query_id, rollout_id)
);

-- TABLE 2: 3 rows per claim (3 checkpoints) — NEVER UPDATE, always INSERT
CREATE TABLE claims (
    record_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id        TEXT,
    rollout_id      TEXT,
    claim_id        TEXT,
    claim_text      TEXT,         -- locked after post_step_A, never changes
    is_material     INTEGER,      -- 1 or 0
    confidence_p    REAL,         -- Agent A's confidence AT THIS checkpoint
    verdict         TEXT,
    checkpoint      TEXT,         -- "post_step_A" / "post_cycle1" / "post_cycle2"
    agent_a_prompt  TEXT,         -- GRPO training input — grows at each checkpoint
    evidence_chunks TEXT,         -- JSON array of chunk_ids
    reasoning       TEXT,
    timestamp       TEXT
);

-- TABLE 3: one row per Agent B challenge
CREATE TABLE attacks (
    attack_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id         TEXT,
    rollout_id       TEXT,
    claim_id         TEXT,
    cycle            INTEGER,     -- 1 or 2
    b_critique_text  TEXT,
    b_challenge_type TEXT,
    p_before_attack  REAL,        -- Agent A's p BEFORE this attack
    p_after_attack   REAL,        -- NULL until Agent A revises (step C)
    agent_b_prompt   TEXT,        -- NULL — Phase 2 (Agent B training)
    timestamp        TEXT,
    b_reward         REAL         -- NULL — feedback loop writes this
);

-- TABLE 4: one row per claim after Judge runs
CREATE TABLE judge_verdicts (
    query_id          TEXT,
    rollout_id        TEXT,
    claim_id          TEXT,
    v_label           REAL,       -- 1.0 / 0.5 / 0.0
    judge_reasoning   TEXT,
    evidence_chunk_ids TEXT,      -- JSON array
    timestamp         TEXT,
    PRIMARY KEY (query_id, rollout_id, claim_id)
);
```

---

## 8. Routing decision logic

```python
# HARD RULE — checked first, overrides everything
for claim in judge_verdicts:
    if claim.is_material and claim.v_label == 0.0:
        return "HARD_BLOCK"   # fabricated regulatory standard — never retry

# AGGREGATE SCORE — after hard rule passes
aggregate = 0.70 * min(v for v in material_scores) + 0.30 * mean(non_material_scores)

if aggregate >= 0.8:  return "DELIVER"        # verified — send to user
if aggregate >= 0.4:  return "RETRY"          # send correction signal to LLM
else:                 return "HUMAN_REVIEW"   # too uncertain — queue for expert
```

**Why min() for material claims?** Compliance is safety-critical. If one
material claim scores 0.5 (partially wrong), the whole answer is potentially
misleading. The weakest link dominates. A chain is only as strong as its weakest link.

**Why HARD_BLOCK before aggregate?** If a regulatory claim scores 0.0 — meaning
the LLM fabricated a specific standard, fine amount, or article that doesn't exist —
no level of aggregate confidence makes it safe to deliver. A 0.0 material claim
bypasses ALL threshold logic.

---

## 9. GRPO data flow

How the rlhf/ feedback loop uses the storage data:

```
1. Read claims (post_cycle2) + judge_verdicts
   → compute Brier reward per claim:
     brier_reward = 2 * p_final * v_label - p_final^2

2. Read attacks
   → compute Agent B precision reward per attack:
     delta_p = p_before_attack - p_after_attack
     if v_label in (0.0, 0.5) and delta_p >= 0.2:  b_reward = +1  (valid challenge)
     if v_label == 1.0:                              b_reward = -1  (gaslighting)
     else:                                           b_reward = 0

3. Apply is_clean filter
   → is_clean = 1 if b_reward != -1 else 0
   → Claims where b_reward = -1: Agent B gaslit Agent A into lowering confidence
     on a valid claim. That corrupted confidence value must not train Agent A.
   → Only is_clean = 1 claims go into TRL GRPO trainer

4. Compute GRPO advantage
   → Run same query N times (different rollout_id each time)
   → rollout_total = sum(brier_reward for clean claims in this rollout)
   → grpo_advantage = rollout_total - mean(rollout_total across all rollouts)

5. TRL GRPO training
   → Input:  agent_a_prompt (from claims table)
   → Reward: brier_reward × is_clean (from rewards table)
   → Updates Agent A's weights to produce better-calibrated confidence scores
```

**MAD code rule:** NEVER write `b_reward`, `brier_reward`, `is_clean`,
`grpo_advantage`, or the `rewards` table. These are exclusively the feedback
loop's responsibility.

---

## 10. Key design decisions and why

### Why no web search for Agent B?
Web search was removed from Agent B. Reasons:
1. Uncontrolled sources — a blog post about HIPAA can be wrong or outdated
2. Non-reproducible — same query returns different results on different days
3. Paper reviewers will challenge: "how do you control for web source quality?"
4. Asymmetric RAG over your curated corpus achieves the same adversarial goal
   with full reproducibility and controlled authority levels

### Why does the Judge not see confidence scores?
If the Judge sees that Agent A assigned confidence p=0.92 to a claim, it will
anchor on that number when scoring. A claim that Agent A was 92% confident
about will be scored more leniently even if the evidence doesn't fully support it.
Stripping confidence forces the Judge to evaluate the claim against the evidence
independently — which is what makes the Judge score meaningful as a reward signal.

### Why does Agent A return per-claim prompts?
TRL GRPO needs the exact prompt Agent A received at each checkpoint as its
training input. A single shared prompt for all claims would give the GRPO trainer
no way to distinguish between Claim 1's context and Claim 2's context. Each claim
gets its own stored prompt because each claim has different supporting evidence.

### Why 2 debate cycles (not more)?
Research shows Brier score reward calibration improves significantly from 1→2 cycles,
with marginal returns from 2→3. Given the latency cost of each cycle (~30-60s on
MacBook with 7B model), 2 cycles is the right tradeoff for a demo and paper system.
The debate engine supports `max_cycles=N` if you want to experiment.

### Why is claim_text never changed?
The LLM's claim is the subject of evaluation. Changing it would mean you're
evaluating a modified version of what the LLM said — not what it actually said.
If the LLM said "AES-256 is required by HIPAA," the claim text stays exactly that.
Agent A can only update whether that specific claim is supported, not what the claim says.

---

## 11. Connecting real Qdrant

When ready, replace ONLY the body of `retrieve()` in `rag_stub.py`:

```python
# In rag_stub.py — replace the body of retrieve() with this:

from qdrant_client import QdrantClient
from your_rag_module import get_embedding_model  # your Qwen3-4B embedder

_qdrant = QdrantClient(host="localhost", port=6333)
_embed  = get_embedding_model()

def retrieve(
    query:    str,
    top_k:    int = TOP_K_CHUNKS,
    min_tier: int = 0,
    max_tier: int = 99,
) -> List[EvidenceChunk]:
    vector = _embed.encode(query).tolist()

    # Build Qdrant filter for tier if needed
    filter_cond = None
    if min_tier > 0 or max_tier < 99:
        from qdrant_client.models import Filter, FieldCondition, Range
        filter_cond = Filter(must=[
            FieldCondition(key="tier", range=Range(gte=min_tier, lte=max_tier))
        ])

    hits = _qdrant.search(
        collection_name="regulatory_chunks",
        query_vector=vector,
        limit=top_k,
        query_filter=filter_cond,
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

Everything else (agent_a, agent_b, judge, debate_engine, storage) calls
`retrieve()` unchanged. Zero other changes needed.

---

## 12. Running and testing

### Prerequisites
```bash
ollama pull qwen2.5:7b     # ~4.7 GB download
ollama serve               # keep running in a separate terminal
pip install -r multi_agent/requirements.txt
```

### Environment variables
```bash
export CHUNKS_JSONL_PATH=/path/to/rag/chunks.jsonl   # optional — falls back to samples
export MAD_DB_PATH=/path/to/mad_store.db              # optional — default: repo root
```

### Running test cases
```bash
# Run from repo root (not from inside multi_agent/)
python -m multi_agent.run_test                          # HIPAA encryption example
python -m multi_agent.run_test --example hipaa_phi_sharing
python -m multi_agent.run_test --example hipaa_patient_access     # fully_correct
python -m multi_agent.run_test --example hipaa_employee_health_records
python -m multi_agent.run_test --list                   # see all examples

# Save full JSON output (for paper results)
python -m multi_agent.run_test --output-json results.json

# Verify SQLite storage is writing correctly
# Run this before showing the storage to your friend
python -m multi_agent.run_test --verify-storage
```

### Running the API
```bash
uvicorn multi_agent.api:app --port 8001 --reload
# Swagger UI: http://localhost:8001/docs
# Health:     http://localhost:8001/mad/health
# Config:     http://localhost:8001/mad/info
```

### Example API request
```bash
curl -X POST http://localhost:8001/mad/verify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Does HIPAA require us to encrypt ePHI stored on our servers?",
    "llm_answer": "Yes. HIPAA mandates AES-256 encryption for all ePHI at rest."
  }'
```

### Expected response structure
```json
{
  "routing_decision": "HARD_BLOCK",
  "aggregate_confidence": 0.25,
  "correction_signal": "HIPAA 45 CFR 164.312 lists encryption as addressable...",
  "query_id": "abc-123...",
  "rollout_id": "def-456...",
  "claims": [...],
  "judge_verdicts": [
    {"claim_id": 1, "score": 0.5, "is_material": true, ...},
    {"claim_id": 2, "score": 0.0, "is_material": true, ...}
  ],
  "debate_transcript": "=== GUARDRAILS GATEWAY — MAD DEBATE TRANSCRIPT ..."
}
```

### Verifying storage is correct
After running 5 test queries, show your friend this output:
```bash
python -m multi_agent.run_test --verify-storage
```

Expected output:
```
queries table: 1 row(s)
  routing_decision: HARD_BLOCK
  final_cse_score : 0.25

claims table: 6 row(s) (expect 3 per claim)
  C1 [post_step_A  ] p=0.82  SUPPORTED
  C1 [post_cycle1  ] p=0.54  PARTIAL
  C1 [post_cycle2  ] p=0.52  PARTIAL
  C2 [post_step_A  ] p=0.71  SUPPORTED
  ...

agent_a_prompt lengths (should be >0 for all rows):
  ✅ C1 [post_step_A  ] 487 chars
  ✅ C1 [post_cycle1  ] 712 chars   ← longer: includes B's challenge
  ✅ C1 [post_cycle2  ] 934 chars   ← longer: includes cycle 2 challenge
  ...

attacks table: 4 row(s)
  C1 cycle=1  CHUNK_CURRENCY  p_before=0.82  p_after=0.54  b_reward=NULL (expected)
  C2 cycle=1  EXCEPTION       p_before=0.71  p_after=0.41  b_reward=NULL (expected)
  ...

judge_verdicts table: 2 row(s)
  C1  v_label=0.5
  C2  v_label=0.0
```

---

## 13. What is not implemented yet

| Component | Status | Where it lives |
|-----------|--------|---------------|
| DeepEval Layer 1 (F_llm, H_llm, relevancy) | Pending Phase 2 | confidence/ |
| Full CSE formula | Pending Phase 2 | confidence/ |
| GRPO feedback loop | Pending Phase 3 | rlhf/ |
| Agent B fine-tuning | Pending Phase 2 | finetuning/ |
| LangSmith tracing decorators | Optional | multi_agent/ |
| Synthetic evaluation dataset (200 HC examples) | Pending | synthetic_data/ |
| Layer 2 G-Eval integration | Pending Phase 2 | multi_agent/judge.py |

**Paper reporting rule:** Report current results as using
"judge-only aggregate confidence (v0.1)". Do not claim the full CSE formula
is implemented until DeepEval Layer 1 is integrated.
