# MAD Pipeline Overview

## What This Is

Multi-Agent Debate (MAD) v3 - a local LLM guardrail system for detecting hallucinations in regulatory/healthcare QA.

## Pipeline Flow

```
User Query
    |
    v
[Baseline LLM] - generates a draft answer
    |
    v
[Decomposer] - extracts atomic claims (JSON: claim_text, is_material, is_critical, confidence_prior)
    |
    v
[RAG Retrieval] - fetches evidence chunks (Qdrant dense + BM25 sparse, hybrid rerank)
    |
    v
For each claim (concurrent, CLAIM_CONCURRENCY in parallel):
    +-- [Agent A] (strict auditor)   Round 0: independent verdict on claim vs evidence
    +-- [Agent B] (adversarial)      Round 0: independent verdict on claim vs evidence
    |
    | confidence_internal is STRIPPED via strip_for_peer() before cross-agent exchange
    |
    +-- [Agent A] Round 1: updated verdict after seeing B verdict+reasoning (no confidence)
    +-- [Agent B] Round 1: updated verdict after seeing A verdict+reasoning (no confidence)
    |
    v
[SQLite DB] - stores agent_outputs (all 4 outputs per claim) + agent_deltas (R0 vs R1)
    |
    v
[Judge] - blind 32B model sees both agents, both rounds, anonymized (Debater 1 / Debater 2)
    |       confidence_internal is NOT shown to judge
    |       output: v_label (0.0=NOT_SUPPORTED, 0.5=PARTIAL, 1.0=SUPPORTED), judge_confidence
    v
[SQLite DB] - stores judge_verdicts
    |
    v
[Export] - optional GRPO JSONL (prompt, completion, v_label, is_critical, judge_confidence)
```

## Agent Roles

**Agent A -- Strict Regulatory Auditor**
- Verifies whether the LLM claim is supported by retrieved evidence
- Finds errors, hallucinations, unsupported claims
- Temperature: 0.4 (less creative, more precise)

**Agent B -- Adversarial Auditor**
- Challenges the claim, looks for what is WRONG, MISSING, or UNSUPPORTED
- Careful skeptic -- acknowledges when claim is well-supported
- Temperature: 0.7 (more exploratory)

**Judge (Blind 32B)**
- Sees both agents Round 0 and Round 1 outputs, anonymized as Debater 1 / Debater 2
- Does NOT see confidence_internal from either agent (stripped by strip_for_peer)
- Renders final verdict: 0.0 / 0.5 / 1.0

## Confidence Stripping

confidence_internal is stored in the database but stripped before:
- Round 1: peer agent sees verdict + reasoning, NOT confidence
- Judge: sees all four outputs without confidence

This prevents agents from anchoring on each other confidence, and prevents judge from being influenced by confidence-signaling.

## Key Design Decisions

1. vLLM for inference - OpenAI-compatible API, no model loading in pipeline code
2. SQLite for storage - simple, portable, no external DB dependency
3. Pydantic schemas - all agent/judge outputs validated before DB write
4. JSON repair loop - if agent returns malformed JSON, a repair prompt is sent before failing
5. Langfuse optional - graceful degradation if keys missing, never crashes
6. Async throughout - all LLM calls are async, claims processed concurrently via asyncio.gather

## Historical Run Stats

- Queries: 50 healthcare/regulatory
- Claims debated: 416
- Judge verdicts: NOT_SUPPORTED=79 (19%), PARTIAL=327 (79%), SUPPORTED=10 (2%)
- Agent models: Qwen2.5-14B-Instruct-AWQ (RTX 5090, 32GB)
- Judge model: Qwen2.5-32B-Instruct (SJSU H100)
