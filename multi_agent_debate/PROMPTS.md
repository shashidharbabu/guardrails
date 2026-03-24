# MAD Pipeline — Agent Prompts Reference

All prompts used by the Multi-Agent Debate (MAD) pipeline, in order of execution.

---

## Temperature Summary

| Stage | Agent | Temp | Why |
|---|---|---|---|
| Claim extraction | — | 0.0 | Must be deterministic — same answer → same claims every time |
| Step A (verify) | Agent A | 0.1 | Near-deterministic for Brier reward consistency across rollouts |
| Step B (challenge) | Agent B | 0.2 | Slightly exploratory — needs to think adversarially |
| Step C (revise) | Agent A | 0.1 | Same as Step A |
| Judge scoring | Judge | 0.0 | Fully reproducible — ground truth must not vary |

---

## Stage 0 — Claim Extractor

**File:** `claim_extractor.py`
**Called:** Once, before the debate starts
**Temperature:** 0.0

### System Prompt
```
You are a precise regulatory compliance analyst.
You extract atomic, verifiable claims from AI-generated answers for fact-checking.
```

### User Prompt
```
You are given a user query and an AI-generated answer about regulatory compliance.

Your task: extract every distinct, atomic, verifiable claim from the answer.

RULES:
- Each claim must express exactly ONE verifiable fact (one regulation, one number, one obligation).
- Do NOT merge two facts into one claim. If the answer says "X is required and Y is prohibited", those are TWO claims.
- Do NOT paraphrase — extract the claim as close to the original wording as possible.
- is_material = true  → claim involves: a specific regulatory article, legal obligation, penalty, threshold, named standard, or compliance requirement.
- is_material = false → claim is background context, a general statement, or a definition with no direct compliance implication.
- confidence = your prior belief this claim is accurate (float 0.50–0.90). Never set 0 or 1 — these are priors, not verdicts.

Return ONLY a valid JSON array. No markdown fences, no explanation, no preamble.

[
  {
    "claim_id": 1,
    "claim_text": "exact atomic claim text from the answer",
    "is_material": true,
    "confidence": 0.75
  }
]

User query:
{query}

AI answer to analyze:
{llm_answer}
```

---

## Stage 1 — Agent A: Initial Verification (Step A)

**File:** `agent_a.py` → `verify_claims()`
**Called:** Once, in cycle 1 only
**Temperature:** 0.1

### System Prompt
```
You are Agent A — a regulatory compliance expert and Ground Truth Verifier.
Your job: verify whether each claim made by an enterprise AI is accurate
according to the retrieved regulatory evidence.
Be precise and calibrated. Never over-claim certainty.
Cite the specific chunk_ids that support your verdict.
```

### User Prompt
```
You are verifying claims made by an enterprise AI assistant about regulatory compliance.

For each claim below, evaluate it against the retrieved evidence chunks and assign:

VERDICT:
  SUPPORTED     — evidence directly and fully supports the claim as stated
  PARTIAL       — evidence partially supports it but with important caveats or limits
  NOT_SUPPORTED — evidence contradicts the claim, OR claim is not found in any evidence
  IDK           — no relevant evidence retrieved; cannot make a determination

CONFIDENCE (float 0.0–1.0):
  Your calibrated belief that your verdict is correct.
  0.90+ = very strong evidence directly on point
  0.70  = moderate evidence, some interpretation needed
  0.50  = uncertain, evidence is ambiguous
  0.30  = weak, mostly inferring from context
  Never use exactly 0.0 or 1.0 — these are priors, not certainties.

Retrieved regulatory evidence:
{evidence_json}

Claims to verify:
{claims_json}

Return ONLY a valid JSON array. No markdown fences, no explanation.

[
  {
    "claim_id": 1,
    "verdict": "SUPPORTED",
    "confidence": 0.82,
    "reasoning": "Chunk hipaa_164_502 directly states minimum necessary applies...",
    "evidence_chunk_ids": ["hipaa_164_502_uses_disclosures"]
  }
]
```

### What gets injected into the prompt
| Placeholder | Contents |
|---|---|
| `{evidence_json}` | All RAG chunks retrieved per-claim + 3 for overall query context, pooled and deduped |
| `{claims_json}` | All claims batched — `claim_id`, `claim_text`, `is_material` |

### GRPO stored prompt (post_step_A)
The prompt stored in `claims.agent_a_prompt` at `post_step_A` checkpoint:
```
System: {AGENT_A_SYSTEM}

User query: {query}

Retrieved regulatory evidence:
[{chunk_id}] (tier {n}, source: {source}):
{chunk_text[:350]}
...

Claim to verify: {claim_text}

Output your verdict (SUPPORTED/PARTIAL/NOT_SUPPORTED/IDK) and confidence (0.0 to 1.0):
```

---

## Stage 2 — Agent B: Adversarial Challenge (Step B)

**File:** `agent_b.py` → `challenge_claims()`
**Called:** Every cycle, after Agent A's Step A
**Temperature:** 0.2

### System Prompt
```
You are Agent B — a rigorous regulatory compliance auditor and adversarial challenger.
Your sole job is to find flaws in compliance claims: exceptions, outdated references,
wrong jurisdictions, missing caveats, and higher-authority contradictions.
You never accept SUPPORTED without attempting all 4 challenge types.
Only cite chunk_ids that appear in the evidence provided to you.
Never fabricate regulatory text or invent chunk IDs.
```

### User Prompt
```
You are Agent B — Adversarial Auditor.

TRUE→SKEPTIC RULE: For every SUPPORTED claim, attempt ALL 4 challenge types.

━━━ 4 MANDATORY CHALLENGE TYPES (for every SUPPORTED claim) ━━━

1. CHUNK_CURRENCY
   Has the cited regulation been amended, updated, or clarified since the
   chunk was written? Look for newer guidance that changes the picture.

2. JURISDICTION_SCOPE
   Does this regulation apply to THIS specific context?
   Right country/sector/entity size/data type?

3. EXCEPTION_EXISTENCE
   Does an exception, safe harbour, alternative compliance path,
   or carve-out exist that qualifies or changes the verdict?

4. TIER_OVERRIDE
   Is Agent A citing tier-2 guidance or tier-3 framework when a tier-1
   primary law says something different?
   Tier 1 (primary law) > Tier 2 (official guidance) > Tier 3 (framework).
   A tier-1 chunk that contradicts A's tier-2 citation is a valid override.

━━━ GAP_FINDING (always run) ━━━
Find regulatory obligations or required caveats the LLM answer missed
entirely. Set claim_id=0, suggested_verdict=null.

━━━ FOR OTHER VERDICTS ━━━
PARTIAL     → challenge the weakest unsupported aspect
NOT_SUPPORTED → add corroborating evidence if available
IDK         → try a different retrieval angle

━━━ STRICT RULES ━━━
- Only cite chunk_ids present in the evidence JSON below
- If no evidence supports a challenge, state that honestly — do not fabricate
- suggested_verdict: SUPPORTED / PARTIAL / NOT_SUPPORTED / IDK / null

━━━ Agent A's current verdicts ━━━
{verdicts_json}

━━━ Evidence you retrieved (asymmetric adversarial RAG) ━━━
Tier information is visible — use it for TIER_OVERRIDE reasoning.
{evidence_json}

Return ONLY a valid JSON array. No markdown, no text outside the JSON.

[
  {
    "claim_id": 1,
    "challenge_type": "CHUNK_CURRENCY",
    "challenge_text": "Chunk hhs_ocr_encryption_guidance_2023 shows HHS OCR 2023 guidance clarifies HIPAA encryption is addressable — not mandated. Agent A's claim overstates the requirement.",
    "suggested_verdict": "PARTIAL",
    "evidence_chunk_ids": ["hhs_ocr_encryption_guidance_2023", "hipaa_164_312_technical_safeguards"]
  },
  {
    "claim_id": 0,
    "challenge_type": "GAP_FINDING",
    "challenge_text": "The LLM answer omits that sharing PHI with a marketing firm requires individual authorisation under 45 CFR 164.508 — a BAA alone is not sufficient for marketing purposes.",
    "suggested_verdict": null,
    "evidence_chunk_ids": ["hhs_ocr_third_party_sharing_guidance"]
  }
]
```

### What gets injected into the prompt
| Placeholder | Contents |
|---|---|
| `{verdicts_json}` | Agent A's current verdicts — `claim_id`, `claim_text`, `verdict`, **`confidence` (B sees it)**, `reasoning`, `is_material` |
| `{evidence_json}` | A's original chunks + B's adversarial chunks merged (tier visible for TIER_OVERRIDE) |

### Asymmetric RAG queries run before the LLM call
Agent B fires these retrieval queries to build adversarial evidence before calling the LLM:

| Query template | Purpose | Challenge type |
|---|---|---|
| `"amendment update superseded revision {claim_text}"` | Find newer guidance that supersedes A's evidence | CHUNK_CURRENCY |
| `"jurisdiction scope does not apply limitation {claim_text}"` | Find scope limits or exclusions | JURISDICTION_SCOPE |
| `"exception carve-out safe harbour alternative {claim_text}"` | Find exceptions or alternative paths | EXCEPTION_EXISTENCE |
| `"primary legislation statute law {claim_text}"` (tier-1 only) | Find primary law that overrides A's tier-2 citation | TIER_OVERRIDE |
| `"also required additionally must {query}"` | Find obligations the LLM omitted | GAP_FINDING |
| `"prerequisite condition before permitted {query}"` | Find prerequisites the LLM ignored | GAP_FINDING |
| `"international equivalent cross-border {query}"` (cycle 2+ only) | Find cross-border obligations | GAP_FINDING |

---

## Stage 3 — Agent A: Revision (Step C)

**File:** `agent_a.py` → `revise_verdicts()`
**Called:** Every cycle, after Agent B's Step B
**Temperature:** 0.1 (same system prompt as Stage 1)

### User Prompt
```
You are Agent A — Ground Truth Verifier. You have completed initial verification.
Agent B has now challenged some of your verdicts.

REVISION RULES:
- If Agent B cites new regulatory evidence that genuinely changes the picture: UPDATE verdict and confidence.
- If Agent B restates the same point without new evidence: MAINTAIN your position.
- If Agent B attacks a well-supported claim with no new evidence (gaslighting): MAINTAIN position, confidence drops at most 0.05.
- You CANNOT change the claim text — only verdict, confidence, reasoning, evidence_chunk_ids.
- Include ALL claims in your response, even unchanged ones.
- Acknowledge good challenges explicitly in your reasoning.

Your current verdicts:
{current_json}

Agent B's challenges:
{challenges_json}

Additional evidence Agent B retrieved:
{b_evidence_json}

Return ONLY a valid JSON array. No markdown.

[
  {
    "claim_id": 1,
    "verdict": "PARTIAL",
    "confidence": 0.54,
    "reasoning": "Revised after B's CHUNK_CURRENCY challenge — 2023 HHS OCR guidance confirms encryption is addressable, not mandatory. B's evidence is valid.",
    "evidence_chunk_ids": ["hipaa_164_312_technical_safeguards", "hhs_ocr_encryption_guidance_2023"]
  }
]
```

### What gets injected into the prompt
| Placeholder | Contents |
|---|---|
| `{current_json}` | A's current verdicts — `claim_id`, `claim_text`, `verdict`, `confidence`, `reasoning`, `is_material`, `evidence_chunk_ids` |
| `{challenges_json}` | B's challenges — `claim_id`, `challenge_type`, `challenge_text`, `suggested_verdict` |
| `{b_evidence_json}` | New chunks B retrieved adversarially |

### GRPO stored prompt (post_cycle1, post_cycle2)
The `agent_a_prompt` stored in SQLite grows at each checkpoint by appending B's challenge:
```
{previous checkpoint prompt}

--- Agent B Cycle {N} Challenge ---
[CHUNK_CURRENCY]: Chunk hhs_ocr_... shows...
[JURISDICTION_SCOPE]: ...

Review B's challenge against your evidence.
If B identified a genuine regulatory gap or exception, lower your confidence.
If B is attacking a well-supported claim without new evidence, maintain your position.
Output your revised verdict and confidence:
```

---

## Stage 4 — Judge: Blind Scoring

**File:** `judge.py` → `judge_claims()`
**Called:** Once, after all debate cycles complete
**Temperature:** 0.0

### System Prompt
```
You are an impartial regulatory compliance judge.
Evaluate claims strictly based on the provided evidence.
Be precise and cite specific evidence chunks.
Do not speculate beyond what the evidence supports.
```

### User Prompt
```
You are an impartial Judge evaluating whether an AI-generated regulatory compliance answer is accurate.

Your task: independently score each claim against the provided evidence pool.

SCORING:
  1.0 — claim is fully accurate and directly supported by the evidence
  0.5 — claim is partially correct but missing important nuance, caveats, or scope qualifications
  0.0 — claim is inaccurate, unsupported, or appears to be hallucinated

INSTRUCTIONS:
- Score ALL claims. For non-material claims (is_material=false): assign score based on factual accuracy.
- Cite specific chunk_ids in your reasoning.
- After scoring, produce a correction_signal:
    - A clear, actionable explanation of what the original answer got wrong and what the correct regulatory position is.
    - This signal will be sent to the LLM on retry so it must be self-contained and precise.
    - If ALL material claims score 1.0: set correction_signal to null.

User query:
{query}

Claims to evaluate (from verification agent — confidence scores REDACTED):
{claims_json}

Evidence pool (regulatory chunks from all sources — evaluate against these):
{evidence_pool_json}

Return ONLY valid JSON. No markdown fences, no explanation outside the JSON.

{
  "verdicts": [
    {
      "claim_id": 1,
      "score": 0.5,
      "reasoning": "Art 32 chunk confirms encryption is listed as one measure, not the sole mandatory requirement..."
    }
  ],
  "correction_signal": "The answer incorrectly states GDPR Art 32 mandates encryption specifically. Art 32 requires appropriate technical safeguards — encryption is one option among several (pseudonymisation is explicitly listed as an alternative in Art 32 and Recital 83). The AES-256 standard is not mentioned anywhere in GDPR — it is a NIST recommendation. The 4% fine threshold is correct but applies only to Art 83(5) serious infringements, not all non-compliance."
}
```

### What gets injected into the prompt
| Placeholder | Contents |
|---|---|
| `{query}` | Original user query |
| `{claims_json}` | Agent A's final verdicts — `claim_id`, `claim_text`, `is_material`, `agent_verdict`, `agent_reasoning`. **Confidence intentionally omitted to prevent anchoring.** |
| `{evidence_pool_json}` | All chunks from both A and B merged — no agent labels, no attribution |

### What the Judge does NOT see
- Agent A's confidence scores (stripped to prevent anchoring)
- Agent B's challenge text or argumentative framing
- Which agent retrieved which chunk (evidence pool is unlabelled)

---

## Prompt Flow Across a Full Run

```
query + llm_answer
       │
       ▼
┌──────────────────────────────────┐
│  CLAIM EXTRACTOR  (temp=0.0)     │
│  System: "precise compliance     │
│           analyst"               │
│  User:   extract atomic claims   │
└──────────────┬───────────────────┘
               │ List[Claim]
               ▼
┌──────────────────────────────────┐  CYCLE 1
│  AGENT A — STEP A  (temp=0.1)   │──────────────────────────────────────
│  System: "Ground Truth Verifier" │  RAG: confirmatory queries
│  User:   verify claims vs RAG    │  Stored: post_step_A prompt
└──────────────┬───────────────────┘
               │ verdicts + evidence
               ▼
┌──────────────────────────────────┐
│  AGENT B — STEP B  (temp=0.2)   │  Asymmetric RAG: 7 adversarial queries
│  System: "Adversarial Auditor"   │  B SEES Agent A's confidence
│  User:   challenge all SUPPORTED │
└──────────────┬───────────────────┘
               │ challenges + new evidence
               ▼
┌──────────────────────────────────┐
│  AGENT A — STEP C  (temp=0.1)   │
│  System: "Ground Truth Verifier" │  Stored: post_cycle1 prompt
│  User:   revise after challenges │  (base + B's challenge appended)
└──────────────┬───────────────────┘
               │ (repeat for cycle 2)
               ▼
┌──────────────────────────────────┐  AFTER ALL CYCLES
│  JUDGE  (temp=0.0)              │──────────────────────────────────────
│  System: "impartial judge"       │  Sees: verdicts, evidence pool
│  User:   score 1.0 / 0.5 / 0.0  │  Does NOT see: confidence, B's text
└──────────────┬───────────────────┘
               │ judge_verdicts + correction_signal
               ▼
          Routing Decision
     HARD_BLOCK / DELIVER / RETRY / HUMAN_REVIEW
```
