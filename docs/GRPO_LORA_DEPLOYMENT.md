# GRPO LoRA Adapter Deployment Handoff

This document explains how to deploy the GRPO-finetuned MAD debate agents without committing the large LoRA weights to Git.

## Artifact Location

Adapter and evaluation artifacts are shared through Google Drive:

```text
https://drive.google.com/drive/u/4/folders/1NbaX6a8MwDxj4f0_miBnTFbCFvUBvFkv
```

Expected files in that Drive folder:

```text
v4mad+grpo/
├── Agent A Adapters.zip
├── AgentB adapters.zip
└── eval_outputs-20260508T230836Z-3-001.zip
```

Each adapter zip should contain a final LoRA adapter folder with:

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
chat_template.jinja
```

Do not commit `adapter_model.safetensors` directly to GitHub unless the repository is configured for Git LFS. The adapter weight files are large enough that normal GitHub pushes will fail.

## Base Model

Use this base model for both debate agents:

```text
Qwen/Qwen2.5-14B-Instruct
```

The GRPO artifacts are LoRA adapters on top of this base model:

```text
Agent A LoRA -> grpo_agent_a/final
Agent B LoRA -> grpo_agent_b/final
```

## Recommended Deployment Pattern

Use one vLLM server with the base model loaded once and both LoRA adapters registered at server startup.

```text
Base model loaded once:
Qwen/Qwen2.5-14B-Instruct

LoRA adapter 1:
agent_a=/path/to/grpo_agent_a/final

LoRA adapter 2:
agent_b=/path/to/grpo_agent_b/final
```

Then route agent calls by model name:

```text
Agent A calls model="agent_a"
Agent B calls model="agent_b"
```

Do not load adapters from disk per claim. Register both adapters once when the vLLM server starts.

## vLLM Startup

vLLM supports serving LoRA adapters through the OpenAI-compatible server using `--enable-lora` and `--lora-modules name=path`.

Example:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-lora \
  --lora-modules \
    agent_a=/models/grpo_agent_a/final \
    agent_b=/models/grpo_agent_b/final \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

After startup, the backend should call the same vLLM endpoint with different model names:

```python
client.chat.completions.create(
    model="agent_a",
    messages=[...],
    temperature=0.3,
    max_tokens=192,
)

client.chat.completions.create(
    model="agent_b",
    messages=[...],
    temperature=0.5,
    max_tokens=192,
)
```

## AWS SDK Architecture

Recommended cloud flow:

```text
Frontend / SDK
  -> Backend API
     -> MAD orchestrator
        -> shared vLLM endpoint
           -> model="agent_a" for Agent A calls
           -> model="agent_b" for Agent B calls
        -> judge model endpoint
```

The app or SDK does not need to know the model internals. It calls the backend API. The backend chooses the correct debate role and model name:

```text
Round 0 Agent A -> vLLM model agent_a
Round 0 Agent B -> vLLM model agent_b
Round 1 Agent A -> vLLM model agent_a
Round 1 Agent B -> vLLM model agent_b
Judge           -> judge model endpoint
```

Recommended AWS placement:

```text
EC2 GPU instance, ECS on a GPU-backed EC2 node, or EKS on a GPU node
vLLM OpenAI-compatible server
Base model: Qwen/Qwen2.5-14B-Instruct
LoRA modules:
  agent_a=/path/to/grpo_agent_a/final
  agent_b=/path/to/grpo_agent_b/final
```

## Runtime Prompts

### Agent A System Prompt

Agent A is the strict verifier. It should be conservative and require direct evidence.

```text
You are a strict regulatory compliance auditor in a structured debate.

YOUR ROLE: Determine precisely whether the claim is supported by the retrieved evidence.

APPROACH:
1. Read the claim and note any specific numbers, thresholds, dates, or scope qualifiers.
2. Examine each retrieved chunk. Ask: does this chunk directly support the claim AS STATED?
3. Partial support is PARTIAL — not SUPPORTED. Missing a qualifier is NOT_SUPPORTED.
4. "The evidence does not say otherwise" is NOT sufficient for SUPPORTED.

VERDICT DEFINITIONS:
- SUPPORTED    : Evidence directly and completely backs the claim with no meaningful gaps.
- PARTIAL      : Evidence supports the core idea but misses a qualifier, scope, or specific value.
- NOT_SUPPORTED: Evidence is absent, contradicts the claim, or the claim introduces specifics not in any chunk.
- IDK          : Evidence exists but is genuinely too ambiguous to resolve the claim.

RULES:
- Cite at most 2 chunk_ids. Each relevant_quote must be at most 240 characters.
- If a claim states a specific number or date — it must appear explicitly in the evidence.
- confidence_internal is YOUR certainty in YOUR verdict (0=very uncertain, 1=certain).
- Keep reasoning to at most 3 short sentences.
- Return valid JSON only. No markdown fences, no text before or after.
```

### Agent B System Prompt

Agent B is the skeptical challenger. It should stress-test scope, missing qualifiers, and overstatements.

```text
You are a skeptical regulatory auditor in a structured debate.

YOUR ROLE: Stress-test the claim. Find what is wrong, overstated, out of scope, or missing.

CORE ASSUMPTION: Treat the claim as INCORRECT until the evidence proves otherwise.
The burden of proof is on the claim — not on you to disprove it.

APPROACH:
1. Ask: "Under what conditions does this claim FAIL or become misleading?"
2. Look specifically for: scope limitations, unstated exceptions, missing qualifiers,
   outdated guidance, overgeneralized numbers, fabricated specifics.
3. Check whether the evidence covers the FULL scope of the claim, not just its core idea.

VERDICT DEFINITIONS:
- NOT_SUPPORTED: Your default when evidence is incomplete, ambiguous, or only partially covers the claim.
- PARTIAL      : Only when you can identify exactly what the evidence supports AND what it fails to cover.
- SUPPORTED    : Only when evidence is unambiguous AND the claim is precisely and completely stated.
- IDK          : Evidence exists but genuinely cannot resolve the claim.

RULES:
- General skepticism without evidence citation does not count. Cite specific chunks.
- If the claim is genuinely well-supported, say so — false challenges damage your credibility.
- Cite at most 2 chunk_ids. Each relevant_quote must be at most 240 characters.
- confidence_internal is YOUR certainty in YOUR verdict.
- Keep reasoning to at most 3 short sentences.
- Return valid JSON only. No markdown fences, no text before or after.
```

## Generation Settings

Recommended deployment settings:

```text
Agent A temperature: 0.2-0.4
Agent B temperature: 0.4-0.6
max_new_tokens / max_tokens: 192
Output: JSON only
```

Use lower temperature for stricter production behavior. If the JSON rate drops, reduce temperature toward `0.0-0.2`.

## JSON Output Schema

Both agents should return only compact JSON:

```json
{
  "verdict": "SUPPORTED | PARTIAL | NOT_SUPPORTED | IDK",
  "reasoning": "At most 3 short sentences.",
  "evidence_cited": [
    {
      "chunk_id": "...",
      "relevant_quote": "..."
    }
  ],
  "confidence_internal": 0.0
}
```

Validation rules:

```text
verdict must be one of: SUPPORTED, PARTIAL, NOT_SUPPORTED, IDK
confidence_internal must be numeric between 0.0 and 1.0
evidence_cited should contain at most 2 chunks
reasoning should stay short and claim-focused
```

## Why Dynamic LoRA Instead Of Merged Models

Prefer dynamic LoRA serving for this MAD architecture:

```text
Use one base model + two LoRA adapters in one vLLM server.
```

Benefits:

```text
Loads the 14B base model only once.
Keeps Agent A and Agent B behavior separate.
Saves GPU memory compared with two merged models.
Allows Agent A and Agent B adapters to be updated independently.
Matches the original MAD design: one shared model server, two logical agents.
Makes backend routing simple through model="agent_a" and model="agent_b".
Makes A/B testing base agents vs GRPO adapters straightforward.
```

Only merge adapters if the deployment environment cannot support dynamic LoRA loading and absolutely requires standalone model folders. Merging creates separate Agent A and Agent B model directories and usually requires more storage and serving complexity.

## Optimization Notes

Recommended optimizations:

```text
1. Register LoRA adapters once at vLLM startup.
2. Keep one shared vLLM endpoint for both agent roles.
3. Use OpenAI-compatible batching through vLLM instead of serial local model loads.
4. Keep deterministic stages cached: decomposer, retrieved claim chunks, and judge calls where inputs are identical.
5. Reuse prompt prefixes where possible so vLLM can benefit from prefix/KV caching.
6. Keep Agent A/B prompts stable and move per-claim data into the user message.
7. Keep max_tokens at 192 unless a production trace proves truncation.
8. For production JSON reliability, use low temperatures and strict output parsing/retry.
9. Store adapter version, base model version, prompt hash, and eval summary with every deployment.
10. Compare base-vs-GRPO runs on the same 50 original MAD queries before promoting the adapters.
```

## Evaluation Summary

The held-out claim-level test evaluation showed the GRPO adapters improved JSON reliability and calibration versus the base 14B agent model.

### Held-Out Test After 150 GRPO Steps

| Metric | Base Agent A | GRPO Agent A | Base Agent B | GRPO Agent B |
| --- | ---: | ---: | ---: | ---: |
| Valid JSON rate | 65.6% | 100.0% | 65.6% | 100.0% |
| Mean confidence | 0.857 | 0.375 | 0.629 | 0.412 |
| Mean absolute error | 0.533 | 0.128 | 0.495 | 0.141 |
| Brier MSE | 0.3667 | 0.0516 | 0.3257 | 0.0616 |
| Brier reward | -0.1762 | 0.1594 | -0.1352 | 0.1494 |
| Overconfident unsupported rate | 87.5% | 0.0% | 75.0% | 0.0% |
| Overconfident partial rate | 100.0% | 0.0% | 66.7% | 0.0% |
| Verdict exact match | 66.7% | 78.1% | 57.1% | 75.0% |

Interpretation:

```text
The adapters reduced overconfident unsupported and partial claims to 0% on the held-out test split.
They also improved JSON validity from 65.6% to 100%.
The largest practical win is better calibration: lower confidence on weak/partial evidence and fewer unsupported high-confidence outputs.
```

## Final Recommendation

For the AWS MAD deployment:

```text
Do not merge the adapters.
Serve Qwen/Qwen2.5-14B-Instruct once with two registered LoRA adapters.
Route Agent A calls to model="agent_a".
Route Agent B calls to model="agent_b".
Keep the Drive folder as the artifact source unless the repo is configured for Git LFS or an artifact registry is added.
```

Source note: vLLM LoRA serving is documented in the official vLLM docs through `--enable-lora` and `--lora-modules {name}={path}`: https://docs.vllm.ai/usage/lora/
