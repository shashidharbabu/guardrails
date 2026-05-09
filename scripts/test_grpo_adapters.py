"""
Quick smoke test: load Agent A and Agent B GRPO LoRA adapters locally and run
one debate round on a single healthcare claim.

Usage:
    python scripts/test_grpo_adapters.py

Requires ~10-12 GB RAM (4-bit quantised base model + two adapters).
Set HF_TOKEN env var if Qwen/Qwen2.5-14B-Instruct is gated on your account.

Adapter paths are resolved relative to the repo root:
    agent-a-adapters/
    agent-b-adapters/
"""

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "multi_agent_debate" / "full_FinalMAD_with_judge"))

ADAPTER_A = REPO_ROOT / "agent-a-adapters"
ADAPTER_B = REPO_ROOT / "agent-b-adapters"

CLAIM = (
    "HIPAA requires covered entities to implement AES-256 encryption for all ePHI "
    "at rest and in transit, with no exceptions for small providers."
)

EVIDENCE = """
[chunk_id: hipaa_164_312_a2]
The HIPAA Security Rule at 45 CFR §164.312(a)(2)(iv) lists encryption and decryption
as an addressable implementation specification under access control. Covered entities
must assess whether it is a reasonable and appropriate safeguard given their environment.

[chunk_id: hipaa_164_312_e2]
45 CFR §164.312(e)(2)(ii) makes encryption of ePHI in transit an addressable
specification. Entities that determine it is not reasonable and appropriate must
document that decision and implement an equivalent alternative measure.
"""

AGENT_A_SYSTEM = """You are a strict regulatory compliance auditor in a structured debate.
YOUR ROLE: Determine precisely whether the claim is supported by the retrieved evidence.
VERDICT DEFINITIONS:
- SUPPORTED: Evidence directly and completely backs the claim with no meaningful gaps.
- PARTIAL: Evidence supports the core idea but misses a qualifier, scope, or specific value.
- NOT_SUPPORTED: Evidence is absent, contradicts the claim, or introduces specifics not in any chunk.
- IDK: Evidence exists but is genuinely too ambiguous to resolve the claim.
Return valid JSON only: {"verdict":"...","reasoning":"...","evidence_cited":[{"chunk_id":"...","relevant_quote":"..."}],"confidence_internal":0.0}"""

AGENT_B_SYSTEM = """You are a skeptical regulatory auditor in a structured debate.
YOUR ROLE: Stress-test the claim. Find what is wrong, overstated, out of scope, or missing.
CORE ASSUMPTION: Treat the claim as INCORRECT until the evidence proves otherwise.
VERDICT DEFINITIONS:
- NOT_SUPPORTED: Your default when evidence is incomplete or only partially covers the claim.
- PARTIAL: Only when you can identify exactly what the evidence supports AND what it fails to cover.
- SUPPORTED: Only when evidence is unambiguous AND the claim is precisely and completely stated.
- IDK: Evidence exists but genuinely cannot resolve the claim.
Return valid JSON only: {"verdict":"...","reasoning":"...","evidence_cited":[{"chunk_id":"...","relevant_quote":"..."}],"confidence_internal":0.0}"""

USER_PROMPT = f"""CLAIM TO EVALUATE:
{CLAIM}

RETRIEVED EVIDENCE:
{EVIDENCE}

Return ONLY valid JSON. No markdown fences."""


def load_adapter(adapter_path: Path, label: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_model_id = "unsloth/qwen2.5-14b-instruct-unsloth-bnb-4bit"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"\n[{label}] Loading base model {base_model_id} in 4-bit NF4...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",
        token=hf_token,
    )
    print(f"[{label}] Applying LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base, str(adapter_path), token=hf_token)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), token=hf_token)
    print(f"[{label}] Ready.")
    return model, tokenizer


def run_inference(model, tokenizer, system: str, user: str, label: str) -> dict:
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().replace("<|im_end|>", "").strip()

    print(f"\n[{label}] Raw output ({latency_ms}ms):\n{response}")

    try:
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", response, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    print(f"[{label}] WARNING: could not parse JSON from output")
    return {"raw": response}


def main():
    print("=" * 60)
    print("GRPO Adapter Smoke Test")
    print("=" * 60)
    print(f"\nClaim: {CLAIM}\n")

    # Check adapter folders exist
    for path, label in [(ADAPTER_A, "Agent A"), (ADAPTER_B, "Agent B")]:
        if not path.exists():
            print(f"ERROR: {label} adapter not found at {path}")
            sys.exit(1)
        print(f"{label} adapter found: {path}")

    # Load and test Agent A
    model_a, tok_a = load_adapter(ADAPTER_A, "Agent A")
    result_a = run_inference(model_a, tok_a, AGENT_A_SYSTEM, USER_PROMPT, "Agent A")
    print(f"\n[Agent A] Parsed result: {json.dumps(result_a, indent=2)}")

    # Free Agent A before loading B to save memory
    del model_a, tok_a
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load and test Agent B
    model_b, tok_b = load_adapter(ADAPTER_B, "Agent B")
    result_b = run_inference(model_b, tok_b, AGENT_B_SYSTEM, USER_PROMPT, "Agent B")
    print(f"\n[Agent B] Parsed result: {json.dumps(result_b, indent=2)}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Agent A verdict:     {result_a.get('verdict', 'PARSE_ERROR')}")
    print(f"Agent A confidence:  {result_a.get('confidence_internal', '?')}")
    print(f"Agent B verdict:     {result_b.get('verdict', 'PARSE_ERROR')}")
    print(f"Agent B confidence:  {result_b.get('confidence_internal', '?')}")

    valid_json = "raw" not in result_a and "raw" not in result_b
    print(f"\nBoth outputs valid JSON: {'YES' if valid_json else 'NO'}")
    if valid_json:
        print("Adapter smoke test PASSED")
    else:
        print("Adapter smoke test FAILED — check raw output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
