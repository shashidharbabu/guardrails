# vLLM Setup Guide

The pipeline uses vLLM as an OpenAI-compatible inference server. One process per model, or a single process with LoRA switching.

## Recommended Architecture

Run four vLLM servers (or fewer if sharing a model):

| Port | Role | Model | Notes |
|------|------|-------|-------|
| 8001 | Agents A+B | Qwen2.5-14B-Instruct-AWQ | Shared endpoint, model="agents" |
| 8002 | Decomposer | Qwen2.5-7B-Instruct | model="decomposer" |
| 8003 | Baseline | Qwen2.5-1.5B-Instruct | model="baseline" |
| 8004 | Judge | Qwen2.5-32B-Instruct | Requires H100/A100+ for 32B |

## Launch Commands

### Single GPU (RTX 5090 / 32GB VRAM)

```bash
# Agents (AWQ quantized 14B fits in ~9GB)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct-AWQ \
  --served-model-name agents \
  --port 8001 \
  --gpu-memory-utilization 0.85 &

# Decomposer (7B, smaller)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name decomposer \
  --port 8002 \
  --gpu-memory-utilization 0.3 &

# Baseline (1.5B, tiny)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --served-model-name baseline \
  --port 8003 \
  --gpu-memory-utilization 0.1 &
```

### Judge (requires separate large GPU)
```bash
# On H100/A100 with 80GB VRAM
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct \
  --served-model-name judge \
  --port 8004 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.90
```

### With Fine-Tuned LoRA Adapters
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --enable-lora \
  --lora-modules agent_a=./training_reference/sft_adapters/sft_agent_a \
                 agent_b=./training_reference/sft_adapters/sft_agent_b \
  --max-lora-rank 16 \
  --port 8001
# Then set AGENTS_MODEL_NAME=agent_a in .env for agent A calls
# (requires per-role client configuration in vllm_client.py)
```

## Health Check

```bash
curl http://localhost:8001/v1/models
curl http://localhost:8004/v1/models
```

## Minimal Single-GPU Setup (testing only)

If you only have one GPU, run one model and point all four env vars at the same port:

```env
VLLM_AGENTS_URL=http://localhost:8001/v1
VLLM_DECOMPOSER_URL=http://localhost:8001/v1
VLLM_BASELINE_URL=http://localhost:8001/v1
VLLM_JUDGE_URL=http://localhost:8001/v1
AGENTS_MODEL_NAME=Qwen/Qwen2.5-14B-Instruct-AWQ
DECOMPOSER_MODEL_NAME=Qwen/Qwen2.5-14B-Instruct-AWQ
BASELINE_MODEL_NAME=Qwen/Qwen2.5-14B-Instruct-AWQ
JUDGE_MODEL_NAME=Qwen/Qwen2.5-14B-Instruct-AWQ
```

This works but reduces judge quality significantly (14B vs 32B).
