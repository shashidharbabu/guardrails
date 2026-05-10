# Local Docker RAG + V4MAD Handoff

## Current State

Branch state:
- Working branch: `test-local-vineeth`
- `test-local-vineeth` was rebased onto `origin/deploy-local`.
- Pushed branches:
  - `origin/test-local-vineeth`
  - `origin/test-local`
- We did not push to `origin/deploy-local`; that branch was used as the Docker/deploy reference base.

Docker services currently working locally:
- `frontend` on `http://localhost:5173`
- `backend` on `http://localhost:8000`
- `gateway` on `http://localhost:8080`
- `postgres`
- `redis`

External services still used:
- Agent A/B LoRA vLLM server through ngrok/remote GPU.
- Claude judge through Anthropic API.
- Ollama on the host for baseline/decomposer through `host.docker.internal:11434`.

RAG is being moved from Colab/ngrok into Docker as `rag-service`.

## Verified Flow So Far

The frontend/backend Docker app was tested end-to-end:

```text
frontend /api/query
-> backend
-> gateway
-> baseline LLM on host Ollama
-> decomposer LLM on host Ollama
-> RAG retrieval
-> agent_a / agent_b LoRA server
-> Claude judge
-> backend session trace
-> UI
```

Expected UI trace:
- RAG pipeline shows `3-5 claims x top-5`.
- MAD pipeline shows `R0+R1 debate`.
- Judge final verdict shows `DELIVER`, `RETRY`, `HARD_BLOCK`, or related route.

## New Docker RAG Service

Added files:
- `rag_v2/Dockerfile`
- `requirements-rag.txt`
- `docs/LOCAL_DOCKER_RAG_MAD_HANDOFF.md`

The RAG container runs:

```bash
uvicorn rag_v2.rag_server:app --host 0.0.0.0 --port 8010
```

Inside Docker, backend should call:

```text
RAG_V2_REMOTE_URL=http://rag-service:8010
```

RAG service behavior:

```text
claim + original query
-> Qwen3 embedding
-> Qdrant Cloud dense retrieval
-> local BM25 from rag_v2/bm25_combined.pkl
-> RRF fusion
-> BGE reranker
-> top 5 chunks per claim
```

## Environment Needed

`.env` must contain real values for:

```bash
QDRANT_MODE=cloud
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_TIMEOUT=120
RAG_V2_COLLECTION_NAME=guardrails_rag_v2

USE_GRPO_LORA_AGENTS=1
AGENT_A_MODEL=agent_a
AGENT_B_MODEL=agent_b
AGENTS_BASE_URL=.../v1
AGENTS_API_KEY=EMPTY

JUDGE_BACKEND=claude
ANTHROPIC_API_KEY=...
```

Do not commit secrets. The current `.dockerignore` excludes `.env`.

## Local Docker Commands

Build and start app plus Docker RAG:

```bash
docker compose -f docker-compose.dev.yml up -d --build \
  postgres redis gateway rag-service backend frontend
```

Check health:

```bash
curl http://localhost:8010/health
curl http://localhost:8000/healthz
curl http://localhost:8080/health
curl http://localhost:5173/healthz
```

Tail logs:

```bash
docker compose -f docker-compose.dev.yml logs -f rag-service backend
```

Run an API smoke test:

```bash
curl -sS http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Under GDPR Article 33, when must a controller notify the supervisory authority of a personal data breach, and what exception applies?","llm_model":"qwen2.5:7b"}'
```

Then open the returned session in the frontend:

```text
http://localhost:5173
```

## Should Colab RAG Be Stopped?

Not until Docker RAG is healthy and one frontend query succeeds.

Safe order:
1. Start Docker `rag-service`.
2. Confirm `curl http://localhost:8010/health` returns `service_loaded: true`.
3. Run one frontend query.
4. Confirm backend logs show RAG retrieval through `rag-service`.
5. Then stop the Colab RAG cell.

Keep Colab vLLM agents running for now.

## Important Performance Note

The Docker RAG container loads:
- `Qwen/Qwen3-Embedding-4B`
- `BAAI/bge-reranker-v2-m3`

On a Mac CPU this may be slow and memory-heavy. It is useful for Docker wiring and local validation, but production should run RAG on GPU-backed infrastructure or a dedicated CPU instance with enough RAM.

## Future AWS Deploy-Cloud Shape

Recommended production split:

```text
Frontend
-> Backend orchestrator
-> Gateway
-> RAG service
   -> Qdrant Cloud
   -> BM25 local artifact in image/EFS/S3-on-startup
   -> embedding + reranker model
-> vLLM agents service
   -> Qwen2.5-14B-Instruct
   -> LoRA adapters registered as agent_a and agent_b
-> Claude or hosted 32B judge
-> Postgres / Redis
```

AWS options:
- Backend/frontend/gateway/RAG: ECS or EKS.
- RAG model hosting: GPU ECS/EKS node, SageMaker endpoint, or EC2 GPU/large CPU service.
- Agent service: GPU host only, preferably SageMaker or EC2/EKS GPU with vLLM and LoRA adapter registration.
- Qdrant: keep Qdrant Cloud.
- BM25 artifact: bake into image for now; later move to S3/EFS if it changes often.
- Secrets: AWS Secrets Manager or SSM Parameter Store.

## Agent Dockerization Later

Do not try to run agents on Mac Docker.

Agent requirements:
- Base model: `Qwen/Qwen2.5-14B-Instruct`
- LoRA adapters:
  - `agent_a`
  - `agent_b`
- vLLM started once with LoRA enabled.
- Backend calls the OpenAI-compatible endpoint:

```bash
USE_GRPO_LORA_AGENTS=1
AGENT_A_MODEL=agent_a
AGENT_B_MODEL=agent_b
AGENTS_BASE_URL=https://agent-host/v1
AGENTS_API_KEY=EMPTY
```

Keep this remote until a GPU deployment is ready.

## Known Warnings

- Docker compose may warn `HF_TOKEN` is not set. Gateway can still run, but real HF Serverless validation needs `HF_TOKEN`.
- Compose may warn `version` is obsolete. This is harmless but can be cleaned later.
- Current confidence engine behavior is intentionally unchanged.
