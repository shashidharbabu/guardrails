# SpartanGuard Deployment Final Plan

Branch/worktree: `deployment-final` at `/Users/spartan/Documents/guardrails-deployment-final`

Primary integration/testing worktree: `codex/deploy-work` at `/Users/spartan/Documents/guardrails-enterprise`

## Current Decision Record

- AWS account: `829108230080`
- AWS region: `us-west-1` / N. California
- Deployment target: production from day one
- Container platform: ECS Fargate for non-GPU services
- GPU runtime: SageMaker or ECS on GPU-backed EC2 for model serving
- Frontend: containerized Nginx/Vite build on ECS for v1
- Domain: `www.spartanguard.ai`
- DNS: not configured yet; decide Route53 hosted zone vs registrar-managed DNS
- Auth: AWS Cognito, simple v1 RBAC
- RBAC v1 personas:
  - `SpartanGuardAdmin`
  - `EnterpriseClient`
- Qdrant: Qdrant Cloud, already running
- Judge: Claude via Anthropic API
- RLHF: live in v1
- MAD: live in v1

## Model Configuration

### Gateway Input Guardrails

Use the updated Hugging Face model IDs:

- PII: `shashidharbabu/deberta-pii-guardrails`
- Jailbreak: `shashidharbabu/roberta-jailbreak-guardrails`
- Prompt injection: `shashidharbabu/llama-prompt-guard-guardrails`

Required secret:

- `HF_TOKEN`

Store in AWS Secrets Manager. Do not commit to Git.

### RAG / Embeddings

Use Qwen embedding 4B for query embeddings.

Required environment variables:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
- `EMBED_MODEL=<final Qwen embedding 4B model id>`

Open item: confirm exact Hugging Face ID or model artifact path for the fine-tuned Qwen embedding 4B model.

### MAD Agents

Use Vineeth's recent GRPO LoRA deployment handoff:

- Base model: `Qwen/Qwen2.5-14B-Instruct`
- Dynamic LoRA adapters:
  - `agent_a=/models/grpo_agent_a/final`
  - `agent_b=/models/grpo_agent_b/final`
- Serve one vLLM OpenAI-compatible endpoint with both LoRA modules registered once at startup.
- Agent A calls model name: `agent_a`
- Agent B calls model name: `agent_b`

Required artifacts:

- `Agent A Adapters.zip`
- `AgentB adapters.zip`
- evaluation archive from Drive

Required cloud runtime:

- GPU-backed model server. Fargate cannot run GPU.
- Recommended for deadline: one GPU EC2 instance running vLLM, registered behind an internal ALB/NLB or private DNS.

### Judge

Use Claude through Anthropic API.

Required secret:

- `ANTHROPIC_API_KEY`

Open implementation item:

- Ensure full MAD judge path uses Claude where required and does not depend on local Ollama for judge.

## Target AWS Architecture

### Public Entry

- Route53 hosted zone for `spartanguard.ai` if domain DNS can be moved to AWS.
- ACM certificate for `www.spartanguard.ai`.
- Public ALB for ECS services.
- Frontend ECS service serves Nginx static build.

### ECS Fargate Services

1. `spartanguard-frontend`
   - Image: app frontend Dockerfile
   - Port: 8080
   - Public route: `https://www.spartanguard.ai`

2. `spartanguard-backend`
   - Image: backend Dockerfile
   - Port: 8000
   - Public route through ALB path or subdomain:
     - preferred: `https://api.spartanguard.ai`
     - alternative: `https://www.spartanguard.ai/api/*`

3. `spartanguard-gateway`
   - Image: gateway Dockerfile
   - Port: 8080
   - Internal-only service
   - Backend talks to it through service discovery/internal ALB

4. `spartanguard-mad-api`
   - New Dockerfile required
   - Port: 8001
   - Internal-only service
   - Calls vLLM GPU endpoint for Agent A/B
   - Calls Claude for judge

5. `spartanguard-rlhf`
   - New Dockerfile required
   - Port: 8002
   - Internal-only service
   - Reads MAD output store/reward tables

### Data Services

- RDS PostgreSQL for backend application data.
- ElastiCache Redis for async/background coordination.
- Qdrant Cloud for vector store.
- AWS Secrets Manager for all runtime secrets.
- CloudWatch Logs for all ECS services.

### GPU Model Runtime

Deadline-friendly recommended path:

- EC2 GPU instance in same VPC/private subnet.
- Run vLLM container with:
  - base model: `Qwen/Qwen2.5-14B-Instruct`
  - `--enable-lora`
  - `--lora-modules agent_a=/models/grpo_agent_a/final agent_b=/models/grpo_agent_b/final`
- Expose only inside VPC.

Later hardening:

- Move to SageMaker endpoint or autoscaled ECS GPU capacity provider.

## Local Container Test Plan Before AWS

Before deploying AWS, run a production-like local Docker stack:

1. Build containers:
   - frontend
   - backend
   - gateway
   - mad-api
   - rlhf

2. Run local compose:
   - Postgres
   - Redis
   - frontend
   - backend
   - gateway
   - MAD API
   - RLHF API
   - optional mock vLLM endpoint if GPU is unavailable locally

3. Verify:
   - `/healthz`
   - `/readyz`
   - `/api/system/health`
   - `/api/gateway/validate`
   - `/api/query`
   - `/api/sessions`
   - session trace page
   - RLHF review routes
   - browser E2E suite

## Immediate Implementation Tasks

### Deployment Worktree: `deployment-final`

1. Add `deploy/aws/` with:
   - architecture README
   - ECS service/task matrix
   - secrets map
   - environment variable map
   - manual console deployment checklist

2. Add production local Docker compose:
   - `docker-compose.prod-local.yml`

3. Add Dockerfiles:
   - `multi_agent_debate/full_FinalMAD_with_judge/Dockerfile`
   - `rlhf/Dockerfile`

4. Add AWS deployment scripts or Terraform/CDK later if time allows.

### Integration Worktree: `codex/deploy-work`

1. Update model env defaults to desired model IDs.
2. Verify gateway loads the three HF models with `HF_TOKEN`.
3. Update MAD config to target:
   - `VLLM_AGENTS_URL`
   - `VLLM_DECOMPOSER_URL`
   - `VLLM_BASELINE_URL`
   - `VLLM_JUDGE_URL` or Claude judge path
4. Wire Agent A/B model names:
   - `AGENTS_MODEL_NAME=agent_a`
   - `AGENT_B_MODEL_NAME=agent_b` if code requires a new split
5. Ensure RLHF service runs against the MAD store.
6. Run full local integration test.

## AWS Console Access Setup

Fastest deadline path:

1. IAM -> Users -> Create user `spartanguard-deployer`
2. Enable AWS Management Console access.
3. Create access key for CLI if we will run CLI locally.
4. Attach `AdministratorAccess` temporarily.
5. After deployment, replace with scoped permissions.

Required services permissions:

- ECR
- ECS
- EC2/VPC
- ELBv2
- RDS
- ElastiCache
- Secrets Manager
- CloudWatch Logs
- IAM role creation/pass role
- ACM
- Route53, if DNS managed in AWS
- Cognito

## Open Questions

1. Exact Qwen embedding 4B model ID/path.
2. Exact Drive artifact access for Agent A/B adapters.
3. Whether `spartanguard.ai` registrar access is available today.
4. Whether to create Route53 hosted zone now.
5. Whether local Docker test should use mock vLLM or remote GPU vLLM.
6. Whether MAD judge must be Claude in full_FinalMAD path before AWS deploy.
