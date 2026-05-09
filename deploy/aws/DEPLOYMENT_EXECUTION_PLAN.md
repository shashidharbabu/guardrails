# SpartanGuard Deployment Execution Plan

Use this as the fast path from the validated local stack to AWS production in account `829108230080`, region `us-west-1`.

## Current Status

- Branch `deployment-final` is pushed to GitHub.
- Local production compose is running with healthy containers:
  - frontend
  - backend
  - gateway
  - mad-api
  - rlhf
  - postgres
  - redis
- Backend `MAD_MODE=api` now calls `MAD_API_URL` instead of importing MAD in-process.
- AWS CLI is not installed in the current shell, so AWS resource creation and ECR pushes are blocked until the CLI is available and authenticated.

## Phase 1 - Local Release Candidate

1. Confirm the working tree is clean after the latest fixes.
2. Run:

```bash
docker compose -f docker-compose.prod-local.yml config --quiet
docker compose -f docker-compose.prod-local.yml build frontend backend gateway mad-api rlhf
docker compose -f docker-compose.prod-local.yml up -d
```

3. Verify:

```bash
curl http://localhost:8088/healthz
curl http://localhost:8000/healthz
curl http://localhost:8080/health
curl http://localhost:8001/mad/health
curl http://localhost:8002/health
docker compose -f docker-compose.prod-local.yml ps
```

Exit criteria: all app endpoints return `status=ok`, and all containers are healthy.

## Phase 2 - AWS CLI And Identity

1. Install AWS CLI v2.
2. Authenticate as deployment principal for account `829108230080`.
3. Verify:

```bash
aws sts get-caller-identity --region us-west-1
```

Expected account: `829108230080`.

## Phase 3 - ECR Repositories

Run:

```bash
deploy/aws/scripts/create-ecr-repos.sh
```

Creates:

- `spartanguard/frontend`
- `spartanguard/backend`
- `spartanguard/gateway`
- `spartanguard/mad-api`
- `spartanguard/rlhf`

## Phase 4 - Build And Push Images

Run:

```bash
deploy/aws/scripts/build-and-push-ecr.sh
```

Default tag is current Git short SHA. Override with:

```bash
IMAGE_TAG=my-tag deploy/aws/scripts/build-and-push-ecr.sh
```

Exit criteria: all five images are visible in ECR with the same tag.

## Phase 5 - AWS Foundation

Create through AWS Console for speed:

1. VPC/subnets or select existing production VPC.
2. Security groups:
   - public ALB
   - backend ECS
   - frontend ECS
   - internal services ECS
   - RDS
   - Redis
   - GPU EC2/vLLM
3. RDS PostgreSQL.
4. ElastiCache Redis.
5. Secrets Manager entries from `secrets-map.md`.
6. CloudWatch log groups for all ECS services.
7. ACM certificate for `www.spartanguard.ai` and optionally `api.spartanguard.ai`.

## Phase 6 - GPU vLLM Runtime

Deadline path:

1. Launch one GPU EC2 instance in a private subnet.
2. Install NVIDIA runtime and Docker.
3. Place LoRA adapters:
   - `/models/grpo_agent_a/final`
   - `/models/grpo_agent_b/final`
4. Start vLLM:

```bash
docker run --gpus all --restart unless-stopped -p 8000:8000 \
  -v /models:/models:ro \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-14B-Instruct \
  --enable-lora \
  --lora-modules agent_a=/models/grpo_agent_a/final agent_b=/models/grpo_agent_b/final
```

5. Verify from inside the VPC:

```bash
curl http://<private-vllm-host>:8000/v1/models
```

## Phase 7 - ECS Services

Deploy order:

1. `spartanguard-gateway`
2. `spartanguard-mad-api`
3. `spartanguard-rlhf`
4. `spartanguard-backend`
5. `spartanguard-frontend`

Use `ecs-service-task-matrix.md` and `environment-variables.md`.

Exit criteria:

- every ECS service has stable desired count
- target groups are healthy
- CloudWatch logs show startup without fatal errors

## Phase 8 - DNS And Public Smoke

1. Point `www.spartanguard.ai` to the public ALB.
2. If using `api.spartanguard.ai`, point it to the same ALB with host routing.
3. Verify:

```bash
curl https://www.spartanguard.ai/healthz
curl https://api.spartanguard.ai/healthz
```

4. Run one browser workflow:
   - open frontend
   - submit a query
   - confirm gateway result
   - confirm session persists
   - confirm MAD transitions to completed or unavailable with an audit trail

## Phase 9 - Immediate Hardening After Launch

- Replace temporary admin deployer permissions.
- Enable WAF on public ALB.
- Configure alarms for ECS restarts, ALB 5xx, RDS, Redis, and GPU host health.
- Confirm `DISABLE_AUTH=false` and Cognito/OIDC values are real.
- Decide durable MAD/RLHF storage path beyond local SQLite/EFS.
