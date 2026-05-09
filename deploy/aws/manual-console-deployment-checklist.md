# Manual AWS Console Deployment Checklist

Use this for the first production deployment in AWS account `829108230080`, region `us-west-1`.

## 1. Identity

- Create IAM user `spartanguard-deployer`.
- Enable console access.
- Create CLI access key only if local image push/deploy commands will be run.
- Attach temporary `AdministratorAccess` for the deadline deployment.
- Replace with scoped permissions after launch.

## 2. Networking

- Create or select a VPC with at least two public and two private subnets in `us-west-1`.
- Create security groups:
  - Public ALB: inbound `443` from internet.
  - Frontend ECS: inbound app port from ALB SG only.
  - Backend ECS: inbound app port from ALB SG only.
  - Internal services ECS: inbound app ports from backend/MAD SGs only.
  - RDS: inbound PostgreSQL from backend ECS SG.
  - Redis: inbound Redis from backend ECS SG.
  - GPU EC2: inbound vLLM port from MAD API ECS SG only.
- Confirm private subnet egress through NAT or VPC endpoints for ECR, CloudWatch Logs, Secrets Manager, and HF downloads.

## 3. DNS And TLS

- Decide Route53 hosted zone vs registrar-managed DNS for `spartanguard.ai`.
- Request ACM certificate for:
  - `www.spartanguard.ai`
  - `api.spartanguard.ai` if using API subdomain
- Validate certificate through DNS.

## 4. Data Services

- Create RDS PostgreSQL.
- Create ElastiCache Redis.
- Confirm Qdrant Cloud endpoint and collection.
- Store required values in Secrets Manager according to `secrets-map.md`.

## 5. ECR And Images

- Create ECR repositories:
  - `spartanguard/frontend`
  - `spartanguard/backend`
  - `spartanguard/gateway`
  - `spartanguard/mad-api`
  - `spartanguard/rlhf`
- Build and push images from this worktree.
- Tag every image with the same Git SHA for traceability.

## 6. GPU vLLM Host

- Launch a GPU EC2 instance in a private subnet.
- Attach an instance profile with access to required artifact storage only.
- Install NVIDIA driver/runtime and Docker.
- Upload or mount LoRA adapter artifacts:
  - `/models/grpo_agent_a/final`
  - `/models/grpo_agent_b/final`
- Start vLLM with LoRA modules `agent_a` and `agent_b`.
- Verify `/v1/models` from inside the VPC.

## 7. ECS

- Create ECS cluster.
- Create CloudWatch log groups for all five services.
- Register task definitions using the matrix in `ecs-service-task-matrix.md`.
- Inject secrets from Secrets Manager.
- Deploy internal services first: gateway, MAD API, RLHF.
- Deploy backend.
- Deploy frontend.

## 8. ALB

- Create public ALB with HTTPS listener.
- Add target groups for frontend and backend.
- Add host/path listener rules.
- Keep gateway, MAD API, and RLHF off the public ALB.

## 9. Smoke Checks

- `https://www.spartanguard.ai/healthz`
- `https://api.spartanguard.ai/healthz` or backend health path through `/api`.
- Backend `/api/system/health` with authenticated request.
- Gateway internal `/health`.
- MAD API internal `/mad/health`.
- RLHF internal `/health`.
- One full query through gateway -> LLM -> MAD -> session trace.
- RLHF review route after MAD reward rows exist.

## 10. Post Launch Hardening

- Replace temporary administrator deployer access with scoped IAM.
- Disable any local/auth bypass settings.
- Add AWS WAF to public ALB.
- Add CloudWatch alarms for ECS task restarts, ALB 5xx, RDS CPU/storage, Redis memory, and GPU host health.
- Decide whether to move MAD/RLHF SQLite state to PostgreSQL or mounted EFS before sustained production use.
