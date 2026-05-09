# SpartanGuard AWS Deployment Assets

This directory captures the production deployment shape for `www.spartanguard.ai` in AWS account `829108230080`, region `us-west-1`.

## Target Architecture

- Public entry: Route53 or registrar DNS -> ACM certificate -> public ALB.
- Public services: `spartanguard-frontend` and `spartanguard-backend` on ECS Fargate.
- Internal services: `spartanguard-gateway`, `spartanguard-mad-api`, and `spartanguard-rlhf` on ECS Fargate.
- GPU model runtime: one private GPU EC2 instance for deadline deployment, running vLLM OpenAI-compatible API with LoRA modules `agent_a` and `agent_b`.
- Data plane: RDS PostgreSQL, ElastiCache Redis, Qdrant Cloud, Secrets Manager, CloudWatch Logs.

## Files

- `ecs-service-task-matrix.md`: ECS task/service definitions, ports, health checks, and routing.
- `secrets-map.md`: required AWS Secrets Manager values and consuming services.
- `environment-variables.md`: non-secret environment variables by service.
- `manual-console-deployment-checklist.md`: console-first deployment checklist.

## Current Deployment Caveats

- Backend `MAD_MODE=api` now calls `MAD_API_URL`, so ECS can run MAD as the separate `spartanguard-mad-api` service.
- If `api.spartanguard.ai` is used instead of same-origin `/api`, the frontend Nginx CSP must allow that API origin in `connect-src`.
- The exact fine-tuned Qwen embedding 4B Hugging Face ID or artifact path is still open.
- DNS ownership is undecided: either move `spartanguard.ai` DNS to Route53 or keep registrar-managed DNS and point records to the ALB.
- Fargate cannot run GPU workloads. The vLLM LoRA server must run on GPU EC2 or SageMaker.

## Image Naming

Use one ECR repository per deployable service:

- `spartanguard/frontend`
- `spartanguard/backend`
- `spartanguard/gateway`
- `spartanguard/mad-api`
- `spartanguard/rlhf`

Recommended image tag for first production cut: the short Git SHA from this worktree.
