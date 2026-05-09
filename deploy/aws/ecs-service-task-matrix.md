# ECS Service And Task Matrix

| Service | Launch type | Image source | Port | ALB exposure | Health check | CPU / memory start |
| --- | --- | --- | ---: | --- | --- | --- |
| `spartanguard-frontend` | Fargate | `app/frontend/Dockerfile` | 8080 | Public, `www.spartanguard.ai` | `GET /healthz` | 512 CPU / 1024 MiB |
| `spartanguard-backend` | Fargate | `app/backend/Dockerfile` | 8000 | Public, `api.spartanguard.ai` or `/api/*` | `GET /healthz` | 1024 CPU / 2048 MiB |
| `spartanguard-gateway` | Fargate | `gateway/Dockerfile` | 8080 | Internal only | `GET /health` | 2048 CPU / 4096 MiB |
| `spartanguard-mad-api` | Fargate | `multi_agent_debate/full_FinalMAD_with_judge/Dockerfile` | 8001 | Internal only | `GET /mad/health` | 2048 CPU / 4096 MiB |
| `spartanguard-rlhf` | Fargate | `rlhf/Dockerfile` | 8002 | Internal only | `GET /health` | 512 CPU / 1024 MiB |

## Routing

- Frontend target group: public ALB listener `443` host `www.spartanguard.ai` -> frontend port `8080`.
- Backend target group preferred: public ALB listener `443` host `api.spartanguard.ai` -> backend port `8000`.
- Backend target group fallback: public ALB path rule `www.spartanguard.ai/api/*` -> backend port `8000`.
- Gateway, MAD API, and RLHF should not have public listener rules. Use ECS Cloud Map service discovery or an internal ALB.

## Service Connectivity

- Backend -> Gateway: `http://spartanguard-gateway:8080`.
- Backend -> MAD API: `http://spartanguard-mad-api:8001/mad/verify` after backend HTTP cutover.
- Backend -> RLHF: `http://spartanguard-rlhf:8002`.
- MAD API -> vLLM: private DNS or internal NLB, OpenAI-compatible `/v1`.
- Gateway -> Hugging Face: outbound HTTPS for model download unless models are baked or mounted.
- Backend -> RDS PostgreSQL and ElastiCache Redis.
- Backend/MAD/RLHF -> CloudWatch Logs.

## GPU Runtime

Deadline path:

- EC2 GPU instance in private subnet, same VPC/security boundary as ECS.
- Inbound only from MAD API security group to vLLM port.
- Run vLLM with:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --enable-lora \
  --lora-modules agent_a=/models/grpo_agent_a/final agent_b=/models/grpo_agent_b/final
```

Later hardening can move this to SageMaker or an ECS GPU capacity provider.
