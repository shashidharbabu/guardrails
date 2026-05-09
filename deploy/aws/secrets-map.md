# AWS Secrets Manager Map

Create these as Secrets Manager secrets in `us-west-1`. Do not store real secret values in Git or task definition plaintext.

| Secret name | Keys | Consuming service | Notes |
| --- | --- | --- | --- |
| `/spartanguard/prod/database` | `DATABASE_URL` | backend | RDS PostgreSQL URL. Prefer generated RDS credentials plus app-specific user. |
| `/spartanguard/prod/redis` | `REDIS_URL` | backend | ElastiCache Redis URL. |
| `/spartanguard/prod/huggingface` | `HF_TOKEN` | gateway, vLLM EC2 bootstrap, optional MAD API | Required for gated/private HF artifacts. |
| `/spartanguard/prod/qdrant` | `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` | backend, MAD/RAG if enabled | Qdrant Cloud is already running. |
| `/spartanguard/prod/anthropic` | `ANTHROPIC_API_KEY` | backend, MAD API | Required for Claude judge/co-pilot paths. |
| `/spartanguard/prod/auth` | `JWT_SECRET_KEY`, Cognito client values if needed | backend | Replace JWT local fallback before production. |
| `/spartanguard/prod/langfuse` | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | MAD API, backend if tracing added | Optional observability. |

## Model And Artifact Secrets

The LoRA adapter archives are deployment artifacts, not text secrets:

- `Agent A Adapters.zip`
- `AgentB adapters.zip`
- evaluation archive from Drive

Store unpacked runtime artifacts on the GPU host under:

- `/models/grpo_agent_a/final`
- `/models/grpo_agent_b/final`

Restrict access to the deployer role and GPU host instance profile.
