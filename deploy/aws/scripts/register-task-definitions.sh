#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-deploy/aws/env/prod.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-829108230080}"
AWS_REGION="${AWS_REGION:-us-west-1}"
IMAGE_TAG="${IMAGE_TAG:?IMAGE_TAG is required}"
ECS_EXECUTION_ROLE_ARN="${ECS_EXECUTION_ROLE_ARN:?ECS_EXECUTION_ROLE_ARN is required}"
ECS_TASK_ROLE_ARN="${ECS_TASK_ROLE_ARN:?ECS_TASK_ROLE_ARN is required}"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
AWS_CMD="${AWS_CMD:-aws}"

if ! command -v "$AWS_CMD" >/dev/null 2>&1; then
  if [ -x "$HOME/Library/Python/3.9/bin/aws" ]; then
    AWS_CMD="$HOME/Library/Python/3.9/bin/aws"
  else
    echo "aws CLI not found. Install AWS CLI or set AWS_CMD=/path/to/aws." >&2
    exit 1
  fi
fi

python3 - "$AWS_REGION" "$REGISTRY" "$IMAGE_TAG" "$ECS_EXECUTION_ROLE_ARN" "$ECS_TASK_ROLE_ARN" <<'PY'
import json
import os
import sys
from pathlib import Path

region, registry, tag, execution_role, task_role = sys.argv[1:6]
out_dir = Path("deploy/aws/generated/task-definitions")
out_dir.mkdir(parents=True, exist_ok=True)


def env(name, default=None):
    value = os.getenv(name, default)
    if value is None:
        raise SystemExit(f"missing required env: {name}")
    return value


def environment(pairs):
    return [{"name": key, "value": str(value)} for key, value in pairs.items() if value is not None]


def secrets(mapping):
    return [{"name": key, "valueFrom": value} for key, value in mapping.items() if value]


def task(name, repo, port, cpu, memory, env_pairs, secret_pairs):
    family = f"spartanguard-{name}"
    return {
        "family": family,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": str(cpu),
        "memory": str(memory),
        "executionRoleArn": execution_role,
        "taskRoleArn": task_role,
        "containerDefinitions": [
            {
                "name": family,
                "image": f"{registry}/{repo}:{tag}",
                "essential": True,
                "portMappings": [{"containerPort": port, "hostPort": port, "protocol": "tcp"}],
                "environment": environment(env_pairs),
                "secrets": secrets(secret_pairs),
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": f"/ecs/spartanguard/{name}",
                        "awslogs-region": region,
                        "awslogs-stream-prefix": "ecs",
                    },
                },
            }
        ],
    }


definitions = {
    "frontend": task(
        "frontend",
        "spartanguard/frontend",
        8080,
        env("FRONTEND_CPU", "512"),
        env("FRONTEND_MEMORY", "1024"),
        {
            "VITE_API_BASE_URL": env("PUBLIC_API_URL", "/api"),
            "VITE_APP_ENV": "production",
            "VITE_APP_VERSION": env("APP_VERSION", "1.0.0"),
            "VITE_MOCK_API": "false",
        },
        {},
    ),
    "backend": task(
        "backend",
        "spartanguard/backend",
        8000,
        env("BACKEND_CPU", "1024"),
        env("BACKEND_MEMORY", "2048"),
        {
            "APP_ENV": "production",
            "APP_VERSION": env("APP_VERSION", "1.0.0"),
            "GATEWAY_URL": env("GATEWAY_URL"),
            "MAD_MODE": "api",
            "MAD_API_URL": env("MAD_API_URL"),
            "FEEDBACK_API_URL": env("FEEDBACK_API_URL"),
            "DISABLE_AUTH": "false",
            "CORS_ALLOWED_ORIGINS": env("PUBLIC_APP_URL"),
            "LLM_PROVIDER_TYPE": "custom",
            "LLM_PROVIDER_URL": env("VLLM_URL"),
            "DEFAULT_LLM_MODEL": env("DEFAULT_LLM_MODEL"),
            "JUDGE_BACKEND": "claude",
            "JUDGE_PROVIDER": "anthropic",
            "EMBED_MODEL": env("EMBED_MODEL"),
            "LOG_LEVEL": "INFO",
        },
        {
            "DATABASE_URL": env("DATABASE_URL_SECRET", ""),
            "REDIS_URL": env("REDIS_URL_SECRET", ""),
            "JWT_SECRET_KEY": env("JWT_SECRET_KEY_SECRET", ""),
            "ANTHROPIC_API_KEY": env("ANTHROPIC_API_KEY_SECRET", ""),
            "QDRANT_URL": env("QDRANT_URL_SECRET", ""),
            "QDRANT_API_KEY": env("QDRANT_API_KEY_SECRET", ""),
            "QDRANT_COLLECTION": env("QDRANT_COLLECTION_SECRET", ""),
        },
    ),
    "gateway": task(
        "gateway",
        "spartanguard/gateway",
        8080,
        env("GATEWAY_CPU", "2048"),
        env("GATEWAY_MEMORY", "4096"),
        {
            "APP_ENV": "production",
            "PII_MODEL_PATH": "shashidharbabu/deberta-pii-guardrails",
            "THREAT_MODEL_PATH": "shashidharbabu/roberta-jailbreak-guardrails",
            "PROMPT_INJECTION_MODEL_PATH": "shashidharbabu/llama-prompt-guard-guardrails",
            "GATEWAY_CORS_ALLOWED_ORIGINS": env("PUBLIC_API_URL", ""),
        },
        {"HF_TOKEN": env("HF_TOKEN_SECRET", "")},
    ),
    "mad-api": task(
        "mad-api",
        "spartanguard/mad-api",
        8001,
        env("MAD_API_CPU", "2048"),
        env("MAD_API_MEMORY", "4096"),
        {
            "APP_ENV": "production",
            "SQLITE_DB_PATH": "/data/mad.db",
            "LANGCHAIN_CACHE_DB": "/data/.langchain_cache.db",
            "VLLM_AGENTS_URL": env("VLLM_URL"),
            "VLLM_DECOMPOSER_URL": env("VLLM_URL"),
            "VLLM_BASELINE_URL": env("VLLM_URL"),
            "VLLM_JUDGE_URL": env("VLLM_URL"),
            "AGENTS_MODEL_NAME": env("AGENTS_MODEL_NAME", "agent_a"),
            "DECOMPOSER_MODEL_NAME": env("DECOMPOSER_MODEL_NAME"),
            "BASELINE_MODEL_NAME": env("BASELINE_MODEL_NAME"),
            "JUDGE_MODEL_NAME": env("JUDGE_MODEL_NAME"),
            "VLLM_API_KEY": "EMPTY",
            "OPENAI_API_KEY": "dummy-not-used",
            "CLAIM_CONCURRENCY": "2",
        },
        {
            "ANTHROPIC_API_KEY": env("ANTHROPIC_API_KEY_SECRET", ""),
            "LANGFUSE_PUBLIC_KEY": env("LANGFUSE_PUBLIC_KEY_SECRET", ""),
            "LANGFUSE_SECRET_KEY": env("LANGFUSE_SECRET_KEY_SECRET", ""),
        },
    ),
    "rlhf": task(
        "rlhf",
        "spartanguard/rlhf",
        8002,
        env("RLHF_CPU", "512"),
        env("RLHF_MEMORY", "1024"),
        {
            "APP_ENV": "production",
            "FEEDBACK_API_HOST": "0.0.0.0",
            "FEEDBACK_API_PORT": "8002",
            "MAD_DB_PATH": "/data/mad.db",
            "HUMAN_FEEDBACK_LOG_PATH": "/data/human_feedback_log.jsonl",
            "FEEDBACK_TRIAGE_LOW": "-0.10",
            "FEEDBACK_TRIAGE_HIGH": "0.30",
            "FEEDBACK_USE_PRESIDIO": "0",
        },
        {},
    ),
}

for name, body in definitions.items():
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(body, indent=2) + "\n")
    print(path)
PY

for task_file in deploy/aws/generated/task-definitions/*.json; do
  "$AWS_CMD" ecs register-task-definition \
    --region "$AWS_REGION" \
    --cli-input-json "file://${task_file}" >/dev/null
  echo "registered: ${task_file}"
done
