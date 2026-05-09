# Environment Variable Map

Values marked `secret` should be injected from Secrets Manager, not literal ECS task environment variables.

## Frontend

| Name | Production value |
| --- | --- |
| `VITE_API_BASE_URL` | `https://api.spartanguard.ai` or `/api` |
| `VITE_APP_ENV` | `production` |
| `VITE_APP_VERSION` | release version or Git SHA |
| `VITE_MOCK_API` | `false` |

## Backend

| Name | Production value |
| --- | --- |
| `APP_ENV` | `production` |
| `DATABASE_URL` | secret |
| `REDIS_URL` | secret |
| `GATEWAY_URL` | internal gateway URL |
| `MAD_MODE` | `api` after HTTP cutover |
| `MAD_API_URL` | `http://spartanguard-mad-api:8001/mad/verify` |
| `FEEDBACK_API_URL` | `http://spartanguard-rlhf:8002` |
| `DISABLE_AUTH` | `false` |
| `CORS_ALLOWED_ORIGINS` | `https://www.spartanguard.ai` |
| `ANTHROPIC_API_KEY` | secret |
| `QDRANT_URL` | secret |
| `QDRANT_API_KEY` | secret |
| `QDRANT_COLLECTION` | secret or environment |
| `EMBED_MODEL` | final Qwen embedding 4B model ID |
| `JUDGE_BACKEND` | `claude` |
| `JUDGE_PROVIDER` | `anthropic` |

## Gateway

| Name | Production value |
| --- | --- |
| `APP_ENV` | `production` |
| `HF_TOKEN` | secret |
| `PII_MODEL_PATH` | `shashidharbabu/deberta-pii-guardrails` |
| `THREAT_MODEL_PATH` | `shashidharbabu/roberta-jailbreak-guardrails` |
| `PROMPT_INJECTION_MODEL_PATH` | `shashidharbabu/llama-prompt-guard-guardrails` |
| `GATEWAY_CORS_ALLOWED_ORIGINS` | backend/internal origin |

## MAD API

| Name | Production value |
| --- | --- |
| `APP_ENV` | `production` |
| `SQLITE_DB_PATH` | `/data/mad.db` for local/EFS; migrate to durable store later |
| `LANGCHAIN_CACHE_DB` | `/data/.langchain_cache.db` |
| `VLLM_AGENTS_URL` | private vLLM OpenAI-compatible URL |
| `VLLM_DECOMPOSER_URL` | private vLLM URL or separate decomposer endpoint |
| `VLLM_BASELINE_URL` | private vLLM URL or backend-selected baseline endpoint |
| `VLLM_JUDGE_URL` | private vLLM URL only if using model judge |
| `AGENTS_MODEL_NAME` | `agent_a` |
| `AGENT_B_MODEL_NAME` | `agent_b` if integration code supports split names |
| `DECOMPOSER_MODEL_NAME` | `Qwen/Qwen2.5-14B-Instruct` or selected smaller model |
| `BASELINE_MODEL_NAME` | selected baseline model |
| `JUDGE_MODEL_NAME` | Claude model name when Claude path is wired |
| `VLLM_API_KEY` | `EMPTY` unless the internal endpoint enforces auth |
| `ANTHROPIC_API_KEY` | secret |
| `CLAIM_CONCURRENCY` | `2` initially |

## RLHF

| Name | Production value |
| --- | --- |
| `APP_ENV` | `production` |
| `FEEDBACK_API_HOST` | `0.0.0.0` |
| `FEEDBACK_API_PORT` | `8002` |
| `MAD_DB_PATH` | same MAD store path used by MAD API |
| `HUMAN_FEEDBACK_LOG_PATH` | `/data/human_feedback_log.jsonl` |
| `FEEDBACK_TRIAGE_LOW` | `-0.10` |
| `FEEDBACK_TRIAGE_HIGH` | `0.30` |
| `FEEDBACK_USE_PRESIDIO` | `0` initially unless image includes required NLP assets |
