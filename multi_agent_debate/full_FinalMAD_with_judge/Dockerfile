# syntax=docker/dockerfile:1
# MAD microservice — Multi-Agent Debate pipeline (full_FinalMAD_with_judge)
# No GPU required: connects to Ollama or vLLM via env vars.
FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip
COPY multi_agent_debate/full_FinalMAD_with_judge/requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim AS runtime

LABEL maintainer="guardrails-platform"
LABEL description="MAD Multi-Agent Debate microservice"

RUN groupadd -r guardrails && useradd -r -g guardrails -d /app -s /bin/false guardrails

WORKDIR /app

COPY --from=builder /install /usr/local

# Copy only the MAD service source — keeps image lean
COPY multi_agent_debate/full_FinalMAD_with_judge/ ./mad/

RUN mkdir -p /app/data && chown -R guardrails:guardrails /app
USER guardrails

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production
# Default: Ollama on host. Override with VLLM_* env vars for GPU deployment.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
ENV VLLM_AGENTS_URL=http://host.docker.internal:11434/v1
ENV VLLM_JUDGE_URL=http://host.docker.internal:11434/v1
ENV AGENTS_MODEL_NAME=qwen2.5:7b
ENV JUDGE_MODEL_NAME=qwen2.5:7b
ENV SQLITE_DB_PATH=/app/data/mad.db
ENV CLAIM_CONCURRENCY=2

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/mad/health')" || exit 1

# Run from inside the mad/ subdirectory so relative imports in configs/src work
WORKDIR /app/mad
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
