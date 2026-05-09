# syntax=docker/dockerfile:1
# Production image for the full_FinalMAD_with_judge FastAPI service.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

COPY multi_agent_debate/full_FinalMAD_with_judge/requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
 && pip install --no-cache-dir --prefix=/install "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0" "gunicorn>=23.0.0"


FROM python:3.11-slim AS runtime

LABEL maintainer="guardrails-platform"
LABEL description="SpartanGuard MAD API"

RUN groupadd -r guardrails && useradd -r -g guardrails -d /app -s /bin/false guardrails

WORKDIR /app

COPY --from=builder /install /usr/local
COPY multi_agent_debate/full_FinalMAD_with_judge/ ./multi_agent_debate/full_FinalMAD_with_judge/

RUN mkdir -p /data \
 && chown -R guardrails:guardrails /app /data

USER guardrails

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production
ENV SQLITE_DB_PATH=/data/mad.db
ENV LANGCHAIN_CACHE_DB=/data/.langchain_cache.db
ENV CLAIM_CONCURRENCY=2

WORKDIR /app/multi_agent_debate/full_FinalMAD_with_judge

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/mad/health')" || exit 1

CMD ["gunicorn", "api:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8001", \
     "--timeout", "600", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
