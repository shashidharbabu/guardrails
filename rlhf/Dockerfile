# syntax=docker/dockerfile:1
# Production image for the RLHF human feedback API.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

COPY rlhf/requirements-feedback.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements-feedback.txt \
 && pip install --no-cache-dir --prefix=/install "gunicorn>=23.0.0"


FROM python:3.11-slim AS runtime

LABEL maintainer="guardrails-platform"
LABEL description="SpartanGuard RLHF Feedback API"

RUN groupadd -r guardrails && useradd -r -g guardrails -d /app -s /bin/false guardrails

WORKDIR /app

COPY --from=builder /install /usr/local
COPY rlhf/ ./rlhf/

RUN mkdir -p /data \
 && chown -R guardrails:guardrails /app /data

USER guardrails

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production
ENV FEEDBACK_API_HOST=0.0.0.0
ENV FEEDBACK_API_PORT=8002
ENV MAD_DB_PATH=/data/mad.db
ENV HUMAN_FEEDBACK_LOG_PATH=/data/human_feedback_log.jsonl
ENV FEEDBACK_USE_PRESIDIO=0

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

CMD ["gunicorn", "rlhf.feedback_loop.api:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8002", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
