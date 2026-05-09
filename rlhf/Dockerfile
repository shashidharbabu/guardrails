# syntax=docker/dockerfile:1
# RLHF Feedback API — human review queue + batch scoring
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY rlhf/requirements-feedback.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements-feedback.txt
# Download spacy model for presidio (PII shaping in feedback)
RUN /install/bin/python -m spacy download en_core_web_lg || true


FROM python:3.11-slim AS runtime

LABEL maintainer="guardrails-platform"
LABEL description="SpartanGuard RLHF Feedback API"

RUN groupadd -r guardrails && useradd -r -g guardrails -d /app -s /bin/false guardrails

WORKDIR /app

COPY --from=builder /install /usr/local
COPY rlhf/ ./rlhf/

RUN chown -R guardrails:guardrails /app
USER guardrails

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

CMD ["uvicorn", "rlhf.feedback_loop.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8002", \
     "--workers", "2"]
