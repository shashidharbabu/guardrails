# syntax=docker/dockerfile:1
# Multi-stage production backend image.
# Stage 1: dependency builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements-backend.txt


# Stage 2: runtime image
FROM python:3.11-slim AS runtime

LABEL maintainer="guardrails-platform"
LABEL description="Enterprise Guardrails App Backend"

# Non-root user
RUN groupadd -r guardrails && useradd -r -g guardrails -d /app -s /bin/false guardrails

# Runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY multi_agent_debate/ ./multi_agent_debate/

RUN chown -R guardrails:guardrails /app
USER guardrails

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["gunicorn", "app.backend.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
