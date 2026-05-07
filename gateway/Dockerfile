# syntax=docker/dockerfile:1
# Production gateway service image.
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY gateway/requirements_gateway.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements_gateway.txt


FROM python:3.11-slim AS runtime

LABEL maintainer="guardrails-platform"
LABEL description="Enterprise Guardrails Gateway Service"

RUN groupadd -r guardrails && useradd -r -g guardrails -d /app -s /bin/false guardrails
RUN apt-get update && apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local
COPY gateway/ ./gateway/

RUN chown -R guardrails:guardrails /app
USER guardrails

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["gunicorn", "gateway.server:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "120"]
