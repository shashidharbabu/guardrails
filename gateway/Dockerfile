# syntax=docker/dockerfile:1
# Gateway — input guardrail service
#
# REMOTE mode (production): set PII_INFERENCE_URL + PI_INFERENCE_URL env vars
#   → gateway is CPU-only, ~1GB image, fast cold start
#   → heavy models (Qwen-7B PII, LLaMA-3.1-8B PI) served on GPU EC2 via TEI/TGI
#
# LOCAL mode (dev/fallback): leave those env vars unset
#   → validators load models in-process (slow, needs more RAM)
#
# Jailbreak classifier (shashidharbabu/roberta-jailbreak-guardrails) always runs
# locally — it is small (~500MB) and fast on CPU.
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY gateway/requirements_gateway.txt ./

# CPU-only torch — avoids pulling the 2GB CUDA wheel
# Jailbreak RoBERTa runs on CPU; PII+PI call remote GPU in production
RUN pip install --no-cache-dir --prefix=/install \
    "torch>=2.2.2" \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --prefix=/install -r requirements_gateway.txt


FROM python:3.11-slim AS runtime

LABEL maintainer="guardrails-platform"
LABEL description="SpartanGuard Gateway — input guardrail (PII + jailbreak + prompt injection)"

RUN groupadd -r guardrails && useradd -r -g guardrails -d /app -s /bin/false guardrails
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local
COPY gateway/ ./gateway/

RUN chown -R guardrails:guardrails /app
USER guardrails

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production

# Remote inference URLs — set these to point at GPU EC2 TEI/TGI servers
# ENV PII_INFERENCE_URL=http://10.0.x.x:3000
# ENV PI_INFERENCE_URL=http://10.0.x.x:3001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["gunicorn", "gateway.server:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "120"]
