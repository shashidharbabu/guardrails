# syntax=docker/dockerfile:1
# MAD API — Multi-Agent Debate verification service
FROM public.ecr.aws/docker/library/python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# CPU-only torch — avoids pulling the 2GB CUDA wheel
RUN pip install --no-cache-dir --prefix=/install \
    "torch>=2.2.2" \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY multi_agent_debate/full_FinalMAD_with_judge/requirements.txt ./mad_requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r mad_requirements.txt

# Install embedding + qdrant deps
RUN pip install --no-cache-dir --prefix=/install \
    transformers>=4.40.0 \
    huggingface-hub>=0.23.0 \
    qdrant-client>=1.9.0 \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.29.0 \
    gunicorn>=22.0 \
    python-dotenv>=1.0 \
    httpx>=0.27.0 \
    tenacity>=8.0 \
    boto3>=1.34.0


FROM public.ecr.aws/docker/library/python:3.11-slim AS runtime

LABEL maintainer="guardrails-platform"
LABEL description="SpartanGuard MAD API — Multi-Agent Debate"

RUN groupadd -r guardrails && useradd -r -g guardrails -d /app -s /bin/false guardrails
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local

# Copy the entire multi_agent_debate tree so the rag package is importable
COPY multi_agent_debate/ ./multi_agent_debate/

RUN chown -R guardrails:guardrails /app
USER guardrails

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production
ENV PYTHONPATH=/app/multi_agent_debate

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/mad/health')" || exit 1

CMD ["sh", "-c", "python multi_agent_debate/full_FinalMAD_with_judge/sagemaker_proxy.py & gunicorn full_FinalMAD_with_judge.api:app -k uvicorn.workers.UvicornWorker --workers 2 --bind 0.0.0.0:8001 --timeout 300"]
