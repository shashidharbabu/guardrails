FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2

# Gateway deps
COPY gateway/requirements_gateway_runtime.txt /tmp/req_gateway.txt
RUN pip install --no-cache-dir -r /tmp/req_gateway.txt

# MAD deps
COPY multi_agent_debate/multi_agent/requirements_runtime.txt /tmp/req_mad.txt
RUN pip install --no-cache-dir -r /tmp/req_mad.txt

# Install langfuse explicitly for observability
RUN pip install --no-cache-dir langfuse

COPY . /app

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "unified_pipeline:app", "--host", "0.0.0.0", "--port", "8002"]
