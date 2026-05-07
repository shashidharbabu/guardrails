FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (minimal). Some ML deps may require build tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (prevents pulling CUDA wheels).
# See https://pytorch.org/get-started/locally/ for wheel indices.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2

COPY gateway/requirements_gateway_runtime.txt /app/gateway/requirements_gateway_runtime.txt
RUN pip install --no-cache-dir -r /app/gateway/requirements_gateway_runtime.txt

COPY . /app

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "gateway.server:app", "--host", "0.0.0.0", "--port", "8080"]
