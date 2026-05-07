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

COPY multi_agent_debate/multi_agent/requirements_runtime.txt /app/multi_agent_debate/multi_agent/requirements_runtime.txt
RUN pip install --no-cache-dir -r /app/multi_agent_debate/multi_agent/requirements_runtime.txt

COPY . /app

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "multi_agent_debate.multi_agent.api:app", "--host", "0.0.0.0", "--port", "8001"]
