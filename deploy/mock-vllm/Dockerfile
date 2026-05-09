# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

COPY deploy/mock-vllm/server.py ./server.py

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models')" || exit 1

CMD ["python", "server.py"]
