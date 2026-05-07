#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Bringing stack up (detached)."
docker compose up -d --build

echo "Waiting for APIs to respond."
for i in {1..60}; do
  if docker compose exec -T gateway curl -fsS "http://127.0.0.1:8080/health" >/dev/null 2>&1 && \
     docker compose exec -T mad curl -fsS "http://127.0.0.1:8001/mad/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Calling Gateway /validate (stub mode ok)."
docker compose exec -T gateway curl -fsS -X POST "http://127.0.0.1:8080/validate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","session_id":"smoke-1"}' >/dev/null

echo "Calling Gateway trace ping."
docker compose exec -T gateway curl -fsS "http://127.0.0.1:8080/health/trace-ping" >/dev/null

echo "Calling MAD Langfuse ping (no LLM required)."
docker compose exec -T mad curl -fsS -X POST "http://127.0.0.1:8001/mad/observability/ping" >/dev/null

echo "Checking for obvious exporter errors in logs."
LOGS="$(docker compose logs --no-color gateway mad 2>/dev/null || true)"
python3 - <<'PY' "$LOGS"
import re, sys
logs = sys.argv[1]
pat = re.compile(r"(traceback|error exporting|failed to send|connection refused|langfuse\].*(warning|failed))", re.I)
if pat.search(logs):
    print("Found error-like patterns in logs.")
    sys.exit(2)
sys.exit(0)
PY
PY_EXIT=$?

if [[ "$PY_EXIT" -ne 0 ]]; then
  echo "Found error-like patterns in logs. Showing recent logs."
  docker compose logs --no-color --tail 200 gateway mad datadog
  exit 1
fi

echo "OK: endpoints returned 200 and no obvious exporter errors detected."
