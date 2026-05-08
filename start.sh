#!/usr/bin/env bash
# start.sh — Launch all three services for the Guardrails Enterprise demo.
#
# Services started:
#   :8080  Gateway    (FastAPI — input guardrail)
#   :8001  MAD API    (FastAPI — output guardrail)
#   :8000  App backend (FastAPI — orchestrator + DB)
#   :5173  Frontend   (Vite dev server)
#
# Prerequisites:
#   1. Ollama running: `ollama serve` (separate terminal)
#   2. Model pulled:   `ollama pull qwen2.5:7b`
#   3. Python deps:    `pip install -r app/requirements.txt -r gateway/requirements_gateway.txt`
#   4. Node deps:      `cd app/frontend && npm install`
#
# Usage:
#   ./start.sh          — start all services
#   ./start.sh stop     — kill all managed services
#   ./start.sh gateway  — start gateway only
#   ./start.sh backend  — start app backend only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_FILE="$REPO_ROOT/.service_pids"

# ── Resolve uvicorn ───────────────────────────────────────────────────────────
# Prefer the system uvicorn; fall back to common install paths.
if command -v uvicorn &>/dev/null; then
  UVICORN="uvicorn"
elif [[ -x "$HOME/.local/bin/uvicorn" ]]; then
  UVICORN="$HOME/.local/bin/uvicorn"
elif [[ -x "/opt/homebrew/Caskroom/miniforge/base/bin/uvicorn" ]]; then
  UVICORN="/opt/homebrew/Caskroom/miniforge/base/bin/uvicorn"
else
  echo "ERROR: uvicorn not found. Run: pip install uvicorn" >&2
  exit 1
fi

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Stop all ─────────────────────────────────────────────────────────────────
stop_all() {
  if [[ -f "$PIDS_FILE" ]]; then
    while IFS= read -r pid; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && info "Stopped PID $pid"
      fi
    done < "$PIDS_FILE"
    rm -f "$PIDS_FILE"
    success "All services stopped."
  else
    warn "No .service_pids file found — nothing to stop."
  fi
}

# ── Start gateway ─────────────────────────────────────────────────────────────
start_gateway() {
  info "Starting Gateway on :8080 ..."
  cd "$REPO_ROOT"
  "$UVICORN" gateway.server:app --host 0.0.0.0 --port 8080 \
    --log-level info > "$REPO_ROOT/logs/gateway.log" 2>&1 &
  echo $! >> "$PIDS_FILE"
  success "Gateway PID $! → logs/gateway.log"
}

# ── Start MAD API ─────────────────────────────────────────────────────────────
start_mad() {
  info "Starting MAD API on :8001 (full_FinalMAD_with_judge) ..."
  # Must run from the full_FinalMAD_with_judge dir so `configs` and `src` resolve
  cd "$REPO_ROOT/multi_agent_debate/full_FinalMAD_with_judge"
  PYTHONPATH="$REPO_ROOT/multi_agent_debate/full_FinalMAD_with_judge:$REPO_ROOT/multi_agent_debate:$REPO_ROOT" \
    "$UVICORN" api:app --host 0.0.0.0 --port 8001 \
    --log-level info > "$REPO_ROOT/logs/mad.log" 2>&1 &
  echo $! >> "$PIDS_FILE"
  cd "$REPO_ROOT"
  success "MAD API PID $! → logs/mad.log"
}

# ── Start app backend ─────────────────────────────────────────────────────────
start_backend() {
  info "Starting App Backend on :8000 ..."
  cd "$REPO_ROOT"
  PYTHONPATH="$REPO_ROOT/multi_agent_debate:$REPO_ROOT/rag_folder:$REPO_ROOT" \
    "$UVICORN" app.backend.main:app --host 0.0.0.0 --port 8000 --reload \
    --log-level info > "$REPO_ROOT/logs/backend.log" 2>&1 &
  echo $! >> "$PIDS_FILE"
  success "Backend PID $! → logs/backend.log"
}

# ── Start frontend ────────────────────────────────────────────────────────────
start_frontend() {
  info "Starting Frontend on :5173 ..."
  cd "$REPO_ROOT/app/frontend"
  npm run dev > "$REPO_ROOT/logs/frontend.log" 2>&1 &
  echo $! >> "$PIDS_FILE"
  success "Frontend PID $! → logs/frontend.log"
  cd "$REPO_ROOT"
}

# ── Main ──────────────────────────────────────────────────────────────────────
CMD="${1:-all}"

mkdir -p "$REPO_ROOT/logs"
> "$PIDS_FILE"   # reset pid file

case "$CMD" in
  stop)
    stop_all
    exit 0
    ;;
  gateway)
    start_gateway
    ;;
  mad)
    start_mad
    ;;
  backend)
    start_backend
    ;;
  frontend)
    start_frontend
    ;;
  all)
    start_gateway
    sleep 3  # wait for gateway model load to begin

    start_mad
    sleep 1

    start_backend
    sleep 1

    start_frontend

    echo ""
    success "All services started."
    echo ""
    echo "  Gateway   → http://localhost:8080/docs"
    echo "  MAD API   → http://localhost:8001/docs"
    echo "  Backend   → http://localhost:8000/docs"
    echo "  Frontend  → http://localhost:5173"
    echo ""
    echo "  Logs in: $REPO_ROOT/logs/"
    echo "  Stop all: ./start.sh stop"
    ;;
  *)
    error "Unknown command: $CMD"
    echo "Usage: $0 [all|gateway|mad|backend|frontend|stop]"
    exit 1
    ;;
esac
