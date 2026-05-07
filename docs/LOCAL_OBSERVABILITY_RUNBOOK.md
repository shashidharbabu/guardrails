# Local observability — clear steps

Use this file in order. Replace `/path/to/guardrails-enterprise` with your real path (or run `cd` into the repo and use `$(pwd)`).

---

## Choose your path

| Goal | Read |
|------|------|
| **Docker-only: Gateway + MAD + Langfuse + Datadog** | [A. Docker-only (recommended)](#a-docker-only-recommended) |
| **Report-only eval (RAGAS JSON)** | [E. Evaluation scripts](#e-report-only-evaluation) |

---

## A. Docker-only (recommended)

This repo is intended to run observability **fully dockerized**: Gateway + MAD + Langfuse stack + Datadog Agent.

### A1. Environment

```bash
cd /path/to/guardrails-enterprise
cp .env.example .env
```

Edit `.env` and set at least:

- `DD_API_KEY` (required to see traces in Datadog APM)
- `DD_SITE` (defaults to `us5.datadoghq.com`)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (required to see MAD traces in Langfuse)

For local smoke tests without model downloads:

- `GATEWAY_STUB_VALIDATORS=true`

### A2. Run everything

```bash
docker compose up -d --build
```

Services:

- Gateway: `http://localhost:8080`
- MAD: `http://localhost:8001`
- Langfuse UI: `http://localhost:3000`
- Datadog Agent (APM): listens on `http://localhost:8126` (internal consumers use `http://datadog:8126`)

### A3. Smoke test (single command)

```bash
./scripts/smoke_observability.sh
```

The smoke test:

- calls `POST /validate`
- calls `GET /health/trace-ping` (always emits a Datadog span)
- calls `POST /mad/observability/ping` (always emits a Langfuse trace; no Ollama/Anthropic required)
- asserts 200 responses and scans logs for obvious exporter errors

### A4. Where to look

- **Datadog**: APM → Traces, filter `service:gateway env:local`. The ping span name is `gateway.observability.ping`.
- **Langfuse**: Traces (last 15 minutes). Ping span name is `mad.observability.ping`.

Note: `POST /mad/verify` may still require Ollama/Anthropic. Ping is the no-LLM check.

---

## E. Report-only evaluation

From repo root (install extra deps if scripts complain):

```bash
python3 synthetic_data/run_llm_pipeline.py --help
python3 synthetic_data/run_llm_pipeline.py --with-ragas
```

Outputs are written by `synthetic_data/pipeline/ragas_eval.py` (e.g. `ragas_summary.json` in the run output directory — see script help for paths).

---

## Other prerequisites (from `.env.example`)

- **Hugging Face**: `HF_TOKEN` for gateway models.
- **PEFT / Qwen**: see `.env.example` and `gateway/hf_peft_loader.py`.
- **MAD judge**: `JUDGE_PROVIDER=ollama` or Anthropic keys; `OLLAMA_BASE_URL` if using Ollama.

---

## Command issues we fixed in this doc

| Issue | Fix |
|--------|-----|
| `/path/to/...` | Use your real repo directory. |
| `uvicorn: command not found` | `python3 -m uvicorn ...` |
| `jq: command not found` | Pipe to `python3 -m json.tool` |
| `curl` silently empty file | Use `curl -fL` when downloading Langfuse compose |
| Langfuse “download” | Use Docker Compose, not `pip install langfuse` for the server |
