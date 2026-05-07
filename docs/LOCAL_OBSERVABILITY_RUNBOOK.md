# Local observability — clear steps

Use this file in order. Replace `/path/to/guardrails-enterprise` with your real path (or run `cd` into the repo and use `$(pwd)`).

---

## Choose your path

| Goal | Read |
|------|------|
| **I only want to run Gateway + MAD and see responses** | [A. Minimal test (no Langfuse, no Datadog)](#a-minimal-test-no-langfuse-no-datadog) |
| **I want Langfuse traces for MAD** | [B. Langfuse](#b-langfuse-docker) then [D. Run services and smoke tests](#d-run-services-and-smoke-tests) |
| **I want Datadog APM for the Gateway** | [C. Datadog Agent](#c-datadog-agent) then [D](#d-run-services-and-smoke-tests) |
| **Report-only eval (RAGAS JSON)** | [E. Evaluation scripts](#e-report-only-evaluation) |

---

## A. Minimal test (no Langfuse, no Datadog)

Best first step to verify the apps work.

### A1. Environment

```bash
cd /path/to/guardrails-enterprise
cp .env.example .env
```

Edit `.env` and set at least:

- `HF_TOKEN` (for Hugging Face gateway models, if needed)
- `LANGFUSE_ENABLED=false`
- `DD_TRACE_ENABLED=false`
- `GATEWAY_TELEMETRY_STDOUT=true`  
  (optional: prints JSON span lines to the terminal so you can see gateway timing without Datadog)

### A2. Install Python dependencies

```bash
pip install -r gateway/requirements_gateway.txt
pip install -r multi_agent_debate/multi_agent/requirements.txt
```

Use `pip3` if your system uses that name.

### A3. Terminal 1 — Gateway

```bash
cd /path/to/guardrails-enterprise
# No ddtrace required for this minimal path:
python3 -m uvicorn gateway.server:app --host 127.0.0.1 --port 8080
```

### A4. Terminal 2 — Smoke test Gateway

Without `jq` (works everywhere):

```bash
curl -sS -X POST "http://127.0.0.1:8080/validate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","session_id":"local-1"}' | python3 -m json.tool
```

You should see JSON with `decision`, `gateway_score`, etc.

### A5. Terminal 3 — MAD API (optional)

Judge needs Ollama and/or Anthropic per `.env`. For a quick try, set `JUDGE_PROVIDER=ollama` and run Ollama locally.

```bash
cd /path/to/guardrails-enterprise
PYTHONPATH=. python3 -m uvicorn multi_agent_debate.multi_agent.api:app --host 127.0.0.1 --port 8001
```

### A6. Terminal 4 — Smoke test MAD

```bash
curl -sS -X POST "http://127.0.0.1:8001/mad/verify" \
  -H "Content-Type: application/json" \
  -d '{"query":"Test question?","llm_answer":"Test answer."}' | python3 -m json.tool
```

If this errors (model / Ollama / judge), fix `.env` and dependencies — observability is optional for this step.

---

## B. Langfuse (Docker)

Langfuse is **not** downloaded with `git clone` of this repo. You run their **Docker Compose** stack, then paste API keys into `.env`.

**Follow the detailed commands and troubleshooting in:**  
[docker/langfuse/README.md](../docker/langfuse/README.md)

Summary:

1. Install **Docker Desktop** (or Docker Engine).
2. Create a folder, download Langfuse’s `docker-compose.yml` with `curl -fL` (see that README if `curl` fails).
3. Add a proper `.env` next to it (secrets — see Langfuse docs).
4. `docker compose up -d`
5. Open `http://localhost:3000`, create API keys, put them in **this repo’s** root `.env` as `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`.

---

## C. Datadog Agent

You do **not** download “Datadog” into this repo. You:

1. **Create a Datadog account** (trial is fine): [Datadog](https://www.datadoghq.com/)
2. **Install the Datadog Agent** on your Mac using their installer (Datadog → Integrations → Agent → Mac).
3. During setup, paste your **Datadog API key** from the Datadog website.
4. In the Agent config, ensure **APM** is enabled and traces can be received (default trace port **8126** on the host where the Agent runs).  
   Exact file locations are in Datadog’s “Mac OS X Agent” documentation (search “APM enabled site:docs.datadoghq.com”).

In **this repo’s** `.env` (for the Python process that runs the Gateway):

```env
DD_SERVICE=gateway
DD_ENV=local
DD_TRACE_ENABLED=true
DD_TRACE_AUTO_INSTRUMENT=true
DD_AGENT_HOST=127.0.0.1
DD_SITE="us5.datadoghq.com" \
DD_LLMOBS_ENABLED=1 \
DD_LLMOBS_ML_APP=guardrails \
DD_API_KEY=873f768b3d5c554ee183c4365a5e5228 \
ddtrace-run <your application command>
```

If the Agent runs in Docker and your Gateway runs on the host Mac, you may need `DD_AGENT_HOST=host.docker.internal` **only if** the Agent is inside a container; for the standard Mac Agent on the host, `127.0.0.1` is typical.

### C1. Run Gateway **with** Datadog tracing

Install deps (includes `ddtrace`):

```bash
pip install -r gateway/requirements_gateway.txt
```

Run:

```bash
cd /path/to/guardrails-enterprise
ddtrace-run python3 -m uvicorn gateway.server:app --host 127.0.0.1 --port 8080
```

If `ddtrace-run` is not found, use:

```bash
python3 -m ddtrace.commands.ddtrace_run python3 -m uvicorn gateway.server:app --host 127.0.0.1 --port 8080
```

(or `python -m ddtrace_run` depending on `ddtrace` version — check `pip show ddtrace`).

### C2. If you skip Datadog

No account needed:

```env
DD_TRACE_ENABLED=false
GATEWAY_TELEMETRY_STDOUT=true
```

Run Gateway with plain `python3 -m uvicorn ...` (no `ddtrace-run`).

---

## D. Run services and smoke tests

### Gateway (with optional Datadog)

```bash
cd /path/to/guardrails-enterprise
ddtrace-run python3 -m uvicorn gateway.server:app --host 127.0.0.1 --port 8080
# or without Datadog:
# python3 -m uvicorn gateway.server:app --host 127.0.0.1 --port 8080
```

### MAD API (with optional Langfuse)

```bash
cd /path/to/guardrails-enterprise
PYTHONPATH=. python3 -m uvicorn multi_agent_debate.multi_agent.api:app --host 127.0.0.1 --port 8001
```

**Why `PYTHONPATH=.`?**  
So Python resolves the package `multi_agent_debate` from the repo root. **Why `python3 -m uvicorn`?**  
Avoids “uvicorn not found” if the script isn’t on your PATH.

### Smoke tests

Gateway:

```bash
curl -sS -X POST "http://127.0.0.1:8080/validate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","session_id":"local-session-1"}' | python3 -m json.tool
```

MAD:

```bash
curl -sS -X POST "http://127.0.0.1:8001/mad/verify" \
  -H "Content-Type: application/json" \
  -d '{"query":"Is AES-256 required for HIPAA ePHI at rest?","llm_answer":"HIPAA mandates AES-256 for all servers."}' | python3 -m json.tool
```

### Where to look

- **Datadog**: Web UI → APM → Traces → service `gateway`, env `local`. If empty, open **`GET http://127.0.0.1:8080/health/observability`** and follow the `hint` (Agent on port **8126**, use **`ddtrace-run`**, optional **`DD_TRACE_AGENT_URL=http://127.0.0.1:8126`**).
- **Langfuse**: Web UI → Traces (time range **Last 15 minutes**). After starting MAD, call **`GET http://127.0.0.1:8001/mad/observability/status`**, then **`POST http://127.0.0.1:8001/mad/observability/ping`** — this sends a minimal trace (`mad.observability.ping`) without running the full pipeline. Keys in `.env` must be for the **same project** you have open in the UI; `LANGFUSE_BASE_URL` should match how you open the UI (e.g. `http://localhost:3000`).

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
