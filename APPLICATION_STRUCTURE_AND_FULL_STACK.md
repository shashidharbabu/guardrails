# Guardrails App — Repository Structure, Behavior, and Full-Stack Wiring

This document is intended for **code review and “enterprise readiness” assessment**. It describes what the system does, how directories relate to each other, and exactly how the **browser UI**, **app backend**, **gateway**, **LLM runtime**, and **MAD (Multi-Agent Debate)** pieces connect.

---

## 1. What the whole application does

**Product intent:** An **enterprise AI guardrails** demonstration platform. User queries flow through:

1. **Input guardrail (Gateway)** — Classifies each user prompt for prompt injection, jailbreak-like behavior, and PII; applies a weighted decision engine. Outcomes are typically **PASS**, **BLOCK**, or **ESCALATE** (plus related scores and optional entity spans).
2. **Downstream LLM** — If the gateway does not **BLOCK**, the orchestrator calls a **local Ollama** instance using the OpenAI-compatible chat completions API and a fixed system prompt framing the model as a compliance assistant.
3. **Output guardrail (MAD)** — After an LLM answer exists, **Multi-Agent Debate** runs (in-process in the app backend by default): claim extraction, adversarial cycles, judge, and a **routing decision** (e.g. **DELIVER**, **RETRY**, **HARD_BLOCK**, **HUMAN_REVIEW**) with evidence and debate structure. MAD is grounded on a **regulatory-style corpus** (JSONL / RAG path depending on configuration), not used to answer the user directly, but to **verify** the LLM answer.
4. **Persistence & UI** — Sessions (query, gateway payload, LLM text, MAD output, timings) are stored in **SQLite** (`app/app_sessions.db`). The React app lists sessions, shows traces, supports **feedback** and **analytics** summaries, and can drive **gateway** configuration through the backend proxy.

**Adjacent / batch infrastructure (same repo, not required for the Vite UI to load):**

- **Airflow** (`dags/`, `plugins/`, `docker/docker-compose.yml`) for data/PII NER style pipelines.
- **`rag_folder/`**, **`synthetic_data/`**, **`finetuning/`**, **`rlhf/`**, **`confidence/`** — research and pipeline code paths described in root `README.md` and `docs/`.
- **`guardrails_enterprise/`** — Python **SDK-style** `GuardrailPipeline` that can orchestrate gateway + LLM + MAD over HTTP (parallel to the FastAPI app backend pattern).

---

## 2. Full-stack topology (services and ports)

| Component | Port (typical) | Role |
|-----------|----------------|------|
| **Frontend** (Vite + React) | `5173` | SPA; dev server proxies `/api/*` to the app backend. |
| **App backend** (FastAPI) | `8000` | Orchestrates pipeline, owns **app SQLite**, exposes REST API to the UI. |
| **Gateway** (FastAPI) | `8080` | Input guardrail; `/validate` etc. **Note:** Root `docker/docker-compose.yml` maps **Airflow webserver** to host `8080` as well — do not run that compose stack on the same host port as the gateway without changing one of them. |
| **MAD API** (FastAPI, optional standalone) | `8001` | `multi_agent.api:app` — direct HTTP access to MAD; **the app orchestrator does not call this URL by default**; it imports `run_mad` in-process. |
| **Ollama** | `11434` | LLM inference (`/v1/chat/completions`). |

**Convenience script:** `start.sh` at repo root can start gateway, MAD API, app backend, and frontend (expects Ollama running separately).

---

## 3. How the stack connects (request path)

### 3.1 Browser → app backend

- The SPA uses **relative** paths such as `/api/sessions`, `/api/query` (see `app/frontend/src/api/client.js`).
- Vite proxies `/api` to `http://localhost:8000` (`app/frontend/vite.config.js`).
- The app backend enables **CORS** for `http://localhost:5173` and `http://localhost:4173` (`app/backend/main.py`).

### 3.2 App backend → gateway and Ollama

Implemented in `app/backend/pipeline.py`:

1. **POST** `http://localhost:8080/validate` with `{"text": "<user query>"}`.
2. If decision is not `BLOCK`, **POST** `http://localhost:11434/v1/chat/completions` (Ollama) with `messages`, model name, `stream: false`.
3. **Insert** a row into SQLite with gateway + LLM fields; MAD fields initially null.
4. If there is a usable LLM answer, schedule **`asyncio.create_task`** to run **`run_mad(query, llm_answer)`** in a thread pool, then **`db.update_session_mad`** to patch MAD results.

**Python import path:** `app/backend/main.py` inserts `multi_agent_debate` on `sys.path`; `start.sh` sets `PYTHONPATH` to prefer `multi_agent_debate` and `rag_folder` so `from multi_agent.mad_pipeline import run_mad` resolves to the **debate** package’s `multi_agent` implementation.

### 3.3 UI → gateway (admin / playground)

`app/backend/routers/gateway.py` **proxies** `/api/gateway/*` to `http://localhost:8080` (validate, health, logs, stats, config CRUD). The Gateway page in the frontend uses these routes so the UI does not talk to port 8080 directly (avoids CORS and centralizes errors).

---

## 4. REST API surface (app backend, `8000`)

| Prefix / route | Purpose |
|----------------|---------|
| `GET /health` | Liveness for the app backend. |
| `GET /api/sessions`, `GET /api/sessions/{id}` | List / get sessions (JSON includes parsed `gateway_payload` and `mad_output`). |
| `GET /api/sessions/{id}/cse` | Intended CSE breakdown from stored MAD output. **Implementation note for reviewers:** `db.get_session()` renames `mad_output_json` → `mad_output` in `db._deserialize_session`, while this handler still reads `mad_output_json`; confirm whether this endpoint returns 404 incorrectly unless aligned. |
| `POST /api/query` | Body: `{ "query", "llm_model" }` — runs full pipeline, returns session (MAD may still be in flight). |
| `GET/POST /api/feedback`, `GET /api/feedback/export` | Feedback CRUD + CSV export. |
| `GET /api/analytics/summary` | Aggregates from SQLite (counts, avg latency, MAD routing distribution). |
| `/api/gateway/*` | Reverse proxy to gateway `:8080`. |

**Mock mode:** If `VITE_MOCK_API=true`, the frontend uses in-browser mock data and does not call the backend (`app/frontend/src/api/client.js`).

---

## 5. Data stores

| Store | Location / note |
|-------|------------------|
| **App sessions + feedback** | `app/app_sessions.db` — tables `sessions`, `feedback` (`app/backend/db.py`). |
| **Gateway audit** | Gateway service maintains its own logging (see `gateway/logger.py`); not the same file as app sessions. |
| **MAD GRPO / research storage** | `multi_agent_debate/mad_store.db` (and related WAL/SHM when used) — used by the debate pipeline’s storage layer when enabled; distinct from the UI’s `app_sessions.db`. |

---

## 6. Exact repository structure (high-signal)

> **Legend:** Folders below exist under the repo root unless noted. Omitted: `node_modules`, `__pycache__`, large notebook copies, etc.

```
guardrails-app-ui/
├── app/                              # ★ Full-stack “product” slice (UI + API + orchestration)
│   ├── backend/
│   │   ├── main.py                   # FastAPI app, CORS, router mount, DB init on startup
│   │   ├── db.py                     # SQLite schema + CRUD + analytics queries
│   │   ├── pipeline.py               # Gateway → Ollama → SQLite → background MAD
│   │   └── routers/
│   │       ├── sessions.py           # /api/sessions, /api/query
│   │       ├── gateway.py            # Proxy /api/gateway → :8080
│   │       ├── analytics.py          # /api/analytics/summary
│   │       └── feedback.py           # /api/feedback + CSV export
│   ├── frontend/                     # Vite + React SPA
│   │   ├── package.json
│   │   ├── vite.config.js            # /api → localhost:8000
│   │   └── src/
│   │       ├── main.jsx
│   │       ├── App.jsx               # Routes: /, /new, /sessions/:id, /gateway, …
│   │       ├── api/client.js         # fetch + optional MOCK
│   │       ├── pages/                # Conversations, NewQuery, SessionTrace, Gateway, Analytics, Evaluation, Feedback, Settings
│   │       └── components/         # Layout, MADDebate, GatewaySpans, tables, etc.
│   ├── requirements.txt              # fastapi, uvicorn, httpx, dotenv, …
│   └── app_sessions.db               # Runtime SQLite (gitignored or local artifact)
├── start.sh                          # Launch gateway :8080, MAD API :8001, backend :8000, frontend :5173
├── gateway/                          # Input guardrail FastAPI service + validators + decision engine
├── multi_agent/                      # Alternate / legacy MAD tree (see README); may duplicate concepts
├── multi_agent_debate/               # ★ Primary MAD implementation used by start.sh PYTHONPATH
│   ├── multi_agent/                  # Package name `multi_agent` when PYTHONPATH includes parent
│   └── rag/                          # Embeddings / retriever helpers for debate path
├── rag_folder/                       # RAG pipeline code (eval, retriever, embedder, …)
├── guardrails_enterprise/            # SDK-style GuardrailPipeline (HTTP to gateway + Ollama + MAD patterns)
├── config/                           # e.g. pipeline_config.yaml (Airflow / NER / GCS oriented)
├── dags/, plugins/                   # Airflow DAG + plugins
├── docker/docker-compose.yml         # Airflow + Postgres (port 8080 = Airflow UI in compose file)
├── docs/                             # Project documentation
├── synthetic_data/                   # Dataset generation pipelines
├── scripts/                          # Airflow helper scripts
├── README.md                         # High-level multi-phase roadmap
├── pyproject.toml, requirements.txt  # Root Python metadata/deps
└── APPLICATION_STRUCTURE_AND_FULL_STACK.md   # ← this file
```

---

## 7. Frontend routes (user-facing)

Defined in `app/frontend/src/App.jsx` (all wrapped in `Layout`):

| Path | Page |
|------|------|
| `/` | Conversations (session list) |
| `/new` | New query submission |
| `/sessions/:id` | Session trace (gateway + LLM + MAD visualization) |
| `/gateway` | Gateway health / validation / config via backend proxy |
| `/analytics` | Aggregated metrics from `/api/analytics/summary` |
| `/evaluation` | Evaluation-oriented UI (metrics/constants in `src/constants/evalMetrics.js`) |
| `/feedback` | Feedback list and submission |
| `/settings` | Settings UI |

**Stack:** React 18, React Router 6, Vite 5, Tailwind 3, Framer Motion (`app/frontend/package.json`).

---

## 8. Security, operations, and “enterprise” review hints

Use this as a **checklist for reviewers**; items are factual observations from the codebase, not judgments.

- **Authentication / authorization:** The described UI and `/api` routes are **not** wrapped in app-level auth in `app/backend/main.py`; enterprise deployments would need identity, RBAC, and network controls.
- **Secrets:** `.env` / `.env.example` pattern exists; gateway loads env for model paths (`gateway/server.py`).
- **Transport:** Local dev uses HTTP on localhost; production would require TLS termination and hardened CORS.
- **Multi-tenant / scaling:** App persistence is **file-backed SQLite** with synchronous `sqlite3` usage — fine for demo/single node; enterprise scale would imply connection pooling, HA DB, migrations, and idempotency for `POST /api/query`.
- **Gateway vs Airflow port clash:** Compose maps Airflow to **8080**; local gateway also uses **8080** — environment separation or port remapping required.
- **Partial async semantics:** `POST /api/query` returns after gateway + LLM + initial DB write; **MAD may complete later** — UI should tolerate null MAD fields until refresh (as the trace page patterns imply).
- **Observability:** Dependencies include `langfuse` in `app/requirements.txt`; orchestrator references `langfuse_trace_id` column — verify end-to-end wiring if claiming production observability.
- **Testing:** Gateway has `gateway/tests/`; full-stack E2E coverage should be verified separately for `app/backend` and `app/frontend`.

---

## 9. Related documentation in-repo

- Root **`README.md`** — phased roadmap (Gateway, RAG, MAD, confidence, GRPO, synthetic data).
- **`docs/`** — setup and pipeline documentation.
- **`MAD_SETUP_GUIDE.md`** — MAD service setup.
- **`multi_agent_debate/CLAUDE.md`** — deep architectural context for MAD and research phases (team project notes).

---

## 10. One-line summary for reviewers

**This repository is a research-and-demo-oriented “guardrails platform”**: a **React** SPA talks to a **FastAPI** orchestrator that calls a separate **Gateway** service and **Ollama**, persists sessions in **SQLite**, and enriches them with **in-process MAD** verification — plus separate **Airflow**, **RAG**, and **training** trees in the same monorepo. Assess **enterprise readiness** against your standards for **identity, data governance, HA, DR, SLOs, audit, and deployment isolation** — many of those concerns are **not fully implemented** in the `app/` slice alone.
