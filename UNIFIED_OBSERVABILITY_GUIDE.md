# Unified Observability & Implementation Guide
**Guardrails Enterprise — Gateway & MAD Pipeline**

This guide provides instructions for teammates on how the unified observability architecture works, how to configure your `.env` file, and how to start the service using Docker.

---

## 1. How the Unified Pipeline Works

The Gateway (Phase 1) and MAD (Phase 2) have been unified into a single API endpoint `/verify_unified`. 

1. **Input Guardrail (Gateway)**: The query is first evaluated by the three classifiers (PII, Jailbreak, Prompt Injection).
    - **Observability**: Traced by **Datadog APM**.
    - If the Gateway issues a `BLOCK` decision, the process terminates immediately and returns the `datadog_trace_url`.
2. **Output Guardrail (MAD)**: If the Gateway passes the query, the candidate LLM answer is run through the multi-agent debate (MAD) pipeline.
    - **Observability**: Step-by-step tracing (claim extraction, debate cycles, judging) is managed by **Langfuse**.
    - **Routing**: The system applies the **Phase 4 Confidence Scoring Engine (CSE)** formula `(0.30 × F_llm + 0.25 × (1 − H_llm) + 0.10 × relevancy + 0.35 × judge_eval_score)` to make the final routing decision (`DELIVER`, `RETRY`, `HUMAN_REVIEW`, `HARD_BLOCK`).
    - The API response will return both the `datadog_trace_url` and `langfuse_trace_url`.

> [!TIP]
> **Graceful Degradation**
> Both Datadog and Langfuse are configured to gracefully degrade. If you are developing locally without Docker or without setting the API keys, the telemetry will simply be disabled and will not crash the application.

---

## 2. Environment Variables Setup (`.env`)

Before running the system, you must configure your `.env` file at the root of the repository. Copy `.env.example` to `.env` and ensure the following essential variables are populated:

### Datadog Configuration
- `DD_API_KEY`: Your Datadog API Key.
- `DD_SITE`: e.g., `us5.datadoghq.com`.
- `DD_SERVICE`: Set to `unified_pipeline`.
- `DD_ENV`: Set to `local` or `production`.

### Langfuse Configuration
- `LANGFUSE_PUBLIC_KEY`: Your Langfuse project's public key.
- `LANGFUSE_SECRET_KEY`: Your Langfuse project's secret key.
- `LANGFUSE_BASE_URL`: If using the self-hosted Docker stack provided in this repo, set to `http://localhost:3000`. If using Langfuse Cloud, set to `https://us.cloud.langfuse.com`.

### Core Pipeline Dependencies
- `TAVILY_API_KEY`: Required for MAD agent web searches.
- `OLLAMA_BASE_URL`: Usually `http://localhost:11434/v1` (ensure you have pulled the model via `ollama pull qwen2.5:7b`).

---

## 3. Starting the Service with Docker

We have consolidated the setup into a `unified` Docker container.

### Step 1: Start the services
To start the Unified API, Datadog Agent, and the Langfuse Stack, run:
```bash
docker-compose up --build
```
> [!NOTE]
> The first time you run this, it may take a few minutes to build the container and initialize the Langfuse databases (Postgres, Clickhouse, Redis).

### Step 2: Verify it's running
The unified API will be exposed on port `8002`. You can check the health endpoint:
```bash
curl http://localhost:8002/health
```

---

## 4. Testing the System

A testing script has been provided to automatically test the deployment, the Gateway blocking logic, and the MAD routing logic.

When you pull the code, you can run this script to instantly verify your local setup:

```bash
python test_unified_pipeline.py
```

**What the script does:**
1. Checks the `/health` endpoint to verify the API is up and checks if Langfuse/Datadog are correctly initialized.
2. Sends a malicious "Jailbreak" query to verify that the Gateway correctly triggers a `BLOCK` and returns a Datadog trace link.
3. Sends a safe query to verify that the Gateway passes it to the MAD pipeline, returning both the Datadog and Langfuse trace links along with the Phase 4 Confidence Score.
