# Langfuse Setup

Langfuse provides LLM observability: traces, token counts, latencies, and scores for every agent/judge call.

The pipeline is designed to run correctly even if Langfuse is NOT configured. It logs a warning and continues.

## Quick Start (self-hosted Docker)

```bash
git clone https://github.com/langfuse/langfuse
cd langfuse
docker compose up -d
```

Default UI: http://localhost:3000

## Configuration

In your `.env`:
```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

Get keys from http://localhost:3000 → Settings → API Keys.

## What Gets Traced

- **Judge node**: one Langfuse trace per claim, with generation (input/output/tokens/latency) and scores (v_label, judge_confidence).
- **Agent calls**: logged via Langfuse generations if `langfuse` client is passed to the pipeline functions.

## Disabling

Leave `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` blank (or unset). No errors will be raised.
