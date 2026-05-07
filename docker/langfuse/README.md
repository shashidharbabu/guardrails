# Langfuse (self-hosted) — step by step

You do **not** install Langfuse with `pip` for the server. The server runs in **Docker**. Your Python app only needs the `langfuse` package (already listed in `multi_agent_debate/multi_agent/requirements.txt`).

## What you need first

1. **Docker Desktop** for Mac (or Docker Engine on Linux) installed and **running** (whale icon idle on Mac).
2. In a terminal, these must work:
   - `docker version`
   - `docker compose version`  
   If `docker compose` fails, try `docker-compose version` (older install) and use `docker-compose` instead of `docker compose` below.

## Download the official Compose file

Run from **any** folder where you want the Langfuse stack files (example uses a folder inside this repo):

```bash
cd /path/to/guardrails-enterprise
mkdir -p docker/langfuse/stack
cd docker/langfuse/stack
```

Download the compose file (fail loudly if the URL breaks):

```bash
curl -fL -o docker-compose.yml \
  https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml
```

- **`-f`**: exit with error on 404/500 (so you know the download failed).
- **`-L`**: follow redirects.

If `curl` fails (corporate proxy, SSL), open the same URL in a browser, save the file as `docker-compose.yml` in `docker/langfuse/stack/`.

## Configure secrets (required)

Langfuse’s compose file uses placeholders marked **CHANGEME** in comments. Before `up`, read the upstream guide and set a minimal `.env` beside `docker-compose.yml`:

- Official self-host docs: [Langfuse self-hosting](https://langfuse.com/docs/deployment/self-host)
- You must set strong values for at least: database password, Redis password, MinIO secrets, `SALT`, `ENCRYPTION_KEY`, `NEXTAUTH_SECRET` (names appear in the downloaded `docker-compose.yml`).

If you start the stack without fixing secrets, services may stay unhealthy.

## Start Langfuse

Still in `docker/langfuse/stack/` (where `docker-compose.yml` is):

```bash
docker compose up -d
```

If that command is not found:

```bash
docker-compose up -d
```

Check containers:

```bash
docker compose ps
```

Wait until **`langfuse-web`** is up. Open **http://localhost:3000** in a browser.

## API keys for this repo

1. In the Langfuse UI, sign up / log in.
2. Create a **project**.
3. Go to project **Settings → API keys**.
4. Copy **public** and **secret** keys into the **repo root** `.env`:

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

The Python SDK reads these automatically when you run the MAD API.

## Stop Langfuse

From the same directory:

```bash
docker compose down
```

## Troubleshooting

| Problem | What to try |
|--------|-------------|
| `curl: (22) The requested URL returned error: 404` | Langfuse moved the file; open [their GitHub](https://github.com/langfuse/langfuse) and use the `docker-compose.yml` from the branch their docs mention. |
| `docker: command not found` | Install Docker Desktop and start it. |
| `docker compose` not found | Use `docker-compose` or update Docker Desktop. |
| Port 3000 already in use | Change the host port in `docker-compose.yml` for `langfuse-web` and set `LANGFUSE_BASE_URL` to match (e.g. `http://localhost:3001`). |
| UI loads but traces empty | Confirm `.env` keys, `LANGFUSE_BASE_URL` matches the URL you use in the browser, and run a MAD request after starting the API. |

Next: run the MAD API and smoke test — [docs/LOCAL_OBSERVABILITY_RUNBOOK.md](../docs/LOCAL_OBSERVABILITY_RUNBOOK.md).
