# Phase 2 — Agent Monitoring Integration Plan

**Project:** Guardrails Gateway · SJSU CS298B · Team 3  
**Status:** Pending integration — being built externally by Nakshatra  
**Depends on:** MAD pipeline (Phase 3) + CSE (Phase 3) running and writing to SQLite

---

## 1. What Agent Monitoring Is

Agent Monitoring is the **observability layer** for the running MAD pipeline. It sits
alongside (not inside) the pipeline and answers three questions continuously:

1. **Health** — Are Agents A, B, and the Judge responding within latency budgets?
2. **Quality** — Are CSE scores trending up or down over time? Are HARD_BLOCK rates anomalous?
3. **Drift** — Are routing distributions (DELIVER / RETRY / HUMAN_REVIEW / HARD_BLOCK) shifting,
   indicating the LLM or RAG corpus is degrading?

It does **not** change any verdict or score. It observes and alerts only.

---

## 2. Integration Points in This Codebase

The monitoring module will consume data from three places — no code changes required to
the main MAD pipeline to support this:

### 2a. SQLite `queries` table (primary source)

```sql
-- The monitoring module reads these columns:
SELECT
    timestamp,
    final_cse_score,
    routing_decision,
    cse_version,
    cse_f_llm,
    cse_h_llm,
    cse_relevancy,
    cse_judge_eval
FROM queries
ORDER BY timestamp DESC;
```

This gives a full time-series of every MAD run with CSE breakdown.

**Path:** Same `DB_PATH` as the main pipeline (env var `MAD_DB_PATH`, default `mad_store.db`).

### 2b. Gateway audit log (secondary source)

The Gateway writes a separate SQLite DB (`gateway/audit_log.db`) with every input
classification result (PII/JB/PI scores, gateway decision). Monitoring can correlate
gateway decision rate with downstream CSE failure rate.

### 2c. FastAPI `/mad/cse/{query_id}` endpoint

```
GET /mad/cse/{query_id}
```

Defined in `multi_agent_debate/multi_agent/api.py`. Returns the full CSE breakdown
for any completed query. The monitoring dashboard can call this on demand for drill-down.

---

## 3. Metrics to Compute

### Per-run (already in DB, just read)
| Metric | Source column | Threshold to alert |
|--------|--------------|-------------------|
| CSE final score | `final_cse_score` | < 0.4 (HUMAN_REVIEW zone) |
| Routing decision | `routing_decision` | HARD_BLOCK rate > 20% |
| CSE version | `cse_version` | v0.1 rate > 50% (DeepEval falling back too often) |
| Faithfulness | `cse_f_llm` | < 0.5 sustained over 10 runs |
| Hallucination rate | `cse_h_llm` | > 0.4 sustained over 10 runs |
| Relevancy | `cse_relevancy` | < 0.4 sustained over 10 runs |

### Aggregate / sliding window (computed by monitoring module)
| Metric | Computation | Alert threshold |
|--------|-------------|----------------|
| HARD_BLOCK rate (1hr) | count(HARD_BLOCK) / count(total) | > 0.20 |
| DELIVER rate (1hr) | count(DELIVER) / count(total) | < 0.30 (regression) |
| Mean CSE score (24hr rolling) | AVG(final_cse_score) | Drops > 0.15 vs prev 24hr |
| DeepEval fallback rate | count(cse_version='v0.1') / total | > 0.50 (Ollama unreliable) |
| P95 pipeline latency | needs timestamp from API response | > 120s |
| Judge score min (material claims) | from `judge_verdicts` table | 0.0 cluster spikes |

---

## 4. Recommended Architecture for the Monitoring Module

```
monitoring/
├── __init__.py
├── collector.py       ← reads SQLite on a schedule, computes rolling metrics
├── alerts.py          ← threshold checks, sends notifications (Slack / email / log)
├── dashboard.py       ← Streamlit or FastAPI /metrics endpoint (for UI or Grafana)
├── models.py          ← MonitoringSnapshot, Alert dataclasses
└── README.md
```

The collector should run as a **sidecar** (separate process or thread), polling the
SQLite DB every 60 seconds. It does not share a connection with the main pipeline —
SQLite WAL mode (already enabled in `storage.py`) makes this safe.

### Suggested collector loop

```python
# pseudocode — monitoring/collector.py
import time, sqlite3

def collect_loop(db_path: str, interval_s: int = 60):
    while True:
        snapshot = compute_snapshot(db_path)
        check_alerts(snapshot)
        persist_snapshot(snapshot)   # optional: write to a separate monitoring.db
        time.sleep(interval_s)

def compute_snapshot(db_path: str) -> MonitoringSnapshot:
    con = sqlite3.connect(db_path)
    # last 1 hour
    rows = con.execute("""
        SELECT routing_decision, final_cse_score, cse_version,
               cse_f_llm, cse_h_llm, cse_relevancy, cse_judge_eval
        FROM queries
        WHERE timestamp > datetime('now', '-1 hour')
    """).fetchall()
    con.close()
    return _aggregate(rows)
```

---

## 5. Integration Handoff Checklist

When your friend is ready to integrate, here is what they need from this repo:

| Item | Location | Notes |
|------|----------|-------|
| DB path | `from multi_agent.config import DB_PATH` | Respects `MAD_DB_PATH` env var |
| Schema | `multi_agent_debate/multi_agent/storage.py` | `queries` table columns |
| CSE API | `GET /mad/cse/{query_id}` in `api.py` | FastAPI, port 8001 |
| Column names | See Section 2a above | All `cse_*` columns |
| SQLite WAL mode | Already enabled in `storage._conn()` | Safe for concurrent reads |
| Test DB | `multi_agent_debate/mad_store.db` | Can use this for dev/test |

The monitoring module should **never** write to the `queries`, `claims`, `attacks`,
or `judge_verdicts` tables. Those are owned by the MAD pipeline. Monitoring can create
its own separate tables in a separate `monitoring.db` file.

---

## 6. Phase 2 Deployment Integration

Once ready, wire monitoring into the startup script:

```bash
# start_services.sh (Phase 2 addition)
# Start MAD API
uvicorn multi_agent.api:app --port 8001 &

# Start agent monitoring sidecar
python -m monitoring.collector --interval 60 &

# Start monitoring dashboard (optional)
streamlit run monitoring/dashboard.py --server.port 8502 &
```

Or add it to `docker/docker-compose.yml` as a new service:

```yaml
  monitoring:
    build: .
    command: python -m monitoring.collector
    environment:
      MAD_DB_PATH: /data/mad_store.db
    volumes:
      - mad_data:/data
    depends_on:
      - mad_api
```

---

## 7. Questions to Answer Before Integration

1. **Alert channel**: Slack webhook? Email? Log file only? (needs a decision from Nakshatra)
2. **Dashboard**: Streamlit (simpler) or Grafana (heavier but better for time-series)?
3. **Historical retention**: How long to keep the `monitoring.db` snapshots?
4. **Latency tracking**: Does the monitoring module get latency data from the MAD API
   response, or does `mad_pipeline.run_mad()` need to emit timing signals?
   — Currently `run_mad()` does not return elapsed time. A `duration_ms` column in
   `queries` would help. This is a **small, non-breaking schema addition** via
   `_add_column_if_missing`.

---

## 8. Open TODO (for when Nakshatra is ready)

- [ ] Add `duration_ms` column to `queries` table (1-line change in `storage.py`)
- [ ] Decide alert channel and implement `monitoring/alerts.py`
- [ ] Implement `monitoring/collector.py` with rolling metrics
- [ ] Wire into startup scripts / docker-compose
- [ ] Write `monitoring/README.md` with setup + runbook

---

*Last updated: 2026-05-05 — Shashi*
