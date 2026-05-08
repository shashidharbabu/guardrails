# Guardrails Enterprise — Testing Plan

## Overview

The test suite covers five services and validates them at four levels:

| Level | Location | Needs services? | CI? |
|-------|----------|----------------|-----|
| Unit | `tests/unit/`, `rlhf/tests/`, `confidence/tests/`, `gateway/tests/` | No | Yes |
| Integration | `tests/integration/` | No (TestClient + mocks) | Yes |
| E2E (mocked) | `tests/e2e/` | No (full stack in-process) | Yes |
| Smoke (live) | `tests/smoke/` | Yes — `./start.sh all` | No (manual) |

Run all CI-safe tests:
```bash
pytest tests/ rlhf/tests/ confidence/tests/ gateway/tests/ -v -m "not smoke"
```

Run smoke tests against live services:
```bash
./start.sh all
pytest tests/smoke/ -v -m smoke
```

---

## Service 1 — Input Gateway (`:8080`)

**Purpose**: Blocks PII, jailbreak attempts, and prompt injection before queries reach the LLM.

### Unit tests — `gateway/tests/test_gateway.py`
| Test | What it verifies |
|------|----------------|
| `test_decision_pass` | Clean input → `PASS`, gateway_score < 0.3 |
| `test_decision_block_jailbreak` | jb_score ≥ 0.7 → immediate `BLOCK` |
| `test_decision_block_pi` | pi_score ≥ 0.7 → immediate `BLOCK` |
| `test_decision_block_pii` | pii_score ≥ 0.9 → immediate `BLOCK` |
| `test_escalate_moderate` | Composite 0.3–0.7 → `ESCALATE` |

### Unit tests — `tests/unit/test_gateway_decision.py`
Additional decision engine tests:
- Hard-block thresholds for all three validators
- Composite score always in `[0, 1]`
- `to_dict()` serialisation contract

### Integration smoke
```bash
curl -X POST http://localhost:8080/process \
  -H "Content-Type: application/json" \
  -d '{"query": "What does HIPAA require?"}'
# Expect: {"decision":"PASS","is_allowed":true,...}
```

**Pass criteria**: Clean healthcare query → `PASS`; jailbreak string → `BLOCK` within 2 s.

---

## Service 2 — MAD Pipeline (`:8001`)

**Purpose**: Two-agent debate (Agent A verifier, Agent B challenger) over extracted claims. Judge scores each claim with v_label ∈ {0, 0.5, 1}.

### Integration tests — `tests/integration/test_mad_api.py`
| Test | What it verifies |
|------|----------------|
| `test_health_ok` | `/mad/health` → `{"status":"ok"}` |
| `test_info_returns_version` | `/mad/info` returns version fields |
| `test_verify_missing_body_returns_422` | Missing body → validation error |
| `test_verify_response_shape` | Response has `routing`, `agg_confidence`, `claims` |
| `test_all_routing_outcomes_return_200` | DELIVER / HARD_BLOCK / RETRY all return 200 |

### Schema validation
- All MAD DB tables exist: `queries`, `claims`, `agent_outputs`, `agent_deltas`, `judge_verdicts`
- `agent_outputs` has at least one `agent_a round_num=1` row per claim after a run

### Integration smoke
```bash
curl -X POST http://localhost:8001/mad/verify \
  -H "Content-Type: application/json" \
  -d '{"query":"What does HIPAA require?","llm_answer":"HIPAA requires safeguards."}'
# Expect: routing in [DELIVER, RETRY, HUMAN_REVIEW, HARD_BLOCK]
```

**Pass criteria**: Responds within 60 s; routing is one of the four valid values; at least one claim extracted.

---

## Service 3 — App Backend (`:8000`)

**Purpose**: Orchestrates the full pipeline (gateway → LLM → MAD → CSE → DB), exposes REST API for the frontend.

### Integration tests — `tests/integration/test_backend_api.py`
| Endpoint | Test |
|----------|------|
| `GET /healthz` | Returns 200 + `{"status":"ok"}` |
| `GET /livez` | Returns 200 |
| `GET /readyz` | Returns 200 when DB is up |
| `GET /api/sessions` | Returns 200, list or dict |
| `POST /api/sessions` | Creates a session, returns session_id |
| `GET /api/analytics/summary` | Returns 200 |
| `GET /api/analytics/routing` | Returns routing distribution |
| `GET /api/audit` | Returns audit log entries |
| `GET /api/system/health` | Returns service status map |
| `GET /api/human-review/pending` | Returns pending review queue |
| `GET /api/rlhf/health` | Proxies to feedback loop or returns 502 |

### E2E tests — `tests/e2e/test_full_pipeline.py`
| Test | What it verifies |
|------|----------------|
| `test_clean_query_returns_response` | Full pipeline executes with mocked LLM |
| `test_blocked_query_returns_block_decision` | Gateway BLOCK propagated correctly |
| `test_session_history_grows_after_query` | Session record created |
| `test_analytics_after_queries` | Analytics counts updated |
| `test_audit_log_has_entries` | Audit trail written |

### Manual tests
```bash
# Create session
curl -X POST http://localhost:8000/api/sessions -H "Content-Type: application/json" \
  -d '{"title":"test"}'

# Submit query
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" \
  -d '{"session_id":"<id>","query":"What are HIPAA minimum necessary requirements?"}'

# Check routing decision in response
```

---

## Service 4 — RLHF Feedback Loop (`:8002`)

**Purpose**: Human review queue for ambiguous MAD outputs; GRPO advantage computation for Agent A fine-tuning.

### Unit tests — `rlhf/tests/`
| File | Tests |
|------|-------|
| `test_reward_fn.py` | Brier score formula, attack b_reward (+1/-1/0), heuristics |
| `test_storage.py` | Full scoring pass: 2 rollouts, gaslighting detection, GRPO advantages |
| `test_human_review.py` | Review queue: next item, submit good/bad/skip, 409 on double submit |

### Integration tests — `tests/integration/test_rlhf_api.py`
| Test | What it verifies |
|------|----------------|
| `test_health_ok` | `/health` → `{"status":"ok"}` |
| `test_review_next_pending` | Ambiguous item appears in queue |
| `test_submit_good_sets_final_reward_080` | `good` decision → `final_reward=0.80` |
| `test_submit_bad_sets_negative_reward` | `bad` decision → `final_reward=-0.60` |
| `test_queue_empty_after_review` | Queue drains after all items reviewed |
| `test_double_submit_returns_409` | Cannot review same item twice |
| `test_log_endpoint_returns_entries` | Audit log populated after review |
| `test_compute_rewards_returns_stats` | Batch scoring returns stat dict |

### GRPO correctness check
After a batch scoring run with 2+ rollouts on the same question:
- Both rows should have `grpo_advantage IS NOT NULL`
- Advantages should differ (mean subtraction applied)
- `sum(advantages) ≈ 0` across the group

```bash
# Trigger batch scoring
curl -X POST http://localhost:8002/rewards/compute
# Check advantages
sqlite3 multi_agent_debate/mad_store.db \
  "SELECT rollout_id, grpo_advantage FROM rewards LIMIT 10;"
```

---

## Service 5 — Frontend (`:5173`)

**Purpose**: React dashboard — 10 pages: Conversations, Session Trace, Analytics, Audit Logs, System Health, Human Review, Evaluation, Gateway, Feedback, New Query.

### Manual test checklist (pre-demo)

#### New Query (`/new-query`)
- [ ] Submit a clean healthcare question → response appears within 60 s
- [ ] MAD Debate panel expands showing claim list
- [ ] CSE score badge shows correct routing decision
- [ ] Copilot panel responds to follow-up questions

#### Conversations (`/conversations`)
- [ ] All past sessions listed
- [ ] Clicking a session navigates to Session Trace

#### Session Trace (`/sessions/:id`)
- [ ] Shows query, LLM answer, gateway spans, RAG spans, MAD debate, CSE score

#### Human Review (`/human-review`)
- [ ] Pending items appear (if ambiguous rewards exist)
- [ ] Approve / Reject buttons call `POST /api/rlhf/review/...`
- [ ] Queue count decrements after decision

#### Analytics (`/analytics`)
- [ ] Routing distribution chart renders
- [ ] Session count, average CSE score populated

#### Audit Logs (`/audit-logs`)
- [ ] Paginated log entries visible
- [ ] Filter by decision works

#### System Health (`/system-health`)
- [ ] All 4 services show green/red status
- [ ] Latency metrics rendered

#### Gateway (`/gateway`)
- [ ] Last N gateway scan results shown

#### Evaluation (`/evaluation`)
- [ ] CSE ablation results table renders

#### Settings (`/settings`)
- [ ] Can update thresholds (CSE deliver / human review)
- [ ] Save persists across page reload

### Component unit tests (Vitest / React Testing Library)
```bash
cd app/frontend && npm test
```
Key components to test:
- `MetricCard` — renders value and label
- `StatusBadge` — correct colour per routing decision
- `ScoreBar` — correct fill percentage
- `MADDebate` — claim list renders, expander works
- `CopilotPanel` — input/submit cycle

---

## Confidence Scoring Engine (CSE) — standalone

### Unit tests — `confidence/tests/test_cse.py`
| Test | What it verifies |
|------|----------------|
| Existing tests | Ablation, weight formula, routing thresholds |

### `tests/unit/test_cse.py`
| Test | What it verifies |
|------|----------------|
| `test_default_weights_sum_to_one` | Weight normalisation invariant |
| `test_deliver_threshold_above_human_review` | Config sanity |
| `test_hard_block_on_false_material_claim` | v_label=0.0 → HARD_BLOCK |
| `test_deliver_on_all_supported_claims` | All v_label=1.0 → DELIVER/RETRY |
| `test_result_score_in_range` | final_score always in [0, 1] |

---

## CI pipeline (`.github/workflows/ci.yml`)

```
on: push / pull_request to deploy / main

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - pip install -r requirements.txt -r requirements-backend.txt
      - pip install -e rlhf/
      - pytest tests/ rlhf/tests/ confidence/tests/ gateway/tests/ -v -m "not smoke"

  lint:
    - ruff check .
    - mypy app/backend/ confidence/ gateway/ rlhf/ --ignore-missing-imports
```

Smoke tests run manually before each demo — not in CI.

---

## Pre-demo checklist

```
[ ] ./start.sh all          — all 5 services start without errors
[ ] pytest -m "not smoke"   — all unit + integration tests green
[ ] pytest tests/smoke/     — all health checks pass
[ ] Submit test query       — full pipeline completes, routing shown in UI
[ ] Human review queue      — at least one item visible after pipeline run
[ ] GRPO advantages         — rewards table populated after /rewards/compute
[ ] Frontend pages          — all 10 pages load without JS errors
```
