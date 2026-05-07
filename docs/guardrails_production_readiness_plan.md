# Guardrails Application Production Readiness Plan

## Purpose

This document converts the existing Guardrails dashboard application into an end-to-end production deployment readiness plan.

The goal is not to make minor development-stage improvements. The goal is to make the application deployment-ready for an enterprise environment, with production-grade security, reliability, scalability, auditability, observability, CI/CD, infrastructure, compliance, and operational controls.

The current app is a strong research/demo platform. The production target is an enterprise AI guardrails control plane that can safely run in a controlled production environment.

---

## Current Application Summary

The current application consists of:

- React + Vite frontend dashboard.
- FastAPI app backend.
- Gateway service for input guardrails.
- Local Ollama-compatible LLM runtime.
- Multi-Agent Debate (MAD) output guardrail.
- Confidence Scoring Engine (CSE) concept layered after MAD.
- SQLite persistence for sessions and feedback.
- Gateway proxy routes through the backend.
- Analytics, feedback, settings, session trace, evaluation, and gateway pages.
- Airflow, RAG, synthetic data, fine-tuning, RLHF, and research-related folders in the same monorepo.

Current request flow:

```text
Browser UI
  -> FastAPI app backend
  -> Gateway input validation
  -> LLM runtime if not blocked
  -> SQLite session insert
  -> MAD verification in background
  -> MAD/CSE result persisted
  -> Dashboard displays session trace, analytics, and feedback
```

Production target flow:

```text
Enterprise User
  -> SSO/OIDC authentication
  -> Frontend served via CDN or hardened web server
  -> API gateway / ingress with TLS, WAF, rate limiting
  -> App backend with RBAC and tenant controls
  -> Gateway input guardrail service
  -> Approved LLM runtime or provider abstraction
  -> MAD verification worker/service
  -> Confidence Engine final routing
  -> Durable database and audit store
  -> Observability, SIEM, alerting, metrics, traces
  -> Human review workflow when required
```

---

## Production Readiness Principles

The production application must be:

1. Secure by default.
2. Auditable and explainable.
3. Reliable under failure.
4. Horizontally scalable where needed.
5. Deterministic in routing decisions.
6. Observable across every pipeline stage.
7. Recoverable after infrastructure or service failures.
8. Configurable without code changes.
9. Testable through automated and manual validation.
10. Ready for enterprise governance, privacy, and compliance review.

---

## Major Production Gaps To Address

The current application should not be deployed to production as-is because it has several demo/local assumptions:

- SQLite is used for application persistence.
- Local HTTP is assumed.
- App-level authentication and authorization are not fully implemented.
- Background MAD execution uses local async behavior instead of a durable job queue.
- CORS is configured for local development origins.
- Ollama is treated as a local runtime.
- Gateway and Airflow can conflict on port 8080 in local compose setups.
- Observability appears partially planned but must be verified end-to-end.
- Full-stack E2E tests, load tests, security tests, and DR tests need to be productionized.
- Deployment isolation across frontend, backend, gateway, MAD, workers, databases, and batch systems needs to be formalized.

---

## Target Production Architecture

### 1. Frontend

Production frontend requirements:

- Build React/Vite assets using a production build pipeline.
- Serve static assets through CDN, Nginx, cloud storage static hosting, or an enterprise-approved web tier.
- Disable development-only mock mode in production.
- Use environment-specific API base URLs.
- Enforce secure headers:
  - Content-Security-Policy
  - X-Content-Type-Options
  - X-Frame-Options or frame-ancestors in CSP
  - Referrer-Policy
  - Strict-Transport-Security at the TLS layer
- Do not expose internal service URLs to the browser.
- All browser API calls must go through the authenticated app backend or API gateway.

### 2. API Gateway / Ingress

Production ingress requirements:

- TLS termination.
- WAF or enterprise gateway policy.
- Request size limits.
- Rate limiting.
- IP allowlisting where required.
- Authentication enforcement.
- Request correlation ID injection.
- Centralized access logs.
- Routing to frontend and backend services.

Example external routes:

```text
https://guardrails.company.com/                 -> Frontend
https://guardrails.company.com/api/*            -> App backend
https://guardrails.company.com/healthz          -> Public shallow health if approved
```

Internal-only services should not be public:

```text
Gateway service
MAD service or worker
LLM runtime
Database
Queue
Vector store
Observability collectors
```

### 3. App Backend

Production backend requirements:

- FastAPI app served by a production process manager.
- Use `gunicorn` with `uvicorn.workers.UvicornWorker` or an equivalent approved ASGI deployment pattern.
- No development reload mode.
- Structured JSON logging.
- Request ID middleware.
- Authentication middleware.
- RBAC authorization dependencies.
- Centralized error response model.
- Strict CORS by environment.
- Timeouts and retries for outbound service calls.
- Idempotency key support for `POST /api/query`.
- No local file assumptions for runtime state.
- Database connection pooling.
- Migration-managed schema.

### 4. Gateway Service

Production gateway requirements:

- Deployed as a separate internal service.
- No port collision with Airflow or other services.
- Authenticated service-to-service access.
- Structured gateway decisions.
- Policy versioning.
- Threshold versioning.
- Audit logs emitted to central logging/SIEM.
- High availability deployment with multiple replicas if traffic requires it.
- Health, readiness, and liveness endpoints.

### 5. LLM Runtime

Production LLM requirements:

- Abstract model provider behind a stable interface.
- Do not hardcode Ollama as the only runtime.
- Support configuration for:
  - local Ollama
  - enterprise-hosted model endpoint
  - cloud LLM provider
  - internal OpenAI-compatible endpoint
- Apply timeouts, retries, and circuit breakers.
- Track model name, model version, prompt version, latency, token usage if available, and failure reason.
- Ensure prompts and outputs are governed according to enterprise data policy.
- Prevent direct public access to the model runtime.

### 6. MAD Verification

Production MAD requirements:

- Move from local background `asyncio.create_task` to a durable execution model.
- Use a queue-backed worker pattern for production:
  - Celery + Redis/RabbitMQ
  - RQ + Redis
  - Arq
  - cloud queue + worker
  - Kubernetes job pattern if appropriate
- Persist job status and retry count.
- Support retry with backoff.
- Support dead-letter queue or failed job table.
- Make MAD timeout configurable.
- Store MAD version, corpus version, retrieval configuration, and judge configuration.
- MAD should update session state reliably even if the app backend restarts.

### 7. Confidence Scoring Engine

Production CSE requirements:

- Make CSE a first-class final decision layer after MAD.
- Store final score, routing decision, score breakdown, fallback mode, flags, failed claims, and explanation.
- Version the CSE formula and threshold configuration.
- Ensure HARD_BLOCK overrides aggregate score.
- Do not silently treat missing evaluator signals as zero.
- Record whether scoring used full mode or fallback mode.
- Make CSE deterministic and test-covered.
- Expose CSE details in the session trace UI.

### 8. Database

SQLite must not be used for enterprise production.

Production database requirements:

- Replace SQLite with PostgreSQL or an enterprise-approved relational database.
- Use SQLAlchemy or another consistent data access layer if not already present.
- Use Alembic or an equivalent migration tool.
- Enable connection pooling.
- Define explicit indexes for:
  - session ID
  - request ID
  - tenant ID
  - created timestamp
  - status
  - gateway decision
  - MAD route
  - CSE route
  - feedback status
- Support backups and point-in-time recovery.
- Encrypt data at rest.
- Define data retention rules.
- Define purge/anonymization jobs.

Recommended production tables:

```text
users
roles
user_roles
tenants
sessions
session_events
gateway_decisions
llm_invocations
mad_results
cse_results
feedback
audit_logs
configuration_versions
policy_versions
review_actions
job_runs
```

Do not over-normalize in the first release, but do separate critical audit/event data from mutable session summaries.

### 9. Queue and Worker Layer

Production queue requirements:

- All long-running verification should be queued.
- `POST /api/query` should create a session and enqueue follow-up work.
- Worker should process LLM, MAD, and CSE tasks depending on desired architecture.
- Workers should be independently scalable.
- Job status should be persisted.
- Failed jobs should be visible in the UI or admin panel.
- Jobs must be idempotent and safe to retry.

Recommended session workflow:

```text
POST /api/query
  -> create session with RECEIVED
  -> run Gateway synchronously or enqueue depending on latency target
  -> if blocked: finalize as GATEWAY_BLOCKED / HARD_BLOCKED
  -> if passed: enqueue LLM task
  -> LLM worker writes result
  -> enqueue MAD task
  -> MAD worker writes result
  -> enqueue or run CSE
  -> CSE writes final route
  -> session moves to COMPLETED, HUMAN_REVIEW_REQUIRED, or HARD_BLOCKED
```

For a simpler production first release, Gateway and LLM can remain synchronous while MAD/CSE become queue-backed. However, the architecture should allow all stages to become async later.

---

## Required Production Features

## 1. Authentication and Authorization

Implement enterprise authentication.

Minimum production requirement:

- OIDC or SAML integration through an enterprise identity provider.
- JWT validation or session-based auth behind an enterprise proxy.
- Backend enforcement, not frontend-only enforcement.
- User identity attached to every session and audit event.

Recommended roles:

```text
admin
reviewer
analyst
viewer
auditor
service_account
```

Role capabilities:

```text
admin:
  - manage system configuration
  - manage guardrail thresholds
  - manage users/roles if local RBAC exists
  - export data
  - view all audit logs
  - override review decisions if policy allows

reviewer:
  - review HUMAN_REVIEW sessions
  - approve, reject, or escalate outputs
  - add review notes
  - update feedback status

analyst:
  - view sessions
  - view analytics
  - export limited reports
  - inspect quality metrics

viewer:
  - view permitted sessions and traces
  - no config changes
  - no exports unless explicitly allowed

auditor:
  - read-only access to audit logs
  - read-only access to session decisions
  - export audit reports if approved

service_account:
  - service-to-service calls only
  - scoped permissions
```

Authorization must be enforced on:

- gateway config routes
- settings routes
- feedback resolution routes
- export routes
- audit log routes
- policy/config version changes
- human review actions

## 2. Multi-Tenancy and Data Isolation

If this app will serve more than one business unit, client, or environment, add tenant isolation.

Production requirements:

- Add `tenant_id` to sessions, feedback, audit logs, users, and configuration.
- Enforce tenant scoping in every query.
- Do not rely on frontend filters for tenant isolation.
- Support tenant-specific policies and thresholds if needed.
- Ensure exports are tenant-scoped.
- Include tenant ID in audit logs.

## 3. Session Lifecycle State Machine

Production sessions must have explicit lifecycle states.

Recommended states:

```text
RECEIVED
GATEWAY_RUNNING
GATEWAY_PASSED
GATEWAY_ESCALATED
GATEWAY_BLOCKED
LLM_RUNNING
LLM_COMPLETED
MAD_QUEUED
MAD_RUNNING
MAD_COMPLETED
CSE_RUNNING
CSE_COMPLETED
RETRY_QUEUED
HUMAN_REVIEW_REQUIRED
HUMAN_REVIEW_APPROVED
HUMAN_REVIEW_REJECTED
HARD_BLOCKED
COMPLETED
FAILED
CANCELLED
```

Each transition should create a `session_events` row.

Each event should include:

```text
session_id
request_id
tenant_id
stage
from_status
to_status
timestamp
actor_type: system | user | worker | service
actor_id
message
metadata_json
error_code
error_message
```

## 4. Human Review Workflow

Production guardrails systems need a real review path.

Add a Human Review page or workflow section for sessions routed to `HUMAN_REVIEW_REQUIRED`.

Reviewer actions:

```text
approve_delivery
reject_answer
request_regeneration
escalate_to_admin
mark_false_positive
mark_false_negative
add_review_note
```

Each action must be audit logged.

Review record fields:

```text
review_id
session_id
reviewer_id
review_status
review_decision
review_notes
created_at
updated_at
completed_at
```

## 5. Audit Logging

Every important action must be auditable.

Audit events required:

- user login/logout if available
- query submitted
- gateway decision produced
- LLM called
- LLM failed
- MAD started/completed/failed
- CSE decision produced
- routing decision selected
- human review action
- feedback created/updated/resolved
- config changed
- policy changed
- export created
- admin action performed

Audit log fields:

```text
audit_id
tenant_id
request_id
session_id
actor_id
actor_role
actor_type
action
resource_type
resource_id
before_json
after_json
metadata_json
ip_address
user_agent
timestamp
```

Audit logs should be append-only from the application perspective.

For regulated environments, forward audit logs to SIEM or immutable log storage.

## 6. Configuration and Policy Versioning

No production scoring, thresholds, service URLs, or model names should be hardcoded.

Production configuration should support:

```text
APP_ENV
APP_VERSION
DEPLOYMENT_REGION
TENANT_MODE
DATABASE_URL
REDIS_URL or QUEUE_URL
GATEWAY_URL
OLLAMA_BASE_URL or LLM_PROVIDER_URL
LLM_PROVIDER_TYPE
DEFAULT_LLM_MODEL
MAD_MODE
MAD_API_URL
MAD_TIMEOUT_SECONDS
CSE_CONFIG_VERSION
CSE_DELIVER_THRESHOLD
CSE_HUMAN_REVIEW_THRESHOLD
CSE_MAX_RETRY_COUNT
CORS_ALLOWED_ORIGINS
OIDC_ISSUER_URL
OIDC_CLIENT_ID
OIDC_AUDIENCE
ENABLE_AUDIT_LOGGING
ENABLE_LANGFUSE
LANGFUSE_HOST
OTEL_EXPORTER_OTLP_ENDPOINT
LOG_LEVEL
DATA_RETENTION_DAYS
```

Add configuration versioning for:

- gateway policy
- prompt templates
- LLM model configuration
- MAD judge configuration
- RAG corpus version
- CSE scoring formula
- CSE threshold configuration

Every session should record the versions used to produce its decision.

## 7. Secrets Management

Production secrets must not be stored in `.env` files inside deployed containers.

Use one of:

- Kubernetes Secrets with external secret manager integration
- HashiCorp Vault
- AWS Secrets Manager
- Azure Key Vault
- Google Secret Manager
- enterprise-approved secret store

Secrets include:

- database credentials
- OIDC client secrets
- LLM provider keys
- Langfuse/API keys
- gateway service credentials
- encryption keys
- object storage credentials

Production requirements:

- no secrets in source control
- no secrets in frontend bundles
- no secrets in logs
- rotation procedure defined
- least-privilege service accounts

## 8. Observability

Production observability must cover logs, metrics, traces, and alerts.

### Logging

Use structured JSON logs with:

```text
timestamp
level
service
version
environment
tenant_id
request_id
session_id
trace_id
span_id
stage
message
error_code
error_message
latency_ms
```

### Metrics

Expose Prometheus/OpenTelemetry-compatible metrics.

Required metrics:

```text
http_request_count
http_request_latency_ms
http_error_count
gateway_decision_count
llm_invocation_count
llm_latency_ms
llm_error_count
mad_job_count
mad_latency_ms
mad_error_count
cse_decision_count
cse_score_distribution
routing_decision_count
human_review_queue_depth
feedback_count
queue_depth
worker_failure_count
database_latency_ms
```

### Tracing

Use OpenTelemetry or an approved tracing stack.

Trace each request through:

```text
frontend request
backend API
Gateway call
LLM call
MAD job
CSE scoring
DB writes
feedback/review action
```

### Alerting

Create alerts for:

- API 5xx error rate
- gateway unavailable
- LLM unavailable
- MAD worker failures
- queue backlog too high
- database connection failures
- high latency
- high HARD_BLOCK anomaly rate
- high HUMAN_REVIEW queue depth
- CSE fallback spike
- failed exports
- authentication failures spike

## 9. Analytics and Reporting

Production analytics must be useful for operations, risk, and quality.

Operational metrics:

```text
total sessions
sessions by status
average end-to-end latency
average gateway latency
average LLM latency
average MAD latency
average CSE latency
error rate by stage
queue backlog
worker success/failure rate
```

Risk metrics:

```text
prompt injection count
jailbreak count
PII detection count
gateway block rate
gateway escalate rate
hard block rate
human review rate
policy violation categories
top blocked categories
```

Quality metrics:

```text
average CSE score
CSE fallback rate
average faithfulness score
average hallucination risk
unsupported claim count
judge disagreement count
retrieval/context relevancy score
retry success rate
```

Review metrics:

```text
human review queue size
average review time
review approval rate
review rejection rate
false positive rate
false negative rate
feedback by category
feedback by severity
```

## 10. Error Handling and Resilience

Define a standard error shape:

```json
{
  "error": true,
  "request_id": "...",
  "session_id": "...",
  "stage": "gateway | llm | mad | cse | database | auth | unknown",
  "code": "...",
  "message": "...",
  "retryable": true,
  "details": {}
}
```

Production requirements:

- Timeouts on every external call.
- Retries with exponential backoff for retryable failures.
- Circuit breaker for unavailable downstream services.
- Failed sessions persisted with error stage and reason.
- No silent background task failures.
- No unbounded retry loops.
- Dead-letter handling for failed jobs.
- Clear UI state for partial completion.

## 11. Data Protection and Privacy

Production data controls:

- Classify stored data fields by sensitivity.
- Redact PII in logs.
- Encrypt data at rest.
- Encrypt data in transit.
- Support retention policies.
- Support deletion/anonymization where required.
- Avoid storing full prompts/answers in logs.
- Use separate storage controls for raw text, embeddings, audit logs, and exports.
- Add export controls and audit every export.

Suggested retention tiers:

```text
session metadata: 365 days or per policy
raw prompt/answer: shorter retention, configurable
feedback/review records: 365 days or per policy
audit logs: per compliance requirement, often longer
metrics: 90-180 days depending on platform
```

## 12. Deployment Isolation

Separate environments:

```text
dev
qa
demo
staging
production
```

Production must not share:

- databases
- queues
- secrets
- model endpoints
- object storage buckets
- observability projects
- identity clients

Use environment-specific config and deployment manifests.

## 13. Containerization

Create production Dockerfiles for:

```text
frontend
app-backend
gateway
mad-worker or mad-api
optional: evaluation worker
```

Production Dockerfile requirements:

- multi-stage builds
- non-root user
- pinned dependencies
- minimal base images
- healthcheck where appropriate
- no dev dependencies in runtime image
- no source secrets copied into image
- reproducible build

Frontend container:

- build static assets
- serve with Nginx or approved static server
- hardened security headers

Backend container:

- install Python dependencies from lockfile
- run production ASGI server
- expose only internal service port
- no reload mode

## 14. Kubernetes or Cloud Deployment

If Kubernetes is used, define:

```text
Deployments
Services
Ingress
ConfigMaps
Secrets or ExternalSecrets
HorizontalPodAutoscalers
PodDisruptionBudgets
NetworkPolicies
ServiceAccounts
RoleBindings
PersistentVolumeClaims only if required
CronJobs for retention/cleanup
Jobs for migrations
```

Required health probes:

```text
livenessProbe
readinessProbe
startupProbe where needed
```

Recommended service layout:

```text
guardrails-frontend
guardrails-backend
guardrails-gateway
guardrails-mad-worker
guardrails-queue
guardrails-postgres
guardrails-observability-collector
```

In managed cloud environments, prefer managed Postgres, managed Redis/queue, managed secret store, and managed logging where available.

## 15. CI/CD

Production CI/CD requirements:

Stages:

```text
lint
format check
type check
unit tests
integration tests
frontend build
backend build
container build
dependency scan
SAST
secret scan
SBOM generation
container vulnerability scan
migration validation
staging deploy
smoke tests
approval gate
production deploy
post-deploy verification
```

Required checks:

- Python tests pass.
- Frontend build passes.
- API schema validation passes.
- Database migrations apply and rollback in staging.
- Container images scan below approved severity threshold.
- No secrets detected.
- E2E smoke test passes.

Deployment strategy:

- Blue/green or rolling deployment.
- Canary if traffic volume justifies it.
- Fast rollback plan.
- Migration rollback plan.

## 16. Testing Strategy

Production readiness requires multiple levels of testing.

### Backend Unit Tests

Test:

- Gateway decision handling.
- LLM failure handling.
- MAD job status transitions.
- CSE scoring and routing.
- CSE fallback behavior.
- RBAC enforcement.
- Config loading.
- Audit log creation.
- Analytics aggregation.
- Error response model.

### Frontend Tests

Test:

- session list rendering
- status badges
- session trace with pending MAD
- CSE panel
- audit timeline
- analytics dashboard
- feedback workflow
- unauthorized page/action behavior
- system health page

### Integration Tests

Test:

- app backend to gateway
- app backend to LLM provider mock
- app backend to queue
- worker to database
- worker to MAD
- CSE persistence
- feedback and review lifecycle

### E2E Tests

Scenarios:

1. Safe query -> DELIVER.
2. Prompt injection query -> BLOCK.
3. Weak answer -> RETRY or HUMAN_REVIEW.
4. False material claim -> HARD_BLOCK.
5. MAD worker failure -> FAILED or HUMAN_REVIEW depending on policy.
6. CSE fallback -> visible fallback mode.
7. Reviewer approves HUMAN_REVIEW session.
8. Feedback is submitted and resolved.
9. Analytics updates.
10. Export is audited.

### Load Tests

Test:

- concurrent query submissions
- gateway throughput
- LLM timeout behavior
- MAD worker queue behavior
- database write throughput
- dashboard read latency

### Security Tests

Run:

- SAST
- dependency scanning
- container scanning
- secret scanning
- DAST against staging
- auth bypass testing
- RBAC negative tests
- CORS validation
- rate limit testing
- injection testing for prompt, SQL, logs, and JSON payloads

## 17. SLOs and Production Targets

Define initial SLOs before production launch.

Example SLOs:

```text
API availability: 99.5% or higher for first production release
Dashboard page load: p95 < 3 seconds
Gateway validation latency: p95 < 1 second
LLM call latency: model-dependent, tracked separately
MAD verification latency: p95 target depends on worker design
CSE scoring latency: p95 < 500 ms
Error rate: < 1% for non-user-caused server errors
Human review queue freshness: critical items reviewed within defined SLA
```

Define stage-specific latency budgets:

```text
Gateway: 300-1000 ms target
LLM: model-dependent
MAD: async, SLA in minutes if needed
CSE: sub-second
Session list: p95 < 1 second
Analytics: p95 < 2 seconds or cached
```

## 18. Disaster Recovery and Backup

Production requirements:

- Automated database backups.
- Point-in-time recovery if using Postgres.
- Backup restoration test.
- Queue recovery behavior documented.
- Audit log retention independent of app database where required.
- Config backup/versioning.
- Runbooks for restoring services.

Define:

```text
RPO: Recovery Point Objective
RTO: Recovery Time Objective
```

Example starting point:

```text
RPO: 15 minutes
RTO: 4 hours
```

Adjust based on business criticality.

## 19. Runbooks

Create production runbooks for:

- Gateway unavailable.
- LLM provider unavailable.
- MAD worker backlog high.
- Database unavailable.
- Queue unavailable.
- CSE fallback spike.
- High HARD_BLOCK anomaly rate.
- Login/auth provider outage.
- Failed deployment rollback.
- Database migration failure.
- Suspicious data export.
- High prompt-injection attack volume.
- Human review queue overload.

Each runbook should include:

```text
symptoms
alerts
impact
first checks
mitigation steps
rollback steps
owner/escalation path
post-incident actions
```

---

# End-to-End Production Implementation Plan

## Phase 0: Production Assessment and Baseline

Objective: Identify what must change before production.

Tasks:

1. Inspect current repository and identify all runtime entrypoints.
2. Map all services, ports, dependencies, and data stores.
3. Confirm current startup flow.
4. Confirm CSE implementation state.
5. Confirm MAD execution mode.
6. Confirm Gateway config and logging behavior.
7. Confirm frontend API usage.
8. Identify all hardcoded localhost URLs.
9. Identify all `.env` values and secrets.
10. Identify missing tests.
11. Identify missing deployment artifacts.
12. Produce current-state architecture diagram.
13. Produce target-state architecture diagram.

Deliverables:

- Production gap report.
- Service dependency map.
- Data flow and trust boundary diagram.
- List of required code changes.
- List of required infrastructure changes.

## Phase 1: Configuration, Environment, and Runtime Hardening

Objective: Remove local/demo assumptions.

Tasks:

1. Add backend settings module.
2. Move service URLs into environment-driven configuration.
3. Add strict environment validation on startup.
4. Add production-safe CORS configuration.
5. Add app version and build metadata.
6. Add `/healthz`, `/readyz`, and `/livez` endpoints.
7. Add standard error response model.
8. Add request ID middleware.
9. Add structured JSON logging.
10. Remove production reliance on `start.sh`.

Deliverables:

- Config layer.
- Production health endpoints.
- Structured logs.
- Updated environment variable documentation.

## Phase 2: Authentication, RBAC, and Tenant Controls

Objective: Protect the application.

Tasks:

1. Add OIDC/JWT validation or enterprise proxy auth integration.
2. Add user identity extraction.
3. Add RBAC model.
4. Enforce RBAC on backend routes.
5. Add frontend role-aware navigation and disabled actions.
6. Add tenant ID propagation if multi-tenancy is required.
7. Add auth audit events.

Deliverables:

- Auth middleware/dependencies.
- RBAC policy map.
- Protected routes.
- Audit events for auth-sensitive actions.

## Phase 3: Production Persistence

Objective: Replace SQLite with production-grade persistence.

Tasks:

1. Add database abstraction layer if not already present.
2. Introduce PostgreSQL support.
3. Add SQLAlchemy models or consistent repository layer.
4. Add Alembic migrations.
5. Create production schema for sessions, events, feedback, MAD, CSE, and audit logs.
6. Add indexes.
7. Add connection pooling.
8. Add migration commands.
9. Add backup and retention documentation.
10. Keep SQLite only for local development if useful.

Deliverables:

- PostgreSQL-ready schema.
- Migration scripts.
- Updated DB access layer.
- Backward-compatible local dev mode.

## Phase 4: Durable Pipeline Execution

Objective: Make the guardrails pipeline reliable.

Tasks:

1. Add queue-backed job model.
2. Move MAD background execution from local `asyncio.create_task` to worker processing.
3. Add job status tracking.
4. Add retry and dead-letter behavior.
5. Add idempotency for query submissions and worker jobs.
6. Add session lifecycle state machine.
7. Add session event table and audit timeline.
8. Add stage-level timestamps.
9. Add worker health metrics.

Deliverables:

- Queue integration.
- Worker process.
- Durable MAD/CSE execution.
- Session lifecycle state machine.
- UI-visible pipeline status.

## Phase 5: Confidence Engine Productionization

Objective: Make CSE a first-class final routing engine.

Tasks:

1. Normalize CSE output schema.
2. Add CSE config version.
3. Add score breakdown.
4. Add fallback modes.
5. Add triggered flags.
6. Add failed claims list.
7. Add final routing explanation.
8. Fix `/api/sessions/{id}/cse` if it reads the wrong field.
9. Add CSE persistence table or JSON column.
10. Add CSE panel in UI.
11. Add CSE tests.

Deliverables:

- CSE API.
- CSE UI panel.
- CSE persistence.
- CSE versioning.
- CSE unit and integration tests.

## Phase 6: Enterprise Dashboard Features

Objective: Make the UI useful for production operations and review.

Tasks:

1. Improve session list with status, risk, score, route, latency, and feedback indicators.
2. Improve session trace with gateway, LLM, MAD, CSE, errors, and audit timeline.
3. Add human review queue page.
4. Improve feedback page with status, severity, category, reviewer notes.
5. Improve analytics page with operational, risk, quality, and review metrics.
6. Add system health page.
7. Add admin/config view for authorized users.
8. Add empty/loading/error states.
9. Add accessible and consistent badges.
10. Ensure no sensitive internal config is leaked in the browser.

Deliverables:

- Production-grade dashboard UX.
- Human review workflow.
- System health UI.
- Analytics improvements.

## Phase 7: Observability and Audit Integration

Objective: Make production behavior measurable and reviewable.

Tasks:

1. Add OpenTelemetry instrumentation.
2. Add trace IDs across backend, gateway, worker, MAD, CSE, and database calls.
3. Add metrics endpoint.
4. Add dashboard metrics.
5. Add alert rules.
6. Add audit log table.
7. Forward logs to centralized logging.
8. Add SIEM forwarding if required.
9. Add audit export for auditors.

Deliverables:

- Logs, metrics, traces.
- Alert definitions.
- Audit trail.
- Observability documentation.

## Phase 8: Security and Compliance Hardening

Objective: Pass enterprise security review.

Tasks:

1. Add security headers.
2. Harden CORS.
3. Add rate limiting.
4. Add request size limits.
5. Add input validation with Pydantic models.
6. Add output encoding and sanitization where needed.
7. Add export controls.
8. Add PII redaction in logs.
9. Add data retention and deletion jobs.
10. Add threat model.
11. Add security test suite.
12. Add SAST, dependency scan, secret scan, and container scan to CI.

Deliverables:

- Security controls.
- Threat model.
- Security test results.
- Compliance-ready documentation.

## Phase 9: Deployment Artifacts and CI/CD

Objective: Make deployment repeatable.

Tasks:

1. Add production Dockerfiles.
2. Add docker-compose only for local/staging simulation if useful.
3. Add Kubernetes manifests or Helm chart.
4. Add Terraform/IaC if cloud infrastructure is in scope.
5. Add GitHub Actions/GitLab CI/Azure DevOps pipeline.
6. Add migration job.
7. Add smoke tests.
8. Add deployment gates.
9. Add rollback procedure.
10. Add release checklist.

Deliverables:

- Container images.
- Deployment manifests.
- CI/CD pipeline.
- Release and rollback documentation.

## Phase 10: Production Validation and Go-Live

Objective: Prove the app is deployment-ready.

Tasks:

1. Run full unit test suite.
2. Run integration tests.
3. Run E2E tests.
4. Run load tests.
5. Run security scans.
6. Run migration test.
7. Run backup/restore test.
8. Run DR tabletop exercise.
9. Run staging smoke test.
10. Complete production readiness review.
11. Complete security review.
12. Complete operations handoff.

Deliverables:

- Production readiness signoff.
- Security signoff.
- Runbooks.
- SLOs and dashboards.
- Release notes.
- Go-live checklist.

---

# Production Dashboard Requirements

## Navigation

Production dashboard navigation should include:

```text
Conversations / Sessions
New Query / Playground
Human Review
Analytics
Gateway
Evaluation
Feedback
System Health
Audit Logs
Settings / Admin
```

Access should depend on role.

## Session List

Each row should show:

```text
session ID
created time
user or service account
tenant
status
gateway decision
LLM model
MAD route
CSE score
final route
latency
feedback indicator
review status
```

## Session Trace

Each session trace should show:

1. Request metadata.
2. User query, redacted if needed.
3. Gateway result.
4. Gateway detected spans/entities.
5. LLM response.
6. MAD debate trace.
7. CSE breakdown.
8. Final routing decision.
9. Audit timeline.
10. Error details if any.
11. Review actions if authorized.
12. Feedback actions.

## CSE Panel

The CSE panel should show:

```text
final_cse_score
routing_decision
scoring_mode
cse_version
score_breakdown
triggered_flags
explanation
retry_reasons
review_reasons
block_reasons
top_failed_claims
metadata
```

## Human Review Page

The Human Review page should support:

- filter by severity
- filter by queue age
- filter by policy category
- assign reviewer
- approve/reject/escalate
- add notes
- show full decision trace
- audit every action

## Analytics Page

The Analytics page should show:

- operational health
- guardrail risk metrics
- CSE quality metrics
- human review metrics
- feedback metrics
- latency metrics
- failure metrics

## System Health Page

The System Health page should show:

```text
backend
database
queue
gateway
LLM runtime
MAD worker
CSE configuration
observability
identity provider
object storage, if used
```

Each component should show:

```text
status
latency
last checked time
error message if unhealthy
version if available
```

---

# Production Acceptance Criteria

The application is production deployment-ready only when all of the following are true:

## Architecture

- Frontend, backend, gateway, workers, database, and queue are independently deployable.
- Internal services are not publicly exposed.
- Localhost assumptions are removed from production config.
- `start.sh` is not required for production.
- SQLite is not used in production.

## Security

- Authentication is enforced.
- RBAC is enforced in the backend.
- TLS is used.
- CORS is restricted.
- Secrets are managed through approved secret storage.
- Sensitive data is not logged.
- Exports are controlled and audited.
- Security scans pass required thresholds.

## Reliability

- Long-running work is queue-backed.
- Failed jobs are visible and recoverable.
- Retries are bounded.
- Health and readiness endpoints exist.
- Backups and restore tests are complete.
- Deployment rollback is documented.

## Audit and Compliance

- Every guardrail decision is reproducible.
- CSE decisions include formula version, thresholds, score breakdown, and explanation.
- Human review actions are audit logged.
- Config changes are audit logged.
- Audit logs are append-only from the app perspective.
- Retention policy is documented.

## Observability

- Logs are structured.
- Metrics are emitted.
- Traces connect pipeline stages.
- Alerts exist for critical failures.
- Dashboards exist for operational, risk, and quality metrics.

## Testing

- Unit tests pass.
- Integration tests pass.
- E2E tests pass.
- Load tests meet baseline targets.
- Security tests pass.
- Migration tests pass.
- Backup/restore test passes.

## UX

- Dashboard handles pending, failed, blocked, reviewed, and completed states.
- CSE is visible and understandable.
- Human review workflow is usable.
- System health is visible.
- Analytics reflect production guardrail performance.

---

# Copy-Ready Coding Agent Prompt

Use the prompt below with a coding agent.

```text
You are a senior full-stack engineer, platform engineer, and enterprise AI safety systems architect.

You are working on our Guardrails application. The app is currently a research/demo-oriented enterprise guardrails platform with:

- React + Vite frontend
- FastAPI app backend
- Gateway input guardrail service
- Ollama/OpenAI-compatible LLM runtime
- Multi-Agent Debate output guardrail
- Confidence Scoring Engine concept after MAD
- SQLite persistence
- session trace, analytics, gateway, evaluation, feedback, and settings pages

Your task is not to make small development-stage upgrades.
Your task is to make this application production deployment-ready end to end.

Do not rewrite the whole system unnecessarily.
Do not add complexity for its own sake.
Do not break the existing app flow.
Do not leave local/demo assumptions in the production path.

You must inspect the repository first, then implement a production-readiness plan with clear, targeted changes.

Primary goal:
Turn the application into a production-ready enterprise AI guardrails control plane with security, RBAC, production persistence, durable job execution, auditability, observability, CI/CD readiness, deployment artifacts, and operational runbooks.

==================================================
1. Inspect the current repository
==================================================

Before coding, inspect and summarize:

- app/backend/main.py
- app/backend/db.py
- app/backend/pipeline.py
- app/backend/routers/sessions.py
- app/backend/routers/gateway.py
- app/backend/routers/analytics.py
- app/backend/routers/feedback.py
- app/frontend/src/api/client.js
- app/frontend/src/App.jsx
- frontend pages and components for session trace, gateway, analytics, evaluation, feedback, settings
- gateway service entrypoints
- MAD pipeline entrypoints
- confidence/CSE related code
- current requirements files
- current Docker, compose, deployment, or startup files

Then produce a short implementation plan listing the minimal files that must change.

==================================================
2. Remove local/demo assumptions from production config
==================================================

Implement an environment-driven settings layer.

Support at minimum:

- APP_ENV
- APP_VERSION
- DATABASE_URL
- SQLITE_DB_PATH for local only
- GATEWAY_URL
- OLLAMA_BASE_URL or LLM_PROVIDER_URL
- LLM_PROVIDER_TYPE
- DEFAULT_LLM_MODEL
- MAD_MODE
- MAD_API_URL
- MAD_TIMEOUT_SECONDS
- QUEUE_URL or REDIS_URL if queue is added
- CSE_CONFIG_VERSION
- CSE thresholds
- CORS_ALLOWED_ORIGINS
- OIDC configuration placeholders or implementation
- ENABLE_AUDIT_LOGGING
- ENABLE_LANGFUSE or OTEL tracing
- LOG_LEVEL
- DATA_RETENTION_DAYS

Production must not hardcode localhost URLs.
Production CORS must be strict.
Production must not use mock frontend mode.

==================================================
3. Add production health/readiness endpoints
==================================================

Add:

- GET /healthz
- GET /livez
- GET /readyz
- GET /api/system/health

/api/system/health should check:

- backend
- database
- gateway
- LLM runtime
- MAD service or worker readiness
- CSE config loaded
- queue if used
- observability status
- auth provider if implemented

Return:

{
  "status": "healthy | degraded | unhealthy",
  "components": {...},
  "timestamp": "...",
  "version": "..."
}

Add frontend System Health page or upgrade Settings page to show this.

==================================================
4. Replace SQLite for production
==================================================

SQLite can remain only for local development.

Add production database support using PostgreSQL or a clean DATABASE_URL-driven abstraction.

Add migration support using Alembic or the repo's preferred migration pattern.

Schema must support:

- sessions
- session_events
- gateway_decisions
- llm_invocations
- mad_results
- cse_results
- feedback
- audit_logs
- human_reviews
- configuration_versions
- job_runs if queue is added

Add indexes for:

- session_id
- request_id
- tenant_id if multi-tenant
- status
- created_at
- gateway_decision
- mad_route
- cse_route
- feedback_status

If full DB migration is too large for one pass, create the abstraction and migration plan, but do not claim production readiness until SQLite is removed from production.

==================================================
5. Add session lifecycle state machine
==================================================

Add explicit statuses:

- RECEIVED
- GATEWAY_RUNNING
- GATEWAY_PASSED
- GATEWAY_ESCALATED
- GATEWAY_BLOCKED
- LLM_RUNNING
- LLM_COMPLETED
- MAD_QUEUED
- MAD_RUNNING
- MAD_COMPLETED
- CSE_RUNNING
- CSE_COMPLETED
- RETRY_QUEUED
- HUMAN_REVIEW_REQUIRED
- HUMAN_REVIEW_APPROVED
- HUMAN_REVIEW_REJECTED
- HARD_BLOCKED
- COMPLETED
- FAILED
- CANCELLED

Every status transition should create a session event.

Frontend must show status clearly in:

- session list
- session trace
- audit timeline

==================================================
6. Make MAD/CSE durable instead of local background-only
==================================================

Do not rely on local background asyncio tasks for production.

Add or prepare a queue-backed worker pattern for long-running tasks:

- MAD verification
- CSE scoring if not inline
- retries
- failed job recovery

Acceptable tools:

- Celery + Redis/RabbitMQ
- RQ + Redis
- Arq
- cloud queue + worker
- equivalent existing repo pattern

Workers must:

- update job status
- update session status
- write errors to DB
- support bounded retries
- support dead-letter or failed job tracking
- be idempotent

==================================================
7. Make CSE first-class and production-grade
==================================================

Fix or implement GET /api/sessions/{id}/cse.

Known issue to check:
The current app may deserialize mad_output_json into mad_output while the CSE endpoint still reads mad_output_json. Fix this if present.

CSE output must include:

- final_cse_score
- routing_decision
- scoring_mode
- cse_version
- score_breakdown
- triggered_flags
- explanation
- retry_reasons
- review_reasons
- block_reasons
- top_failed_claims
- metadata

CSE must be:

- deterministic
- versioned
- auditable
- test-covered
- visible in the frontend

HARD_BLOCK must override aggregate score.
Missing evaluator signals must not be silently treated as zero.
Fallback mode must be explicit.

==================================================
8. Add authentication, RBAC, and optional tenant controls
==================================================

Implement production-ready auth or a clean integration point for enterprise OIDC/SAML.

Backend must enforce roles.
Frontend role checks are not enough.

Roles:

- admin
- reviewer
- analyst
- viewer
- auditor
- service_account

Protect:

- config changes
- gateway config routes
- export routes
- audit logs
- human review actions
- feedback resolution
- admin/settings routes

If full OIDC is not feasible in this pass, implement the backend dependency layer and clearly mark mock/header auth as non-production. Do not claim production readiness with mock auth only.

==================================================
9. Add audit logging
==================================================

Create append-only audit logging for:

- query submitted
- gateway decision
- LLM invocation
- MAD result
- CSE result
- final route
- human review action
- feedback action
- config change
- export
- admin action

Audit log fields:

- audit_id
- tenant_id
- request_id
- session_id
- actor_id
- actor_role
- action
- resource_type
- resource_id
- before_json
- after_json
- metadata_json
- ip_address
- user_agent
- timestamp

Do not log secrets.
Redact sensitive values.

==================================================
10. Improve dashboard for production use
==================================================

Upgrade frontend pages.

Navigation should include:

- Sessions
- New Query
- Human Review
- Analytics
- Gateway
- Evaluation
- Feedback
- System Health
- Audit Logs
- Settings/Admin

Session list should show:

- status
- gateway decision
- MAD route
- CSE score
- final route
- user/tenant if available
- model
- latency
- feedback indicator
- review state

Session trace should show:

- query
- gateway result
- LLM answer
- MAD trace
- CSE breakdown
- final route
- audit timeline
- errors
- feedback/review actions

Add human review queue for HUMAN_REVIEW_REQUIRED sessions.

Add feedback fields:

- status
- category
- severity
- reviewer notes
- linked session

==================================================
11. Improve analytics
==================================================

Add production analytics:

Operational:

- total sessions
- sessions by status
- average latency by stage
- error rate by stage
- queue backlog
- worker failures

Risk:

- gateway block rate
- gateway escalate rate
- PII detections
- prompt injection detections
- jailbreak detections
- HARD_BLOCK count
- HUMAN_REVIEW count

Quality:

- average CSE score
- CSE fallback rate
- unsupported claim count
- hallucination risk average
- judge disagreement count
- retry success rate

Feedback/review:

- feedback count
- feedback by status/category/severity
- human review queue size
- average review time
- approval/rejection rate

==================================================
12. Add observability
==================================================

Implement structured JSON logs.

Add request_id, session_id, trace_id, tenant_id, stage, latency_ms, and error fields.

Add OpenTelemetry tracing or an equivalent approved tracing approach.

Instrument:

- backend request
- gateway call
- LLM call
- MAD job
- CSE scoring
- database writes
- review/feedback actions

Expose metrics for:

- request count/latency/errors
- gateway decisions
- LLM latency/errors
- MAD jobs/errors
- CSE scores/routes/fallbacks
- human review queue
- feedback
- database latency
- queue depth

Add alert definitions or documentation for alerts.

==================================================
13. Add security hardening
==================================================

Add or document:

- strict CORS
- security headers
- request size limits
- rate limiting
- input validation
- output sanitization where needed
- PII redaction in logs
- export controls
- secrets management
- dependency scanning
- container scanning
- secret scanning
- SAST
- DAST in staging if available

Production must not expose internal services directly.

==================================================
14. Add deployment artifacts
==================================================

Create production-ready deployment artifacts:

- frontend Dockerfile
- backend Dockerfile
- gateway Dockerfile if missing or not production-grade
- worker Dockerfile
- docker-compose only for local/staging simulation
- Kubernetes manifests or Helm chart if Kubernetes is target
- environment variable examples
- migration job
- health probes
- resource requests/limits
- network policies if Kubernetes

Containers must:

- use non-root user
- have pinned dependencies
- avoid dev dependencies
- avoid secrets
- avoid reload mode
- use minimal runtime images where practical

==================================================
15. Add CI/CD production pipeline
==================================================

Add pipeline stages:

- lint
- format check
- type check
- unit tests
- integration tests
- frontend build
- backend build
- container build
- dependency scan
- secret scan
- SAST
- SBOM generation
- container vulnerability scan
- migration validation
- staging deploy
- smoke tests
- approval gate
- production deploy
- post-deploy verification

Add rollback documentation.

==================================================
16. Add testing and validation
==================================================

Add tests for:

- health endpoints
- config loading
- auth/RBAC
- session lifecycle transitions
- gateway block path
- LLM failure path
- MAD queue path
- CSE endpoint
- CSE fallback
- HARD_BLOCK override
- analytics summary
- feedback workflow
- human review workflow
- audit log creation
- system health

Add E2E test scenarios:

1. safe query -> DELIVER
2. injection query -> BLOCK
3. unsupported output -> HUMAN_REVIEW
4. false material claim -> HARD_BLOCK
5. MAD failure -> visible failed state
6. CSE fallback -> visible fallback mode
7. human reviewer approves/rejects
8. feedback submitted/resolved
9. analytics updates
10. export is audited

Add load tests and security tests before production signoff.

==================================================
17. Add documentation and runbooks
==================================================

Update documentation with:

- target architecture
- request flow
- deployment flow
- environment variables
- database migrations
- auth/RBAC model
- session lifecycle states
- CSE scoring and routing
- audit logging
- observability
- CI/CD
- backup/restore
- disaster recovery
- runbooks
- known limitations

Runbooks required:

- gateway unavailable
- LLM unavailable
- MAD backlog high
- database unavailable
- queue unavailable
- CSE fallback spike
- auth outage
- failed deployment rollback
- migration failure
- suspicious export
- human review queue overload

==================================================
18. Production acceptance criteria
==================================================

The app is not production-ready until:

- Auth is enforced.
- RBAC is enforced in backend.
- SQLite is not used in production.
- Long-running MAD/CSE work is durable and recoverable.
- Internal services are not public.
- TLS/CORS/security headers are production-safe.
- Secrets are managed by approved secret store.
- CSE is versioned, auditable, and visible.
- Every decision is reproducible.
- Audit logs exist for critical actions.
- Logs, metrics, traces, and alerts are implemented.
- CI/CD includes tests and security scans.
- Deployment artifacts exist.
- Backups and restore are tested.
- Runbooks exist.
- E2E, load, and security tests pass in staging.

==================================================
Final output expected from you
==================================================

When finished, provide:

1. Files changed.
2. Backend changes.
3. Frontend changes.
4. Database/migration changes.
5. Queue/worker changes.
6. Auth/RBAC changes.
7. Observability changes.
8. Deployment artifacts added.
9. Tests added.
10. How to run locally.
11. How to deploy to staging/production.
12. How to verify production readiness.
13. Known limitations.
14. Remaining risks.
15. Recommended next steps.
```

---

# Go-Live Checklist

Use this before production deployment.

## Security

- [ ] OIDC/SAML authentication configured.
- [ ] RBAC enforced in backend.
- [ ] TLS enabled.
- [ ] CORS restricted to production domains.
- [ ] Internal services private.
- [ ] Secrets stored in approved secret manager.
- [ ] No secrets in source control or images.
- [ ] Security headers configured.
- [ ] Rate limits configured.
- [ ] Request size limits configured.
- [ ] Logs redact sensitive data.
- [ ] Export actions are audited.

## Infrastructure

- [ ] Frontend deployed through production static hosting or web tier.
- [ ] Backend deployed with production ASGI server.
- [ ] Gateway deployed as internal service.
- [ ] MAD/CSE worker deployed.
- [ ] PostgreSQL provisioned.
- [ ] Queue provisioned.
- [ ] Object/vector store provisioned if required.
- [ ] Network policies/firewall rules configured.
- [ ] Health probes configured.
- [ ] Resource limits configured.
- [ ] Autoscaling configured if required.

## Data

- [ ] SQLite not used in production.
- [ ] Migrations applied.
- [ ] Indexes created.
- [ ] Backups enabled.
- [ ] Restore tested.
- [ ] Retention policy implemented.
- [ ] Audit logs protected.

## Application

- [ ] Session lifecycle implemented.
- [ ] CSE endpoint working.
- [ ] CSE UI panel working.
- [ ] Human review workflow working.
- [ ] Feedback workflow working.
- [ ] Analytics working.
- [ ] System health working.
- [ ] Error states visible.
- [ ] Config versioning implemented.
- [ ] Policy versioning implemented.

## Observability

- [ ] JSON logs emitted.
- [ ] Metrics emitted.
- [ ] Traces emitted.
- [ ] Alerts configured.
- [ ] Dashboards created.
- [ ] SIEM forwarding configured if required.

## CI/CD

- [ ] Unit tests pass.
- [ ] Integration tests pass.
- [ ] E2E tests pass.
- [ ] Frontend build passes.
- [ ] Backend build passes.
- [ ] Container scans pass.
- [ ] Dependency scans pass.
- [ ] Secret scans pass.
- [ ] Migration validation passes.
- [ ] Staging smoke test passes.
- [ ] Rollback tested.

## Operations

- [ ] Runbooks complete.
- [ ] On-call ownership assigned.
- [ ] SLOs defined.
- [ ] DR plan documented.
- [ ] Incident process documented.
- [ ] Release notes created.
- [ ] Production approval received.

---

# Recommended Production Milestones

## Milestone 1: Production foundation

- Config/settings layer.
- Production health endpoints.
- Structured logging.
- Request IDs.
- Strict CORS.
- Production Dockerfiles.

## Milestone 2: Security foundation

- OIDC/JWT auth.
- RBAC.
- Secure headers.
- Secrets management.
- Audit log foundation.

## Milestone 3: Production persistence

- PostgreSQL.
- Migrations.
- Session event table.
- CSE/MAD persistence.
- Feedback/review schema.

## Milestone 4: Durable execution

- Queue.
- MAD worker.
- CSE worker or deterministic scoring module.
- Retry/dead-letter handling.
- Worker metrics.

## Milestone 5: Enterprise dashboard

- Session lifecycle UI.
- CSE panel.
- Human review queue.
- System health.
- Production analytics.
- Audit log view.

## Milestone 6: Observability and operations

- Metrics.
- Traces.
- Alerts.
- Dashboards.
- Runbooks.
- Backup/restore.

## Milestone 7: CI/CD and release readiness

- Automated tests.
- Security scans.
- Container builds.
- Staging deployment.
- Smoke tests.
- Rollback plan.
- Production readiness review.

---

# Final Recommendation

Do not position the current app as production-ready until the following minimum bar is met:

1. PostgreSQL replaces SQLite in production.
2. Authentication and backend RBAC are enforced.
3. MAD/CSE execution is durable and recoverable.
4. CSE is first-class, versioned, and auditable.
5. Every decision has an audit trail.
6. Internal services are isolated behind private networking.
7. Observability covers logs, metrics, traces, and alerts.
8. CI/CD includes security scanning and staging validation.
9. Backups, restore, and runbooks exist.
10. End-to-end production validation passes.

Once these are in place, the application can credibly move from a research/demo guardrails dashboard to a production-ready enterprise AI guardrails control plane.
