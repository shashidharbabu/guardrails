# Enterprise & Production Readiness — RLHF Feedback Loop & Guardrails

This document proposes upgrades to move the current **MAD + SQLite + feedback scoring + Colab-oriented GRPO** stack toward **deployment-ready, enterprise-grade** operation. It is organized by concern so you can prioritize by risk, compliance, and customer expectations.

---

## 1. Product definition and scope boundaries

| Upgrade | Why it matters |
|--------|----------------|
| **Publish a “RLHF scope” page** for customers: what is trained (Agent A vs B), what data is retained, human review SLAs, and what is *not* claimed (e.g. no substitute for legal/clinical sign-off). | Prevents overselling; aligns sales and security review. |
| **Separate “shadow scoring” vs “training export”** environments. | Production traffic should not accidentally feed training DBs without governance. |
| **Versioned reward spec** (semver or date-stamped config). | Reproducibility and dispute resolution when rewards change. |

---

## 2. MLOps and training lifecycle

| Upgrade | Why it matters |
|--------|----------------|
| **Replace ad-hoc Colab as the only training path** with at least one of: CI-triggered GPU jobs (GitHub Actions + cloud GPU), Kubeflow/Vertex/SageMaker pipeline, or a pinned Docker image + scheduled runs. | Repeatability, auditability, no “works on my notebook.” |
| **Artifact store** for LoRA/checkpoints (S3/GCS + metadata: git SHA, dataset hash, reward spec version, hyperparameters). | Rollback, comparisons, regulatory traceability. |
| **Data lineage**: every `rewards` row or export row links to `query_id`, `rollout_id`, model versions (gateway, MAD, judge), and prompt hash. | Debugging and compliance evidence. |
| **Promotion gates**: no adapter promoted without passing eval suite + policy checks (see §8). | Stops reward hacking and regressions from reaching prod. |

---

## 3. Human-in-the-loop at enterprise scale

| Upgrade | Why it matters |
|--------|----------------|
| **Authentication and authorization** on the feedback API (OAuth2/OIDC, roles: reviewer, admin, auditor). | Current API-only MVP is not tenant-safe. |
| **Reviewer queues** with assignment, deduplication, and conflict resolution (two reviewers on high-risk items). | Quality and coverage for ambiguous band. |
| **SLA metrics**: time-in-queue, timeout behavior, % skipped vs decided, inter-reviewer agreement. | Operations and continuous improvement. |
| **Integration** with existing ITSM (Jira, ServiceNow) for escalations that exceed automation. | Enterprise workflow fit. |
| **PII minimization in review UI**: show redacted payloads by default; expand only for privileged roles with logging. | Reduces insider risk and scope of PHI processing. |

---

## 4. Security, secrets, and network

| Upgrade | Why it matters |
|--------|----------------|
| **TLS everywhere** (feedback API, MAD API, gateway) and **mTLS** for service-to-service in cluster. | Baseline for any regulated deployment. |
| **Secrets management** (Vault, AWS Secrets Manager, GCP Secret Manager) — no DB paths or API keys in env files on disk in prod. | Standard enterprise hygiene. |
| **Network policies**: feedback service reachable only from admin plane or batch workers; SQLite **not** on shared mutable NFS without locking strategy. | Lateral movement and corruption risk. |
| **Rate limiting and payload size limits** on `/review/*` and `/rewards/compute`. | Abuse and DoS protection. |

---

## 5. Data platform and storage

| Upgrade | Why it matters |
|--------|----------------|
| **Move from single-file SQLite to a managed store** for high-volume deployments: Postgres (or cloud equivalent) with migrations (Alembic/Flyway), same logical schema. | HA, backups, concurrent writers, observability. |
| **Retention and deletion policies** aligned with HIPAA/GDPR/customer DPA: TTL on raw prompts, anonymized aggregates for metrics. | Legal defensibility. |
| **Encryption at rest** for DB and object storage; **encryption in transit** for all APIs. | Checklist item for enterprise security questionnaires. |
| **Backup/restore drills** documented and tested quarterly. | Real readiness vs paper policy. |

---

## 6. Observability and operations

| Upgrade | Why it matters |
|--------|----------------|
| **Structured logging** (JSON) with correlation IDs spanning gateway → MAD → feedback job → training run ID. | End-to-end incident response. |
| **Metrics**: scoring job duration, rows processed, queue depth, Presidio failures, export row counts, training loss/KL (if exposed). | SLOs and alerting. |
| **Tracing** (OpenTelemetry) for MAD and feedback API hot paths. | Latency and dependency analysis. |
| **Runbooks**: “scoring stuck,” “DB locked,” “Presidio OOM,” “export empty,” “review queue backlog.” | On-call effectiveness. |

---

## 7. Reward quality and anti-gaming

| Upgrade | Why it matters |
|--------|----------------|
| **Tighten citation signal**: RAG-grounded citation verification (chunk text match) instead of regex-only where feasible. | Reduces reward hacking and false “cited” bonuses. |
| **Reward model** trained on `human_feedback_log` to reduce human load while preserving escalation rules. | Scale without diluting the “H” in RLHF. |
| **Canary prompts** in scoring CI: fixed set where expected reward sign is known; fail build on drift. | Catches accidental reward spec regressions. |
| **Judge pipeline clarity**: document whether `v_label` is always from the same judge model/version; version judge prompts and models in DB or sidecar metadata. | Reproducibility and audit. |

---

## 8. Evaluation and release discipline

| Upgrade | Why it matters |
|--------|----------------|
| **Held-out eval suite** (your 12-case + expanded regulatory set) run automatically on **base vs candidate** adapter before promotion. | Objective gate aligned with the paper claims. |
| **Online shadow mode**: new adapter serves a fraction of traffic; compare routing rates, block rates, and human escalations vs baseline. | Safe production learning signal. |
| **Rollback**: one-click revert to previous adapter version in inference path. | Operational safety net. |

---

## 9. Application integration (Gateway + MAD + RLHF)

| Upgrade | Why it matters |
|--------|----------------|
| **Single configuration surface** for `MAD_DB_PATH`, feedback API URL, feature flags (`ENABLE_FEEDBACK_EXPORT`, `ENABLE_HUMAN_QUEUE`). | Fewer misconfigurations across envs. |
| **Async job runner** for `run_reward_scoring` and export (Celery, RQ, Cloud Tasks) instead of only manual CLI. | Production scheduling without shell access. |
| **Idempotent scoring jobs** keyed by watermark (`scored_at` / batch ID) to allow safe retries. | Exactly-once semantics from an ops perspective. |

---

## 10. Deployment topology

| Upgrade | Why it matters |
|--------|----------------|
| **Container images** for gateway, MAD API, feedback API with non-root user, read-only root FS where possible, health/readiness probes. | Kubernetes / ECS readiness. |
| **Horizontal scaling**: stateless API replicas; **single writer** or DB-backed queue for scoring to avoid SQLite write contention. | Matches enterprise traffic patterns. |
| **GPU pool** isolated from interactive services for training jobs. | Cost and stability. |

---

## 11. Compliance and audit

| Upgrade | Why it matters |
|--------|----------------|
| **Immutable audit stream** for human decisions (append-only object storage or WORM bucket), not only local JSONL. | Tamper-evidence and retention policy enforcement. |
| **Access logs** for who viewed which prompt/completion in review tools. | Insider threat and audit requests. |
| **SOC2 / ISO-style control mapping** (optional): map each control to a feature or process above. | Enterprise procurement acceleration. |

---

## 12. Suggested phased roadmap

**Phase A — “Safe production of scoring & review” (weeks)**  
TLS, auth on feedback API, secrets manager, structured logs, metrics, retention policy, Postgres migration plan, rate limits.

**Phase B — “Train and promote with gates” (weeks–months)**  
Pinned training pipeline, artifact store, eval gates, shadow traffic, rollback.

**Phase C — “Scale quality” (ongoing)**  
Reward model from human logs, RAG-grounded citation checks, async RL when stack allows, multi-tenant isolation.

---

## 13. What you can already leverage in-repo

- **MAD SQLite contract** and **`rewards` / `attacks.b_reward` / `is_clean` / `grpo_advantage`** provide a solid data plane to build on.
- **Batch scorer + export JSONL + human review API** are the right primitives to wrap with jobs, auth, and storage upgrades above.

---

*This is a roadmap, not a commitment to implement every item. Prioritize with your compliance officer, security reviewer, and first design partner customer.*
