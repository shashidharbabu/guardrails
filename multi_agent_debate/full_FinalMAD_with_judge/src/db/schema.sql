PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS queries (
    query_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    user_query      TEXT NOT NULL,
    rag_chunk_ids   TEXT NOT NULL,
    rag_chunks      TEXT NOT NULL,
    baseline_answer TEXT NOT NULL,
    baseline_model  TEXT NOT NULL,
    timestamp       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id           TEXT PRIMARY KEY,
    query_id           TEXT NOT NULL,
    claim_text         TEXT NOT NULL,
    claim_index        INTEGER NOT NULL,
    is_material        BOOLEAN NOT NULL,
    is_critical        BOOLEAN NOT NULL,
    confidence_prior   REAL NOT NULL,
    coverage_check     BOOLEAN NOT NULL,
    coverage_ratio     REAL NOT NULL,
    FOREIGN KEY (query_id) REFERENCES queries(query_id)
);

CREATE TABLE IF NOT EXISTS agent_outputs (
    output_id           TEXT PRIMARY KEY,
    claim_id            TEXT NOT NULL,
    agent_role          TEXT NOT NULL,
    round_num           INTEGER NOT NULL,
    verdict             TEXT NOT NULL,
    reasoning           TEXT NOT NULL,
    evidence_cited      TEXT NOT NULL,
    confidence_internal REAL NOT NULL,
    raw_response        TEXT NOT NULL,
    latency_ms          INTEGER NOT NULL,
    tokens_in           INTEGER NOT NULL,
    tokens_out          INTEGER NOT NULL,
    timestamp           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
);

CREATE TABLE IF NOT EXISTS agent_deltas (
    delta_id            TEXT PRIMARY KEY,
    claim_id            TEXT NOT NULL,
    agent_role          TEXT NOT NULL,
    confidence_r0       REAL NOT NULL,
    confidence_r1       REAL NOT NULL,
    delta               REAL NOT NULL,
    verdict_r0          TEXT NOT NULL,
    verdict_r1          TEXT NOT NULL,
    verdict_changed     BOOLEAN NOT NULL,
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
);

CREATE TABLE IF NOT EXISTS judge_verdicts (
    verdict_id          TEXT PRIMARY KEY,
    claim_id            TEXT NOT NULL,
    v_label             REAL NOT NULL,
    judge_confidence    REAL NOT NULL,
    judge_reasoning     TEXT NOT NULL,
    evidence_chunk_ids  TEXT NOT NULL,
    judge_model         TEXT NOT NULL,
    raw_response        TEXT NOT NULL,
    latency_ms          INTEGER NOT NULL,
    timestamp           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
);

CREATE INDEX IF NOT EXISTS idx_claims_query ON claims(query_id);
CREATE INDEX IF NOT EXISTS idx_outputs_claim ON agent_outputs(claim_id);
CREATE INDEX IF NOT EXISTS idx_outputs_role_round ON agent_outputs(agent_role, round_num);
CREATE INDEX IF NOT EXISTS idx_deltas_claim ON agent_deltas(claim_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_claim ON judge_verdicts(claim_id);
