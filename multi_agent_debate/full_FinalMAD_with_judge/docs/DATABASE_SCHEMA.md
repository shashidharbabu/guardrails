# Database Schema

All data is stored in a single SQLite file (default: `mad.db`).

## Tables

### queries
| Column | Type | Description |
|--------|------|-------------|
| query_id | TEXT PK | Unique query identifier |
| run_id | TEXT | Run identifier (groups multiple queries) |
| user_query | TEXT | Original user question |
| rag_chunk_ids | TEXT (JSON) | List of chunk IDs used for RAG |
| rag_chunks | TEXT (JSON) | Full chunk objects with text |
| baseline_answer | TEXT | Baseline LLM answer before debate |
| baseline_model | TEXT | Model name used for baseline |
| timestamp | TIMESTAMP | Row creation time |

### claims
| Column | Type | Description |
|--------|------|-------------|
| claim_id | TEXT PK | UUID |
| query_id | TEXT FK | Parent query |
| claim_text | TEXT | The atomic factual claim |
| claim_index | INTEGER | Order within query |
| is_material | BOOLEAN | Affects correctness of answer |
| is_critical | BOOLEAN | Wrong = real-world harm |
| confidence_prior | REAL | Decomposer prior confidence (0.5-0.9) |
| coverage_check | BOOLEAN | Decomposer coverage check passed |
| coverage_ratio | REAL | Fraction of answer words covered |

### agent_outputs
| Column | Type | Description |
|--------|------|-------------|
| output_id | TEXT PK | UUID |
| claim_id | TEXT FK | Parent claim |
| agent_role | TEXT | agent_a or agent_b |
| round_num | INTEGER | 0 (independent) or 1 (peer-aware) |
| verdict | TEXT | SUPPORTED / PARTIAL / NOT_SUPPORTED / IDK |
| reasoning | TEXT | Agent reasoning (max 3 sentences) |
| evidence_cited | TEXT (JSON) | [{chunk_id, relevant_quote}] |
| confidence_internal | REAL | Agent self-reported confidence (0.0-1.0) |
| raw_response | TEXT | Raw LLM output before parsing |
| latency_ms | INTEGER | Inference latency |
| tokens_in / tokens_out | INTEGER | Token counts |
| timestamp | TIMESTAMP | Row creation time |

**Note:** `confidence_internal` is stored in the DB but is STRIPPED before being shown to the peer agent or judge (via `strip_for_peer()`).

### agent_deltas
Tracks verdict and confidence changes from Round 0 → Round 1.

| Column | Type | Description |
|--------|------|-------------|
| delta_id | TEXT PK | UUID |
| claim_id | TEXT FK | Parent claim |
| agent_role | TEXT | agent_a or agent_b |
| confidence_r0 / confidence_r1 | REAL | Confidence before/after peer review |
| delta | REAL | confidence_r1 - confidence_r0 |
| verdict_r0 / verdict_r1 | TEXT | Verdicts before/after |
| verdict_changed | BOOLEAN | Whether verdict changed after peer review |

### judge_verdicts
| Column | Type | Description |
|--------|------|-------------|
| verdict_id | TEXT PK | UUID |
| claim_id | TEXT FK | Parent claim |
| v_label | REAL | 0.0=NOT_SUPPORTED, 0.5=PARTIAL, 1.0=SUPPORTED |
| judge_confidence | REAL | Judge self-reported confidence |
| judge_reasoning | TEXT | Judge analysis |
| evidence_chunk_ids | TEXT (JSON) | Chunks judge cited |
| judge_model | TEXT | Judge model name |
| raw_response | TEXT | Raw judge LLM output |
| latency_ms | INTEGER | Inference latency |
| timestamp | TIMESTAMP | Row creation time |

## Indexes
- `idx_claims_query` on claims(query_id)
- `idx_outputs_claim` on agent_outputs(claim_id)
- `idx_outputs_role_round` on agent_outputs(agent_role, round_num)
- `idx_deltas_claim` on agent_deltas(claim_id)
- `idx_verdicts_claim` on judge_verdicts(claim_id)
