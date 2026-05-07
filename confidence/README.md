# Confidence Scoring Engine (CSE) v1.1

Enterprise-grade scoring component that combines DeepEval signals and MAD judge verdicts into a deterministic routing decision.

## Scoring Formula

```
weighted_raw = Σ (weight_i × signal_i)   for all *available* signals
final_score  = clamp(weighted_raw − penalties, 0.0, 1.0)
```

### Default Weights (FULL mode)

| Signal | Default Weight | Source |
|--------|---------------|--------|
| `faithfulness_score` | 0.25 | DeepEval `FaithfulnessMetric` |
| `hallucination_risk_inverse` (1 − H_llm) | 0.20 | DeepEval `HallucinationMetric` |
| `contextual_relevancy_score` | 0.15 | DeepEval `ContextualRelevancyMetric` |
| `judge_eval_score` | 0.35 | MAD judge aggregate |
| `context_quality_score` | 0.05 | Lightweight context checks |

When a signal is unavailable, its weight is **dropped and the remaining weights are re-normalised** to sum to 1.0. Missing signals appear as `None` in `score_breakdown` and are never substituted with 0.

All weights and thresholds live in `cse_config.py` and are overridable at runtime via environment variables or a custom `CSEScoringConfig` instance.

## Routing Priority

Routing is evaluated in strict priority order:

```
1. HARD_BLOCK   — any is_material claim with judge score = 0.0
2. HUMAN_REVIEW — ERROR_FALLBACK mode, or retry limit reached,
                   or final_score < human_review_threshold (0.55)
3. RETRY        — human_review_threshold ≤ final_score < deliver_threshold (0.78)
4. DELIVER      — final_score ≥ deliver_threshold, no hard flags
```

HARD_BLOCK always overrides the aggregate score — even a perfect aggregate score does not override a false material claim.

## Fallback Modes

| Mode | Condition | Behaviour |
|------|-----------|-----------|
| `FULL` | DeepEval + MAD judge available | All five signals used |
| `JUDGE_ONLY_FALLBACK` | DeepEval unavailable | Weights re-normalised to judge + claims |
| `CONTEXT_ONLY_FALLBACK` | MAD judge unavailable | DeepEval signals only; DELIVER avoided unless strong |
| `ERROR_FALLBACK` | Both unavailable | Forces `HUMAN_REVIEW` |

## Versioning

Every CSE result carries:

| Field | Example |
|-------|---------|
| `cse_version` | `cse_v1.1_weighted_claim_aware` |
| `config_version` | `v1.1` |
| `formula_version` | `weighted_claim_aware_v1` |

These are emitted in `metadata` on every result and enable side-by-side comparisons across formula versions.

## Output Contract

```python
CSEResult:
    final_score:      float           # 0.0–1.0, always clamped
    routing_decision: str             # DELIVER | RETRY | HUMAN_REVIEW | HARD_BLOCK
    scoring_mode:     str             # FULL | JUDGE_ONLY_FALLBACK | ...
    score_breakdown:  ScoreBreakdown  # per-signal scores + applied weights
    triggered_flags:  TriggeredFlags  # boolean flags driving the decision
    top_failed_claims: List[FailedClaim]
    explanation:      str             # human-readable routing reason
    retry_reasons:    List[str]       # populated when routing = RETRY
    review_reasons:   List[str]       # populated when routing = HUMAN_REVIEW
    block_reasons:    List[str]       # populated when routing = HARD_BLOCK
    metadata:         CSEMetadata     # timestamp, versions, request_id, chunk_ids
    # legacy compat (unchanged from v0.1 / v2.0)
    components:       ComponentScores
    version:          str
    hard_blocked:     bool
    error:            str
```

## Usage

```python
from confidence.scorer import ConfidenceScorer
from confidence.cse_config import CSEScoringConfig

# Default config (all weights and thresholds from cse_config.py)
scorer = ConfidenceScorer()
result = scorer.score(
    query          = "What does HIPAA require for PHI encryption?",
    llm_answer     = "...",
    rag_chunks     = ["45 CFR § 164.312 ...", "NIST SP 800-111 ..."],
    final_claims   = claims,
    judge_verdicts = verdicts,
    retry_count    = 0,
    request_id     = "req-abc123",
    policy_domain  = "HIPAA",
)
print(result.routing_decision)          # e.g. "DELIVER"
print(result.explanation)               # human-readable reason
print(result.score_breakdown.as_dict()) # full breakdown for audit logs

# Custom thresholds
cfg = CSEScoringConfig(deliver_threshold=0.85, human_review_threshold=0.60)
scorer = ConfidenceScorer(config=cfg)
```

## Environment Variable Overrides

| Variable | Type | Default |
|----------|------|---------|
| `CSE_DELIVER_THRESHOLD` | float | 0.78 |
| `CSE_HUMAN_REVIEW_THRESHOLD` | float | 0.55 |
| `CSE_MAX_RETRY_COUNT` | int | 2 |
| `JUDGE_BACKEND` | `ollama` \| `claude` | `ollama` |

## Running Tests

```bash
# Unit tests (no external services required)
pytest confidence/tests/test_cse.py -v

# Manual integration harness (requires Ollama or set JUDGE_BACKEND=claude)
PYTHONPATH=multi_agent_debate:rag_folder:. python confidence/test_harness.py --no-deepeval
PYTHONPATH=multi_agent_debate:rag_folder:. python confidence/test_harness.py --verbose
```

## Claim Severity

| Severity | Weight | Trigger |
|----------|--------|---------|
| `critical` | 1.0 | Reserved — pending `severity` field in MAD models |
| `material` | 0.8 | `is_material=True` on the `Claim` |
| `minor` | 0.2 | `is_material=False` |

A material claim with judge score = 0.0 triggers **HARD_BLOCK regardless of all other signals**.

## Penalties

| Condition | Default Penalty |
|-----------|----------------|
| Missing context | −0.10 |
| Low context relevancy (< 0.40) | −0.05 |
| Insufficient chunks (< 2) | −0.05 |
| Judge disagreement (σ ≥ 0.30) | −0.05 |

Penalties are deducted after the weighted sum. The final score is always clamped to [0.0, 1.0].
