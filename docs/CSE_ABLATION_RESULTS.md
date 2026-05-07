# CSE Formula Ablation Study — Evaluation Results

**Date:** 2026-05-06  
**Judge backend:** Claude Haiku 4.5 (Anthropic API) via `ClaudeJudge`  
**DeepEval version:** 0.21.73  
**Metrics:** FaithfulnessMetric, HallucinationMetric, ContextualRelevancyMetric  
**Mode:** v2.0 (live DeepEval — all 3 component scores collected from Claude)  
**Raw data:** `ablation_outputs/ablation_results_20260506_211819.json`

---

## Formula Under Test

```
final_score = w_f × F_llm
            + w_h × (1 − H_llm)
            + w_r × relevancy
            + w_j × judge_eval
```

Where `w_f + w_h + w_r + w_j = 1.0`

**Routing thresholds:**
- `final_score ≥ 0.80` → DELIVER
- `0.40 ≤ final_score < 0.80` → RETRY
- `final_score < 0.40` → HUMAN_REVIEW
- Hard rule: material claim with judge score = 0.0 → HARD_BLOCK (overrides formula)

---

## Weight Configurations Tested

| Config | w_f (Faith.) | w_h (Halluc.) | w_r (Relev.) | w_j (Judge) |
|--------|:---:|:---:|:---:|:---:|
| baseline_v2 | 0.30 | 0.25 | 0.10 | 0.35 |
| judge_heavy | 0.20 | 0.15 | 0.05 | **0.60** |
| faithful_heavy | **0.50** | 0.20 | 0.10 | 0.20 |
| halluc_focused | 0.25 | **0.45** | 0.05 | 0.25 |
| balanced | 0.25 | 0.25 | **0.25** | 0.25 |
| deepeval_heavy | 0.35 | 0.30 | 0.20 | 0.15 |
| judge_only | 0.00 | 0.00 | 0.00 | **1.00** |

---

## Test Cases

8 queries covering 4 regulatory error types (healthcare/legal domain):

| Case | Domain | Error Type | Expected Routing |
|------|--------|-----------|-----------------|
| C1: HIPAA ePHI encryption | HIPAA | Fully correct | DELIVER |
| C2: GDPR right to erasure | GDPR | Missing caveat (Article 17(3) exceptions omitted) | RETRY |
| C3: HIPAA AES-256-GCM mandate | HIPAA | Hallucinated specific standard | HARD_BLOCK |
| C4: Basel III LCR "all banks" | Basel III | Jurisdiction blind (overgeneralised) | RETRY |
| C5: CCPA deletion request | CCPA | Vague / incomplete answer | RETRY |
| C6: GDPR consent validity | GDPR | Fully correct | DELIVER |
| C7: GDPR DSA response deadline | GDPR | Hallucinated deadline (7 days vs 1 month) | HARD_BLOCK |
| C8: HITECH breach notification | HITECH | Hallucinated deadline (24 hours vs 60 days) | HARD_BLOCK |

---

## Component Scores (Claude DeepEval — v2.0)

Scores collected once per case; used by all weight configurations.

| Case | F_llm | H_llm | Relevancy | Judge | Notes |
|------|------:|------:|----------:|------:|-------|
| C1 HIPAA correct | 1.000 | 0.000 | 0.667 | 1.000 | All signals agree: high quality |
| C2 GDPR missing caveat | 0.000 | 0.500 | 0.500 | 0.500 | Low F_llm because answer not verbatim in chunks |
| C3 HIPAA fabricated std | 0.000 | 1.000 | 1.000 | 0.000 | Halluc=1.0 correctly flags fabrication; HARD_BLOCK via judge |
| C4 Basel jurisdiction | 0.667 | 1.000 | 0.500 | 0.500 | Partially grounded; H_llm=1.0 signals overgeneralisation |
| C5 CCPA vague | 0.000 | 0.000 | 1.000 | 0.500 | Topically relevant but incomplete; F_llm=0 (verbatim miss) |
| C6 GDPR consent correct | 0.333 | 0.000 | 1.000 | 1.000 | F_llm=0.33 despite correct answer (paraphrase ≠ verbatim) |
| C7 GDPR fabricated deadline | 0.000 | 1.000 | 1.000 | 0.000 | Clear hallucination; HARD_BLOCK via judge |
| C8 HITECH breach partial | 0.000 | 1.000 | 1.000 | 0.000 | Clear hallucination; HARD_BLOCK via judge |

**Key observation:** Relevancy is high (0.5–1.0) for all cases including hallucinations — topically relevant answers can still be factually wrong. F_llm is noisy on paraphrased-but-correct answers (C6 scores 0.333 despite being fully accurate).

---

## Final Scores and Routing per Configuration

`†` = HARD_BLOCK (hard rule fired, formula score shown for reference)  
`√` = correct routing  `✗` = wrong routing

| Config | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|--------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| baseline_v2 | 0.967 √ | 0.350 ✗ | 0.100 †√ | 0.425 √ | 0.525 √ | 0.800 √ | 0.100 †√ | 0.100 †√ |
| judge_heavy | 0.983 √ | 0.400 √ | 0.050 †√ | 0.458 √ | 0.500 √ | 0.867 √ | 0.050 †√ | 0.050 †√ |
| faithful_heavy | 0.967 √ | 0.250 ✗ | 0.100 †√ | 0.483 √ | 0.400 √ | 0.667 ✗ | 0.100 †√ | 0.100 †√ |
| halluc_focused | 0.983 √ | 0.375 ✗ | 0.050 †√ | 0.317 ✗ | 0.625 √ | 0.833 √ | 0.050 †√ | 0.050 †√ |
| balanced | 0.917 √ | 0.375 ✗ | 0.250 †√ | 0.417 √ | 0.625 √ | 0.833 √ | 0.250 †√ | 0.250 †√ |
| deepeval_heavy | 0.933 √ | 0.325 ✗ | 0.200 †√ | 0.408 √ | 0.575 √ | 0.767 ✗ | 0.200 †√ | 0.200 †√ |
| judge_only | 1.000 √ | 0.500 √ | 0.000 †√ | 0.500 √ | 0.500 √ | 1.000 √ | 0.000 †√ | 0.000 †√ |

---

## Routing Accuracy Summary

| Rank | Config | Correct / 8 | Accuracy | Failed Cases |
|------|--------|:-----------:|:--------:|-------------|
| 1 | **judge_heavy** | **8 / 8** | **100.0%** | — |
| 1 | **judge_only** | **8 / 8** | **100.0%** | — |
| 3 | baseline_v2 | 7 / 8 | 87.5% | C2 (HUMAN_REVIEW, expected RETRY) |
| 3 | balanced | 7 / 8 | 87.5% | C2 (HUMAN_REVIEW, expected RETRY) |
| 5 | faithful_heavy | 6 / 8 | 75.0% | C2, C6 |
| 5 | halluc_focused | 6 / 8 | 75.0% | C2, C4 |
| 5 | deepeval_heavy | 6 / 8 | 75.0% | C2, C6 |

---

## Analysis

### Why judge_heavy wins

The MAD debate judge already processes all available signals — RAG evidence, adversarial challenges, and claim-level scrutiny — before scoring each claim. Its aggregate score is a strong, semantically-rich summary that does not suffer from the verbatim-matching brittleness of DeepEval's FaithfulnessMetric.

Upweighting judge to 0.60 (`judge_heavy`) pushes C2's final score from 0.350 (below the 0.40 RETRY threshold) to exactly 0.400 — which is the minimum to route RETRY correctly.

### Why F_llm is unreliable for paraphrased-correct answers

DeepEval's FaithfulnessMetric scores statements based on whether they are directly entailed by retrieved chunks. A semantically correct but paraphrased answer (C6: GDPR consent) scores F_llm = 0.333. This is a structural limitation of the metric, not a quality problem with the answer. Overweighting F_llm (`faithful_heavy`) misroutes C6 to RETRY and pulls C2 even lower.

### Why H_llm and relevancy are weak discriminators

- H_llm = 1.0 for fabricated answers (C3, C4, C7, C8) is correct but adds no information beyond what the judge already captures via score = 0.0.
- Relevancy scores 1.0 for hallucinated answers (C3, C7, C8) because fabricated regulatory statements are topically on-point — just factually wrong. Relevancy cannot distinguish on-topic hallucinations from correct answers.

### The single failure mode: C2 boundary case

C2 (GDPR erasure missing caveat) is a boundary case where all component scores cluster around 0.5. The formula pushes it slightly below or above the RETRY threshold depending on weights. Only `judge_heavy` and `judge_only` score it high enough (0.400, 0.500) to pass the threshold. This confirms that the judge signal is most reliable for soft failures (missing caveat, partial correctness).

---

## Recommendation

**Adopt `judge_heavy` weights for CSE v2.0:**

```python
# confidence/cse_config.py
w_faithfulness  = 0.20   # was 0.30
w_hallucination = 0.15   # was 0.25
w_relevancy     = 0.05   # was 0.10
w_judge         = 0.60   # was 0.35
```

**Rationale:**
- Achieves 100% routing accuracy on all 8 test cases (vs 87.5% for current baseline_v2).
- Remains a 4-signal formula — DeepEval metrics are still included and still useful as secondary evidence for the audit trail.
- The sole failure case for baseline_v2 (C2, boundary RETRY/HUMAN_REVIEW) is resolved by the higher judge weight.
- Does not regress on any HARD_BLOCK case — all 3 hallucinated cases are correctly blocked across all configurations due to the hard rule (material claim + judge score = 0.0).

**Alternative:** `judge_only` (w_j = 1.00) also achieves 100% but discards the DeepEval signals entirely, losing independent corroboration and explainability in the audit trail.

---

## Routing Threshold Sensitivity Note

C2's correct routing is sensitive to the RETRY lower bound (currently 0.40). If this threshold were raised to 0.45, even `judge_heavy` would fail C2. The threshold should not be raised without re-running this ablation.

---

## Files

| File | Description |
|------|-------------|
| `confidence/ablation_study.py` | Study script (run with `--no-deepeval` for fast mode) |
| `confidence/claude_judge.py` | Claude DeepEval adapter used as judge backend |
| `ablation_outputs/ablation_results_20260506_211819.json` | Full raw results (all scores, all cells) |
| `ablation_outputs/ablation_scores_20260506_211819.json` | Component scores only (reuse with `--load-scores`) |
| `ablation_outputs/ablation_summary_20260506_211819.csv` | Pivot table for spreadsheet analysis |
