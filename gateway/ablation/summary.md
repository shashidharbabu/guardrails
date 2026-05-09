# Gateway Formula Ablation Study — Summary

## Ranking (lowest missed-threat rate first, then highest accuracy)

| Rank | Condition | PII w | JB w | PI w | Accuracy | F1-BLOCK | FNR-BLOCK | FPR-PASS | Over-Esc |
|------|-----------|-------|------|------|----------|----------|-----------|----------|----------|
| 1 | classifier_only | 0.40 | 0.30 | 0.30 | 0.717 | 1.000 | 0.000 | 0.000 | 0.000 |
| 2 | classifier_only | 0.33 | 0.33 | 0.33 | 0.683 | 1.000 | 0.000 | 0.000 | 0.000 |
| 3 | classifier_only | 0.30 | 0.40 | 0.30 | 0.667 | 1.000 | 0.000 | 0.000 | 0.000 |
| 4 | classifier_only | 0.20 | 0.50 | 0.30 | 0.667 | 1.000 | 0.000 | 0.000 | 0.000 |
| 5 | classifier_only | 0.10 | 0.60 | 0.30 | 0.667 | 1.000 | 0.000 | 0.000 | 0.000 |
| 6 | classifier_only | 0.20 | 0.40 | 0.40 | 0.667 | 1.000 | 0.000 | 0.000 | 0.000 |
| 7 | classifier_only | 0.30 | 0.30 | 0.40 | 0.667 | 1.000 | 0.000 | 0.000 | 0.000 |
| 8 | classifier_only | 0.25 | 0.50 | 0.25 | 0.667 | 1.000 | 0.000 | 0.000 | 0.000 |

## Recommended Configuration

**Best weights**: PII=0.40, JB=0.30, PI=0.30  
**Condition**: classifier_only  
**Accuracy**: 0.717  
**F1-BLOCK**: 1.000  
**FNR-BLOCK** (missed threats): 0.000  
**FPR-PASS** (over-blocking): 0.000  


## Metric Definitions

- **FNR-BLOCK**: False Negative Rate on BLOCK class — fraction of real threats that slipped through (lower is safer)
- **FPR-PASS**: False Positive Rate on PASS class — fraction of safe queries that got blocked/escalated (lower is more usable)
- **Over-Esc**: Fraction of safe queries pushed to ESCALATE (review queue noise)
- **F1-BLOCK**: Harmonic mean of BLOCK precision and recall (higher = better threat detection)