# GRPO LoRA Evaluation Artifacts

Full evaluation artifacts are shared through Google Drive with the adapter zips:

```text
https://drive.google.com/drive/u/4/folders/1NbaX6a8MwDxj4f0_miBnTFbCFvUBvFkv
```

Expected eval archive:

```text
eval_outputs-20260508T230836Z-3-001.zip
```

Expected contents:

```text
eval_outputs/
├── base_test/
│   ├── agent_a_eval_summary.json
│   ├── agent_a_eval_rows.csv
│   ├── agent_b_eval_summary.json
│   └── agent_b_eval_rows.csv
└── test_after_150/
    ├── agent_a_eval_summary.json
    ├── agent_a_eval_rows.csv
    ├── agent_b_eval_summary.json
    └── agent_b_eval_rows.csv
```

## Held-Out Test Summary

| Metric | Base Agent A | GRPO Agent A | Base Agent B | GRPO Agent B |
| --- | ---: | ---: | ---: | ---: |
| Valid JSON rate | 65.6% | 100.0% | 65.6% | 100.0% |
| Mean confidence | 0.857 | 0.375 | 0.629 | 0.412 |
| Mean absolute error | 0.533 | 0.128 | 0.495 | 0.141 |
| Brier MSE | 0.3667 | 0.0516 | 0.3257 | 0.0616 |
| Brier reward | -0.1762 | 0.1594 | -0.1352 | 0.1494 |
| Overconfident unsupported rate | 87.5% | 0.0% | 75.0% | 0.0% |
| Overconfident partial rate | 100.0% | 0.0% | 66.7% | 0.0% |
| Verdict exact match | 66.7% | 78.1% | 57.1% | 75.0% |

The GRPO adapters improved JSON reliability, reduced Brier error, and removed high-confidence unsupported/partial outputs on the held-out test split.
