"""
Gateway Formula Ablation Study
================================
Grid-searches over PII/JB/PI weight combinations and two conditions:
  1. classifier-only  (no LLM judge)
  2. classifier+judge (Claude Haiku upgrade-only)

Outputs:
  gateway/ablation/results.csv    — full metric table
  gateway/ablation/summary.md     — ranked results + recommended weights

Usage:
  # From repo root:
  python -m gateway.ablation.run_ablation

  # With judge (requires ANTHROPIC_API_KEY in env):
  ANTHROPIC_API_KEY=sk-ant-... python -m gateway.ablation.run_ablation

  # Classifier-only (no API key needed):
  python -m gateway.ablation.run_ablation --no-judge
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure repo root is on path when running as a module
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gateway.ablation.dataset import get_dataset
from gateway.decision_engine import DecisionEngine, Decision
from gateway.judge import GatewayJudge

_LABELS = ["PASS", "ESCALATE", "BLOCK"]
_SEVERITY = {"PASS": 0, "ESCALATE": 1, "BLOCK": 2}

WEIGHT_GRID = [
    (0.30, 0.40, 0.30),          # current baseline
    (1/3,  1/3,  1/3),           # equal weights
    (0.20, 0.50, 0.30),          # jailbreak-heavy
    (0.10, 0.60, 0.30),          # jailbreak-dominant
    (0.20, 0.40, 0.40),          # PI-heavy
    (0.30, 0.30, 0.40),          # PI-dominant
    (0.40, 0.30, 0.30),          # PII-heavy
    (0.25, 0.50, 0.25),          # symmetric jailbreak emphasis
]


def _classify(engine: DecisionEngine, example: dict) -> str:
    result = engine.decide(
        raw_input=example["query"],
        pii_score=example["pii"],
        jb_score=example["jb"],
        pi_score=example["pi"],
    )
    return result.decision.value


def _apply_judge(judge: GatewayJudge, example: dict, classifier_decision: str) -> str:
    if not judge.enabled:
        return classifier_decision
    jr = judge.judge(
        query=example["query"],
        pii_score=example["pii"],
        jb_score=example["jb"],
        pi_score=example["pi"],
        initial_decision=classifier_decision,
    )
    # Upgrade-only (already enforced inside judge, but double-check)
    if _SEVERITY.get(jr.verdict, 0) > _SEVERITY.get(classifier_decision, 0):
        return jr.verdict
    return classifier_decision


def _compute_metrics(predictions: list[str], ground_truth: list[str]) -> dict:
    n = len(predictions)
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    accuracy = correct / n if n else 0.0

    metrics = {"accuracy": round(accuracy, 4)}

    for label in _LABELS:
        tp = sum(p == label and g == label for p, g in zip(predictions, ground_truth))
        fp = sum(p == label and g != label for p, g in zip(predictions, ground_truth))
        fn = sum(p != label and g == label for p, g in zip(predictions, ground_truth))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        key = label.lower()
        metrics[f"precision_{key}"] = round(precision, 4)
        metrics[f"recall_{key}"] = round(recall, 4)
        metrics[f"f1_{key}"] = round(f1, 4)

    # False Negative Rate on BLOCK (missed threats — most critical)
    block_gt = [g == "BLOCK" for g in ground_truth]
    block_pred = [p == "BLOCK" for p in predictions]
    fn_block = sum(not p and g for p, g in zip(block_pred, block_gt))
    total_block_gt = sum(block_gt)
    metrics["fnr_block"] = round(fn_block / total_block_gt if total_block_gt else 0.0, 4)

    # False Positive Rate on PASS (over-blocking safe queries)
    pass_gt = [g == "PASS" for g in ground_truth]
    pass_pred = [p == "PASS" for p in predictions]
    fp_pass = sum(not p and g for p, g in zip(pass_pred, pass_gt))
    total_pass_gt = sum(pass_gt)
    metrics["fpr_pass"] = round(fp_pass / total_pass_gt if total_pass_gt else 0.0, 4)

    # Over-escalate rate (safe queries pushed to ESCALATE)
    over_esc = sum(p == "ESCALATE" and g == "PASS" for p, g in zip(predictions, ground_truth))
    metrics["over_escalate_rate"] = round(over_esc / total_pass_gt if total_pass_gt else 0.0, 4)

    return metrics


def run_ablation(use_judge: bool = True):
    dataset = get_dataset()
    ground_truth = [d["label"] for d in dataset]

    judge = GatewayJudge() if use_judge else None
    if use_judge and not judge.enabled:
        print("[Ablation] Warning: ANTHROPIC_API_KEY not set — judge condition will mirror classifier-only.")

    results = []

    for pii_w, jb_w, pi_w in WEIGHT_GRID:
        engine = DecisionEngine(
            pii_weight=pii_w,
            jb_weight=jb_w,
            pi_weight=pi_w,
        )

        # Condition 1: classifier-only
        preds_clf = [_classify(engine, ex) for ex in dataset]
        metrics_clf = _compute_metrics(preds_clf, ground_truth)
        results.append({
            "weights_pii": pii_w,
            "weights_jb": jb_w,
            "weights_pi": pi_w,
            "condition": "classifier_only",
            **metrics_clf,
        })

        # Condition 2: classifier + judge
        if use_judge and judge:
            preds_judge = []
            for ex, clf_pred in zip(dataset, preds_clf):
                final = _apply_judge(judge, ex, clf_pred)
                preds_judge.append(final)
            metrics_judge = _compute_metrics(preds_judge, ground_truth)
            results.append({
                "weights_pii": pii_w,
                "weights_jb": jb_w,
                "weights_pi": pi_w,
                "condition": "classifier_plus_judge",
                **metrics_judge,
            })

        label = f"pii={pii_w:.2f} jb={jb_w:.2f} pi={pi_w:.2f}"
        acc = metrics_clf["accuracy"]
        fnr = metrics_clf["fnr_block"]
        print(f"  [{label}] clf-only → accuracy={acc:.3f} fnr_block={fnr:.3f}")

    return results


def save_csv(results: list[dict], path: Path):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Ablation] Results saved to {path}")


def save_summary(results: list[dict], path: Path):
    # Rank by: lowest fnr_block first (safety), then highest accuracy
    ranked = sorted(results, key=lambda r: (r["fnr_block"], -r["accuracy"]))

    lines = ["# Gateway Formula Ablation Study — Summary\n"]
    lines.append("## Ranking (lowest missed-threat rate first, then highest accuracy)\n")
    lines.append("| Rank | Condition | PII w | JB w | PI w | Accuracy | F1-BLOCK | FNR-BLOCK | FPR-PASS | Over-Esc |")
    lines.append("|------|-----------|-------|------|------|----------|----------|-----------|----------|----------|")

    for i, r in enumerate(ranked[:16], 1):
        lines.append(
            f"| {i} | {r['condition']} "
            f"| {r['weights_pii']:.2f} | {r['weights_jb']:.2f} | {r['weights_pi']:.2f} "
            f"| {r['accuracy']:.3f} | {r['f1_block']:.3f} "
            f"| {r['fnr_block']:.3f} | {r['fpr_pass']:.3f} | {r['over_escalate_rate']:.3f} |"
        )

    best = ranked[0]
    lines.append(f"\n## Recommended Configuration\n")
    lines.append(
        f"**Best weights**: PII={best['weights_pii']:.2f}, JB={best['weights_jb']:.2f}, PI={best['weights_pi']:.2f}  \n"
        f"**Condition**: {best['condition']}  \n"
        f"**Accuracy**: {best['accuracy']:.3f}  \n"
        f"**F1-BLOCK**: {best['f1_block']:.3f}  \n"
        f"**FNR-BLOCK** (missed threats): {best['fnr_block']:.3f}  \n"
        f"**FPR-PASS** (over-blocking): {best['fpr_pass']:.3f}  \n"
    )

    lines.append("\n## Metric Definitions\n")
    lines.append("- **FNR-BLOCK**: False Negative Rate on BLOCK class — fraction of real threats that slipped through (lower is safer)")
    lines.append("- **FPR-PASS**: False Positive Rate on PASS class — fraction of safe queries that got blocked/escalated (lower is more usable)")
    lines.append("- **Over-Esc**: Fraction of safe queries pushed to ESCALATE (review queue noise)")
    lines.append("- **F1-BLOCK**: Harmonic mean of BLOCK precision and recall (higher = better threat detection)")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Ablation] Summary saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Gateway formula ablation study")
    parser.add_argument("--no-judge", action="store_true", help="Skip judge condition (classifier-only)")
    args = parser.parse_args()

    use_judge = not args.no_judge

    print(f"[Ablation] Running with {len(WEIGHT_GRID)} weight combinations, judge={'enabled' if use_judge else 'disabled'}")
    print(f"[Ablation] Dataset: 60 examples (20 PASS / 20 ESCALATE / 20 BLOCK)\n")

    results = run_ablation(use_judge=use_judge)

    out_dir = Path(__file__).parent
    save_csv(results, out_dir / "results.csv")
    save_summary(results, out_dir / "summary.md")

    print("\n[Ablation] Done.")


if __name__ == "__main__":
    main()
