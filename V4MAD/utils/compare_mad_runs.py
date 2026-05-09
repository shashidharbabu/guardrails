"""
Compare base-agent MAD results against GRPO-LoRA MAD results.

Example:
    python utils/compare_mad_runs.py \
      --base-db mad_v4.db \
      --grpo-db mad_v4_grpo.db \
      --out-csv grpo_mad_comparison_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


VERDICT_TO_LABEL = {
    "NOT_SUPPORTED": 0.0,
    "IDK": 0.5,
    "PARTIAL": 0.5,
    "SUPPORTED": 1.0,
}


def _rows(conn: sqlite3.Connection) -> list[dict]:
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(
        """
        SELECT
            c.query_id,
            c.claim_id,
            c.claim_index,
            c.claim_text,
            j.v_label,
            ao.agent_role,
            ao.round_num,
            ao.verdict,
            ao.confidence_internal,
            ao.raw_response,
            ao.latency_ms
        FROM claims c
        JOIN judge_verdicts j ON j.claim_id = c.claim_id
        JOIN agent_outputs ao ON ao.claim_id = c.claim_id
        ORDER BY c.query_id, c.claim_index, ao.agent_role, ao.round_num
        """
    )]


def _is_valid_json(raw: str) -> bool:
    try:
        parsed = json.loads(raw)
    except Exception:
        return False
    return isinstance(parsed, dict)


def _verdict_mae(verdict: str, v_label: float) -> float:
    return abs(VERDICT_TO_LABEL.get(verdict, 0.5) - float(v_label))


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}

    by_claim_role = defaultdict(dict)
    for r in rows:
        by_claim_role[(r["claim_id"], r["agent_role"])][int(r["round_num"])] = r

    r1_rows = [r for r in rows if int(r["round_num"]) == 1]
    unsupported_r1 = [r for r in r1_rows if float(r["v_label"]) == 0.0]
    partial_r1 = [r for r in r1_rows if float(r["v_label"]) == 0.5]

    confidence_by_label = defaultdict(list)
    verdict_distribution = Counter()
    json_valid = 0
    exact = 0
    maes = []
    latencies = []

    for r in r1_rows:
        v = float(r["v_label"])
        c = float(r["confidence_internal"])
        confidence_by_label[v].append(c)
        verdict_distribution[r["verdict"]] += 1
        json_valid += int(_is_valid_json(r["raw_response"] or ""))
        exact += int(_verdict_mae(r["verdict"], v) == 0.0)
        maes.append(_verdict_mae(r["verdict"], v))
        latencies.append(int(r["latency_ms"] or 0))

    deltas = []
    verdict_changes = 0
    for rounds in by_claim_role.values():
        if 0 not in rounds or 1 not in rounds:
            continue
        r0 = rounds[0]
        r1 = rounds[1]
        deltas.append(float(r1["confidence_internal"]) - float(r0["confidence_internal"]))
        verdict_changes += int(r0["verdict"] != r1["verdict"])

    def rate(items: list[dict], pred) -> float:
        return (sum(1 for item in items if pred(item)) / len(items)) if items else 0.0

    return {
        "claims_with_judge": len({r["claim_id"] for r in rows}),
        "agent_r1_outputs": len(r1_rows),
        "valid_json_rate": json_valid / len(r1_rows) if r1_rows else 0.0,
        "mean_confidence": mean([float(r["confidence_internal"]) for r in r1_rows]) if r1_rows else 0.0,
        "mean_confidence_v0": mean(confidence_by_label[0.0]) if confidence_by_label[0.0] else 0.0,
        "mean_confidence_v05": mean(confidence_by_label[0.5]) if confidence_by_label[0.5] else 0.0,
        "mean_confidence_v1": mean(confidence_by_label[1.0]) if confidence_by_label[1.0] else 0.0,
        "overconf_unsupported_rate": rate(
            unsupported_r1,
            lambda r: float(r["confidence_internal"]) >= 0.7,
        ),
        "overconf_partial_rate": rate(
            partial_r1,
            lambda r: float(r["confidence_internal"]) >= 0.8,
        ),
        "verdict_exact_match_vs_judge": exact / len(r1_rows) if r1_rows else 0.0,
        "verdict_mae_vs_judge": mean(maes) if maes else 0.0,
        "r0_to_r1_mean_conf_delta": mean(deltas) if deltas else 0.0,
        "r0_to_r1_verdict_change_rate": verdict_changes / len(deltas) if deltas else 0.0,
        "mean_latency_ms": mean(latencies) if latencies else 0.0,
        "judge_label_distribution": dict(Counter(float(r["v_label"]) for r in r1_rows)),
        "r1_verdict_distribution": dict(verdict_distribution),
    }


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _print_comparison(base: dict, grpo: dict) -> None:
    metrics = [
        ("claims_with_judge", "Claims judged", "{:.0f}"),
        ("agent_r1_outputs", "Agent R1 outputs", "{:.0f}"),
        ("valid_json_rate", "Valid JSON rate", None),
        ("mean_confidence", "Mean confidence", "{:.3f}"),
        ("mean_confidence_v0", "Mean confidence when judge=0.0", "{:.3f}"),
        ("mean_confidence_v05", "Mean confidence when judge=0.5", "{:.3f}"),
        ("mean_confidence_v1", "Mean confidence when judge=1.0", "{:.3f}"),
        ("overconf_unsupported_rate", "Overconf unsupported rate", None),
        ("overconf_partial_rate", "Overconf partial rate", None),
        ("verdict_exact_match_vs_judge", "Verdict exact match vs judge", None),
        ("verdict_mae_vs_judge", "Verdict MAE vs judge", "{:.3f}"),
        ("r0_to_r1_mean_conf_delta", "R0->R1 mean confidence delta", "{:.3f}"),
        ("r0_to_r1_verdict_change_rate", "R0->R1 verdict change rate", None),
        ("mean_latency_ms", "Mean R1 latency ms", "{:.0f}"),
    ]

    print("\n" + "=" * 78)
    print("MAD BASE AGENTS vs GRPO-LORA AGENTS")
    print("=" * 78)
    print(f"{'Metric':42} {'Base':>15} {'GRPO':>15}")
    print("-" * 78)
    for key, label, fmt in metrics:
        b = base.get(key, 0)
        g = grpo.get(key, 0)
        if fmt is None:
            b_text = _pct(float(b))
            g_text = _pct(float(g))
        else:
            b_text = fmt.format(float(b))
            g_text = fmt.format(float(g))
        print(f"{label:42} {b_text:>15} {g_text:>15}")

    print("-" * 78)
    print("Base judge label distribution:", base.get("judge_label_distribution", {}))
    print("GRPO judge label distribution:", grpo.get("judge_label_distribution", {}))
    print("Base R1 verdict distribution:", base.get("r1_verdict_distribution", {}))
    print("GRPO R1 verdict distribution:", grpo.get("r1_verdict_distribution", {}))
    print("=" * 78 + "\n")


def _write_csv(path: Path, base: dict, grpo: dict) -> None:
    keys = sorted(set(base) | set(grpo))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "base", "grpo"])
        for key in keys:
            writer.writerow([key, base.get(key), grpo.get(key)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base MAD DB against GRPO LoRA MAD DB.")
    parser.add_argument("--base-db", default="mad_v4.db", type=Path)
    parser.add_argument("--grpo-db", default="mad_v4_grpo.db", type=Path)
    parser.add_argument("--out-csv", default="mad_grpo_comparison.csv", type=Path)
    args = parser.parse_args()

    with sqlite3.connect(args.base_db) as base_conn:
        base_summary = _summarize(_rows(base_conn))
    with sqlite3.connect(args.grpo_db) as grpo_conn:
        grpo_summary = _summarize(_rows(grpo_conn))

    _print_comparison(base_summary, grpo_summary)
    _write_csv(args.out_csv, base_summary, grpo_summary)
    print(f"Wrote comparison CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
