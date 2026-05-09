#!/usr/bin/env python3
"""
Create query-level train/validation/test splits for V4MAD GRPO JSONL.

Splitting by query_id avoids leakage because rows from the same query share
retrieved context and related claims. The splitter uses a small greedy pass to
keep split sizes and v_label distributions close to the requested targets.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


LABELS = (0.0, 0.5, 1.0)


def _label_counts(rows: list[dict]) -> Counter:
    return Counter(float(r["v_label"]) for r in rows)


def _score(candidate_counts: Counter, target_counts: dict[float, float]) -> float:
    return sum(abs(candidate_counts.get(label, 0) - target_counts[label]) for label in LABELS)


def _split_score(
    splits: dict[str, list[str]],
    query_rows: dict[str, list[dict]],
    target_query_counts: dict[str, int],
    target_label_props: dict[float, float],
    target_row_props: dict[str, float],
) -> float:
    score = 0.0
    total_rows = sum(len(rows) for rows in query_rows.values())
    for name, query_ids in splits.items():
        rows = [row for qid in query_ids for row in query_rows[qid]]
        counts = _label_counts(rows)
        n_rows = max(1, len(rows))
        score += abs(len(query_ids) - target_query_counts[name]) * 0.05
        score += abs((len(rows) / total_rows) - target_row_props[name]) * 2.0
        for label in LABELS:
            score += abs((counts[label] / n_rows) - target_label_props[label])
    return score


def _assign_queries(
    query_rows: dict[str, list[dict]],
    seed: int,
    train_frac: float,
    val_frac: float,
    search_iters: int,
) -> dict[str, str]:
    query_ids = list(query_rows)
    rng = random.Random(seed)

    total_queries = len(query_ids)
    train_target_n = round(total_queries * train_frac)
    val_target_n = round(total_queries * val_frac)
    test_target_n = total_queries - train_target_n - val_target_n

    all_rows = [r for rows in query_rows.values() for r in rows]
    total_label_counts = _label_counts(all_rows)
    target_label_props = {
        label: total_label_counts[label] / len(all_rows)
        for label in LABELS
    }
    target_row_props = {
        "train": train_frac,
        "val": val_frac,
        "test": 1.0 - train_frac - val_frac,
    }
    target_query_counts = {"train": train_target_n, "val": val_target_n, "test": test_target_n}

    best_splits = None
    best_score = float("inf")
    for _ in range(search_iters):
        shuffled = query_ids[:]
        rng.shuffle(shuffled)
        splits = {
            "train": shuffled[:train_target_n],
            "val": shuffled[train_target_n : train_target_n + val_target_n],
            "test": shuffled[train_target_n + val_target_n :],
        }
        score = _split_score(
            splits,
            query_rows,
            target_query_counts,
            target_label_props,
            target_row_props,
        )
        if score < best_score:
            best_score = score
            best_splits = splits

    if best_splits is None:
        raise RuntimeError("failed to create query split")

    assignment: dict[str, str] = {}
    for split, split_query_ids in best_splits.items():
        for qid in split_query_ids:
            assignment[qid] = split
    return assignment


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split V4MAD GRPO JSONL by query_id")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/vineethrayadurgam/Downloads/V4MAD/data/grpo_combined.jsonl"),
        help="Input combined V4MAD GRPO JSONL",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory for grpo_train.jsonl, grpo_val.jsonl, grpo_test.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--search-iters", type=int, default=20000)
    args = parser.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    query_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        query_id = row.get("query_id")
        if not query_id:
            raise ValueError("Every row must have query_id for leakage-safe splitting")
        query_rows[str(query_id)].append(row)

    assignment = _assign_queries(
        query_rows,
        args.seed,
        args.train_frac,
        args.val_frac,
        args.search_iters,
    )
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        splits[assignment[str(row["query_id"])]].append(row)

    _write_jsonl(args.out_dir / "grpo_train.jsonl", splits["train"])
    _write_jsonl(args.out_dir / "grpo_val.jsonl", splits["val"])
    _write_jsonl(args.out_dir / "grpo_test.jsonl", splits["test"])

    print(f"input: {args.input}")
    for name in ("train", "val", "test"):
        split_rows = splits[name]
        split_queries = {r["query_id"] for r in split_rows}
        counts = _label_counts(split_rows)
        print(
            f"{name}: rows={len(split_rows)} queries={len(split_queries)} "
            f"NOT_SUPPORTED={counts[0.0]} PARTIAL={counts[0.5]} SUPPORTED={counts[1.0]}"
        )


if __name__ == "__main__":
    main()
