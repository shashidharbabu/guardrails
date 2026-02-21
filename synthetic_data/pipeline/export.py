from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, List, Tuple

import jsonlines


def stratified_split(
    data: List[dict],
    *,
    key_fn: Callable[[dict], str],
    train: float,
    val: float,
    test: float,
    seed: int = 123,
) -> Tuple[List[dict], List[dict], List[dict]]:
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("train+val+test must equal 1.0")

    rng = random.Random(seed)
    by_class: Dict[str, List[dict]] = {}
    for item in data:
        k = key_fn(item) or "unknown"
        by_class.setdefault(k, []).append(item)

    train_set: List[dict] = []
    val_set: List[dict] = []
    test_set: List[dict] = []

    for _, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train)
        n_val = int(n * val)
        train_set.extend(items[:n_train])
        val_set.extend(items[n_train : n_train + n_val])
        test_set.extend(items[n_train + n_val :])

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)
    return train_set, val_set, test_set


def export_dataset(
    data: List[dict],
    *,
    exports_dir: str,
    name: str,
    key_fn: Callable[[dict], str],
    train: float,
    val: float,
    test: float,
    seed: int = 123,
) -> Dict[str, str]:
    train_set, val_set, test_set = stratified_split(
        data,
        key_fn=key_fn,
        train=train,
        val=val,
        test=test,
        seed=seed,
    )

    os.makedirs(exports_dir, exist_ok=True)
    out_paths: Dict[str, str] = {}
    for split_name, split_data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        path = f"{exports_dir}/{name}_{split_name}.jsonl"
        with jsonlines.open(path, "w") as w:
            w.write_all(split_data)
        out_paths[split_name] = path
    return out_paths

