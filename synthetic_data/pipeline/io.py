from __future__ import annotations

import os
from typing import Any, Iterable, List

import jsonlines


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with jsonlines.open(path) as r:
        return list(r)


def write_jsonl(path: str, items: Iterable[dict]) -> int:
    ensure_parent_dir(path)
    items_list = list(items)
    with jsonlines.open(path, "w") as w:
        w.write_all(items_list)
    return len(items_list)


def append_jsonl(path: str, items: Iterable[dict]) -> int:
    ensure_parent_dir(path)
    items_list = list(items)
    with jsonlines.open(path, "a") as w:
        w.write_all(items_list)
    return len(items_list)


def save_checkpoint(items: List[dict], base_dir: str, rel_path: str) -> str:
    path = os.path.join(base_dir, rel_path)
    write_jsonl(path, items)
    return path


def load_checkpoint(base_dir: str, rel_path: str) -> List[dict]:
    path = os.path.join(base_dir, rel_path)
    return load_jsonl(path)


def atomic_write_text(path: str, text: str) -> None:
    ensure_parent_dir(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

