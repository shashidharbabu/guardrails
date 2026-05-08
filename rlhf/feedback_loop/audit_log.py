"""Append-only JSONL audit log for human review decisions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_audit(record: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_audit_tail(path: str, n: int = 50) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
