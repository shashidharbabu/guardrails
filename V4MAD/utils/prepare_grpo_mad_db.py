"""
Prepare a GRPO-agent MAD database from an existing V4MAD run.

Copies the baseline/decomposer/RAG state from mad_v4.db into mad_v4_grpo.db,
then clears agent debate outputs, judge verdicts, and LLM cache so Step 4 and
Step 5 rerun only the debate and judge stages with the LoRA agents.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


def prepare_db(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source DB not found: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    with sqlite3.connect(target) as conn:
        conn.executescript(
            """
            DELETE FROM agent_outputs;
            DELETE FROM judge_verdicts;
            DELETE FROM llm_cache;
            VACUUM;
            """
        )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mad_v4_grpo.db for GRPO LoRA MAD rerun.")
    parser.add_argument("--source", default="mad_v4.db", type=Path)
    parser.add_argument("--target", default="mad_v4_grpo.db", type=Path)
    args = parser.parse_args()

    prepare_db(args.source, args.target)
    print(f"Prepared {args.target} from {args.source}")
    print("Cleared tables: agent_outputs, judge_verdicts, llm_cache")


if __name__ == "__main__":
    main()
