"""CLI: run batch reward scoring + GRPO advantage on MAD SQLite."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from rlhf.feedback_loop.config import get_db_path
from rlhf.feedback_loop.scorer import run_batch_scoring

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score rewards from MAD tables")
    p.add_argument(
        "--db",
        default=None,
        help="SQLite path (default: MAD_DB_PATH env or ./mad_store.db)",
    )
    p.add_argument(
        "--no-advantage",
        action="store_true",
        help="Skip GRPO advantage normalization",
    )
    args = p.parse_args(argv)
    db = args.db or get_db_path()
    stats = run_batch_scoring(db, apply_advantage=not args.no_advantage)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
