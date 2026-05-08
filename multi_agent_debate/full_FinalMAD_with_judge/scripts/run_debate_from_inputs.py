"""
Run the full MAD debate pipeline from a JSON file with pre-built claims.

Input JSON format:
[
  {
    "query_id": "q_001",
    "user_query": "What does GDPR require?",
    "baseline_answer": "GDPR requires data minimization...",
    "rag_chunks": [{"chunk_id": "c1", "text": "..."}],
    "claims": [
      {
        "claim_id": "uuid",
        "claim_text": "GDPR requires data minimization.",
        "claim_index": 0,
        "is_material": true,
        "is_critical": false,
        "confidence_prior": 0.75
      }
    ]
  }
]

Usage:
  python scripts/run_debate_from_inputs.py \
      --input examples/sample_input.json \
      --run-id my-run-01 \
      --db mad.db \
      --run-judge
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from configs import config
from src.debate.pipeline import run_debate_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAD debate from claims JSON")
    parser.add_argument("--input", required=True, help="Path to claims JSON file")
    parser.add_argument("--run-id", default=None, help="Run ID (auto-generated if omitted)")
    parser.add_argument("--db", default=None, help="SQLite DB path (default: SQLITE_DB_PATH env)")
    parser.add_argument("--run-judge", action="store_true", help="Also run the judge after debate")
    parser.add_argument("--concurrency", type=int, default=None, help="Override CLAIM_CONCURRENCY")
    args = parser.parse_args()

    if args.concurrency:
        config.CLAIM_CONCURRENCY = args.concurrency

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    db_path = args.db or config.SQLITE_DB_PATH
    summary = asyncio.run(
        run_debate_from_file(
            args.input,
            run_id=args.run_id,
            db_path=db_path,
            run_judge=args.run_judge,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
