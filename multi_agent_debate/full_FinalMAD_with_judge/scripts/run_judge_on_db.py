"""
Run judge on all agent outputs already stored in the database for a given run_id.

Usage:
  python scripts/run_judge_on_db.py --run-id my-run-01 --db mad.db
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from configs import config
from src.db.db import fetch_all, fetch_one
from src.judge.node import judge_node
from src.schemas.schemas import (
    AgentOutputFull,
    AgentRole,
    AgentVerdict,
    Claim,
    EvidenceCitation,
    PipelineState,
)


logger = logging.getLogger(__name__)


def _loads(value):
    return json.loads(value) if value else None


def _claim_from_row(row: dict) -> Claim:
    return Claim(
        claim_id=row["claim_id"],
        claim_text=row["claim_text"],
        claim_index=row["claim_index"],
        is_material=bool(row["is_material"]),
        is_critical=bool(row["is_critical"]),
        confidence_prior=row["confidence_prior"],
    )


def _agent_output_from_row(row: dict) -> AgentOutputFull:
    evidence = [EvidenceCitation(**item) for item in _loads(row["evidence_cited"])]
    return AgentOutputFull(
        agent_role=AgentRole(row["agent_role"]),
        round_num=row["round_num"],
        verdict=AgentVerdict(row["verdict"]),
        reasoning=row["reasoning"],
        evidence_cited=evidence,
        confidence_internal=row["confidence_internal"],
    )


def _load_state(query_row: dict, db_path: str) -> PipelineState:
    query_id = query_row["query_id"]
    claim_rows = fetch_all(
        "SELECT * FROM claims WHERE query_id = ? ORDER BY claim_index",
        (query_id,),
        db_path,
    )
    claims = [_claim_from_row(row) for row in claim_rows]

    outputs_by_claim: dict = {}
    for claim in claims:
        output_rows = fetch_all(
            "SELECT * FROM agent_outputs WHERE claim_id = ? ORDER BY round_num, agent_role",
            (claim.claim_id,),
            db_path,
        )
        outputs_by_claim[claim.claim_id] = {"agent_a": {}, "agent_b": {}}
        for row in output_rows:
            output = _agent_output_from_row(row)
            outputs_by_claim[claim.claim_id][row["agent_role"]][row["round_num"]] = output

    return PipelineState(
        query_id=query_id,
        run_id=query_row["run_id"],
        user_query=query_row["user_query"],
        rag_chunks=_loads(query_row["rag_chunks"]) or [],
        baseline_answer=query_row["baseline_answer"],
        claims=claims,
        agent_outputs_by_claim=outputs_by_claim,
    )


def _verdict_count(run_id: str, db_path: str) -> int:
    row = fetch_one(
        """
        SELECT COUNT(*) AS count FROM judge_verdicts j
        JOIN claims c ON c.claim_id = j.claim_id
        JOIN queries q ON q.query_id = c.query_id
        WHERE q.run_id = ?
        """,
        (run_id,),
        db_path,
    )
    return int(row["count"]) if row else 0


def _verdict_distribution(run_id: str, db_path: str) -> Counter:
    rows = fetch_all(
        """
        SELECT j.v_label, COUNT(*) AS count
        FROM judge_verdicts j
        JOIN claims c ON c.claim_id = j.claim_id
        JOIN queries q ON q.query_id = c.query_id
        WHERE q.run_id = ?
        GROUP BY j.v_label ORDER BY j.v_label
        """,
        (run_id,),
        db_path,
    )
    return Counter({str(row["v_label"]): row["count"] for row in rows})


async def _run(run_id: str, db_path: str) -> None:
    query_rows = fetch_all(
        "SELECT * FROM queries WHERE run_id = ? ORDER BY timestamp, query_id",
        (run_id,),
        db_path,
    )
    logger.info("Found %d queries for run_id=%s", len(query_rows), run_id)
    before = _verdict_count(run_id, db_path)

    for query_row in query_rows:
        try:
            state = _load_state(query_row, db_path)
            await judge_node(state)
            logger.info("Judged query %s", state.query_id)
        except Exception:
            logger.exception("Failed judging query %s", query_row["query_id"])

    after = _verdict_count(run_id, db_path)
    distribution = _verdict_distribution(run_id, db_path)
    summary = {
        "run_id": run_id,
        "verdicts_written": after - before,
        "total_verdicts": after,
        "distribution": dict(distribution),
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run judge on existing DB debate outputs")
    parser.add_argument("--run-id", required=True, help="Run ID to judge")
    parser.add_argument("--db", default=None, help="SQLite DB path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db_path = args.db or config.SQLITE_DB_PATH
    asyncio.run(_run(args.run_id, db_path))


if __name__ == "__main__":
    main()
