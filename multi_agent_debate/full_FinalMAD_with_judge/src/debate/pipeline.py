"""
Full MAD debate pipeline: decomposer -> round0 -> round1 -> judge (optional).

Entry point for programmatic use. Scripts use this module.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from configs import config
from src.agents.round0 import debate_round0_node
from src.agents.round1 import debate_round1_node
from src.db.db import connect, init_db, insert_claim, insert_query
from src.decomposer.node import decompose_node
from src.schemas.schemas import Claim, PipelineState


logger = logging.getLogger(__name__)


def _claim_from_dict(item: dict) -> Claim:
    return Claim(
        claim_id=item["claim_id"],
        claim_text=item["claim_text"],
        claim_index=item["claim_index"],
        is_material=item["is_material"],
        is_critical=item["is_critical"],
        confidence_prior=item["confidence_prior"],
    )


def _init_db_from_data(data: list[dict], db_path: str, run_id: str) -> None:
    init_db(db_path)
    for query in data:
        insert_query(
            query_id=query["query_id"],
            run_id=run_id,
            user_query=query["user_query"],
            rag_chunk_ids=query.get("rag_chunk_ids")
                or [c.get("chunk_id", "") for c in query.get("rag_chunks", [])],
            rag_chunks=query["rag_chunks"],
            baseline_answer=query.get("baseline_answer", ""),
            baseline_model=query.get("baseline_model", "unknown"),
            db_path=db_path,
        )
        with connect(db_path) as conn:
            for raw_claim in query.get("claims", []):
                insert_claim(
                    conn,
                    _claim_from_dict(raw_claim),
                    query["query_id"],
                    bool(raw_claim.get("coverage_check", True)),
                    float(raw_claim.get("coverage_ratio", 1.0)),
                )


async def run_debate_for_query(
    query: dict,
    *,
    run_id: str,
    db_path: str,
    run_judge: bool = False,
    langfuse=None,
) -> dict:
    """
    Run debate for a single query dict.

    query must have: query_id, user_query, rag_chunks, claims (list of claim dicts).
    If baseline_answer is missing, decompose_node will not run (claims must be pre-built).
    """
    claims = [_claim_from_dict(c) for c in query.get("claims", [])]
    state = PipelineState(
        query_id=query["query_id"],
        run_id=run_id,
        user_query=query["user_query"],
        rag_chunks=query["rag_chunks"],
        baseline_answer=query.get("baseline_answer"),
        claims=claims,
    )

    if not claims and state.baseline_answer:
        state = await decompose_node(state)

    state = await debate_round0_node(state)
    state = await debate_round1_node(state)

    if run_judge:
        from src.judge.node import judge_node
        state = await judge_node(state, langfuse=langfuse)

    return {
        "query_id": state.query_id,
        "claims_count": len(state.claims),
        "agent_outputs_count": sum(
            len(v.get("agent_a", {})) + len(v.get("agent_b", {}))
            for v in state.agent_outputs_by_claim.values()
        ),
        "judge_verdicts_count": len(state.judge_verdicts_by_claim),
    }


async def run_debate_from_file(
    claims_path: str | Path,
    *,
    run_id: str | None = None,
    db_path: str | None = None,
    run_judge: bool = False,
) -> dict:
    """
    Run full debate from a JSON file containing queries with pre-built claims.
    """
    if run_id is None:
        run_id = f"run-{uuid4().hex[:8]}"
    if db_path is None:
        db_path = config.SQLITE_DB_PATH

    data = json.loads(Path(claims_path).read_text())
    _init_db_from_data(data, db_path, run_id)

    from src.langfuse_log.logger import get_langfuse_client
    langfuse = get_langfuse_client()

    semaphore = asyncio.Semaphore(config.CLAIM_CONCURRENCY)
    results = []
    start = time.perf_counter()

    async def run_one(query: dict) -> None:
        async with semaphore:
            result = await run_debate_for_query(
                query, run_id=run_id, db_path=db_path,
                run_judge=run_judge, langfuse=langfuse,
            )
            results.append(result)
            logger.info("Completed query %s", query["query_id"])

    await asyncio.gather(*(run_one(q) for q in data))

    if langfuse:
        try:
            langfuse.flush()
        except Exception:
            pass

    return {
        "run_id": run_id,
        "db_path": db_path,
        "queries_processed": len(results),
        "total_claims": sum(r["claims_count"] for r in results),
        "total_agent_outputs": sum(r["agent_outputs_count"] for r in results),
        "runtime_s": round(time.perf_counter() - start, 3),
    }
