"""
Step 2: Decompose baseline answers into atomic claims.

Launches 7B vLLM server → extracts claims from baseline answers → saves to DB.
Idempotent: queries that already have claims are skipped.

Run: python steps/step2_decompose.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph

from config import (CHECKPOINT_DB, DB_PATH, DECOMPOSER_MODEL,
                    DECOMPOSER_PORT, QUERIES_FILE, QUERY_CHUNKS_FILE)
from db import get_db_conn, init_db, query_has_baseline, query_has_claims
from nodes.decompose_node import decompose_node
from schemas import MADState
from server_utils import launch_vllm_server, shutdown_server, wait_for_server


def _build_graph(checkpointer):
    wf = StateGraph(MADState)
    wf.add_node("decompose", decompose_node)
    wf.add_edge(START, "decompose")
    wf.add_edge("decompose", END)
    return wf.compile(checkpointer=checkpointer)


async def main():
    init_db(str(DB_PATH))
    conn = get_db_conn()

    queries = json.loads(QUERIES_FILE.read_text())
    query_chunks_map = {
        q["query_id"]: q.get("chunks", [])
        for q in json.loads(QUERY_CHUNKS_FILE.read_text())
    }

    # Need baseline answer but no claims yet
    pending = [
        q for q in queries
        if query_has_baseline(conn, q["query_id"])
        and not query_has_claims(conn, q["query_id"])
    ]
    print(f"Step 2: {len(pending)} queries need decomposition")

    if not pending:
        print("All queries already decomposed. Nothing to do.")
        return

    # Load baseline answers from DB for pending queries
    rows = conn.execute(
        f"SELECT query_id, user_query, baseline_answer FROM queries "
        f"WHERE query_id IN ({','.join('?'*len(pending))})",
        [q["query_id"] for q in pending]
    ).fetchall()
    db_map = {r["query_id"]: dict(r) for r in rows}

    proc = launch_vllm_server(DECOMPOSER_MODEL, DECOMPOSER_PORT)
    if not wait_for_server(DECOMPOSER_PORT):
        shutdown_server(proc)
        return

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as ckpt:
            graph = _build_graph(ckpt)
            for q in pending:
                qid    = q["query_id"]
                db_row = db_map.get(qid, {})
                initial: MADState = {
                    "query_id":        qid,
                    "run_id":          db_row.get("run_id", "v4"),
                    "user_query":      db_row.get("user_query", q["user_query"]),
                    "baseline_answer": db_row.get("baseline_answer"),
                    "claims":          [],
                    "query_chunks":    query_chunks_map.get(qid, []),
                    "claim_chunks":    {},
                    "agent_outputs":   {},
                    "judge_verdicts":  {},
                    "errors":          [],
                }
                cfg = {"configurable": {"thread_id": f"s2_{qid}"}}
                try:
                    await graph.ainvoke(initial, config=cfg)
                except Exception as e:
                    print(f"  [ERROR] {qid}: {e}")
    finally:
        shutdown_server(proc)

    done = sum(1 for q in queries if query_has_claims(conn, q["query_id"]))
    print(f"\nStep 2 complete: {done}/{len(queries)} queries have claims.")


if __name__ == "__main__":
    asyncio.run(main())
