"""
Step 1: Baseline LLM answers (NO RAG).

Launches 3B vLLM server → generates answers from parametric memory → saves to DB.
Already-processed queries are skipped (idempotent — safe to re-run).

Run: python steps/step1_baseline.py
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph

from config import (BASELINE_MODEL, BASELINE_PORT, CHECKPOINT_DB,
                    DB_PATH, QUERIES_FILE, QUERY_CHUNKS_FILE)
from db import get_db_conn, init_db, query_has_baseline
from nodes.baseline_node import baseline_node
from schemas import MADState
from server_utils import launch_vllm_server, shutdown_server, wait_for_server


def _build_graph(checkpointer):
    wf = StateGraph(MADState)
    wf.add_node("baseline", baseline_node)
    wf.add_edge(START, "baseline")
    wf.add_edge("baseline", END)
    return wf.compile(checkpointer=checkpointer)


async def main():
    run_id = f"v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    init_db(str(DB_PATH))
    conn = get_db_conn()

    if not QUERIES_FILE.exists():
        print(f"[ERROR] {QUERIES_FILE} not found. Run utils/extract_query_chunks.py first.")
        return
    if not QUERY_CHUNKS_FILE.exists():
        print(f"[ERROR] {QUERY_CHUNKS_FILE} not found. Run utils/extract_query_chunks.py first.")
        return

    queries           = json.loads(QUERIES_FILE.read_text())
    query_chunks_map  = {
        q["query_id"]: q.get("chunks", [])
        for q in json.loads(QUERY_CHUNKS_FILE.read_text())
    }

    pending = [q for q in queries if not query_has_baseline(conn, q["query_id"])]
    print(f"Step 1: {len(queries)} queries total, {len(pending)} need baseline")

    if not pending:
        print("All queries already have baseline answers. Nothing to do.")
        return

    proc = launch_vllm_server(BASELINE_MODEL, BASELINE_PORT)
    if not wait_for_server(BASELINE_PORT):
        shutdown_server(proc)
        return

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as ckpt:
            graph = _build_graph(ckpt)
            for q in pending:
                qid     = q["query_id"]
                initial: MADState = {
                    "query_id":       qid,
                    "run_id":         run_id,
                    "user_query":     q["user_query"],
                    "baseline_answer": None,
                    "claims":         [],
                    "query_chunks":   query_chunks_map.get(qid, []),
                    "claim_chunks":   {},
                    "agent_outputs":  {},
                    "judge_verdicts": {},
                    "errors":         [],
                }
                cfg = {"configurable": {"thread_id": f"s1_{qid}"}}
                try:
                    await graph.ainvoke(initial, config=cfg)
                except Exception as e:
                    print(f"  [ERROR] {qid}: {e}")
    finally:
        shutdown_server(proc)

    done = sum(1 for q in queries if query_has_baseline(conn, q["query_id"]))
    print(f"\nStep 1 complete: {done}/{len(queries)} queries have baseline answers.")


if __name__ == "__main__":
    asyncio.run(main())
