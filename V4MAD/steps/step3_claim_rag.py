"""
Step 3: Per-claim live NewRAG chunk assignment.

For each decomposed claim, retrieves evidence with claim + original query via
NewRAG v2, then assigns the ranked top-5 chunks:
Assigns:
  Agent A → ranked positions [0,1,2]   (supporting, high-relevance)
  Agent B → ranked positions [0,2,3]   (anchor + alternate evidence)
  Judge   → all 5 chunks              (complete evidence pool)

Set MAD_USE_LIVE_RAG=0 to fall back to the old stored-query-chunk TF-IDF mode.

Run: python steps/step3_claim_rag.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph

from config import CHECKPOINT_DB, DB_PATH, QUERIES_FILE, QUERY_CHUNKS_FILE
from db import (get_claim_chunks, get_claims_for_query, get_db_conn,
                init_db, query_has_claim_chunks, query_has_claims)
from nodes.claim_rag_node import claim_rag_node
from schemas import MADState


def _build_graph(checkpointer):
    wf = StateGraph(MADState)
    wf.add_node("claim_rag", claim_rag_node)
    wf.add_edge(START, "claim_rag")
    wf.add_edge("claim_rag", END)
    return wf.compile(checkpointer=checkpointer)


async def main():
    init_db(str(DB_PATH))
    conn = get_db_conn()

    queries          = json.loads(QUERIES_FILE.read_text())
    query_chunks_map = {
        q["query_id"]: q.get("chunks", [])
        for q in json.loads(QUERY_CHUNKS_FILE.read_text())
    }

    pending = [
        q for q in queries
        if query_has_claims(conn, q["query_id"])
        and not query_has_claim_chunks(conn, q["query_id"])
    ]
    print(f"Step 3: {len(pending)} queries need chunk assignment")

    if not pending:
        print("All queries already have per-claim chunks. Nothing to do.")
        return

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as ckpt:
        graph = _build_graph(ckpt)
        for q in pending:
            qid    = q["query_id"]
            claims = get_claims_for_query(conn, qid)
            initial: MADState = {
                "query_id":       qid,
                "run_id":         "v4",
                "user_query":     q["user_query"],
                "baseline_answer": None,
                "claims":         claims,
                "query_chunks":   query_chunks_map.get(qid, []),
                "claim_chunks":   {},
                "agent_outputs":  {},
                "judge_verdicts": {},
                "errors":         [],
            }
            cfg = {"configurable": {"thread_id": f"s3_{qid}"}}
            try:
                await graph.ainvoke(initial, config=cfg)
            except Exception as e:
                print(f"  [ERROR] {qid}: {e}")

    done = sum(1 for q in queries if query_has_claim_chunks(conn, q["query_id"]))
    print(f"\nStep 3 complete: {done}/{len(queries)} queries have per-claim chunks.")


if __name__ == "__main__":
    asyncio.run(main())
