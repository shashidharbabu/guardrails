"""
Step 5: Judge verdicts.

Launches 14B (or 32B on HPC) vLLM server.
Judge sees ALL 5 chunks + both agents' R0 and R1 outputs (stripped, anonymized).
Renders v_label ∈ {0.0, 0.5, 1.0} per claim.

Change JUDGE_MODEL in config.py to Qwen2.5-32B-Instruct when running on H100.

Run: python steps/step5_judge.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph

from config import (CHECKPOINT_DB, DB_PATH, JUDGE_MODEL,
                    JUDGE_PORT, QUERIES_FILE, QUERY_CHUNKS_FILE)
from db import (get_agent_output, get_claim_chunks, get_claims_for_query,
                get_db_conn, init_db, query_has_debate)
from nodes.judge_node import judge_node
from schemas import MADState
from server_utils import launch_vllm_server, shutdown_server, wait_for_server


def _build_graph(checkpointer):
    wf = StateGraph(MADState)
    wf.add_node("judge", judge_node)
    wf.add_edge(START,   "judge")
    wf.add_edge("judge", END)
    return wf.compile(checkpointer=checkpointer)


def _load_agent_outputs(conn, claims: list[dict]) -> dict:
    """Reconstruct agent_outputs dict from DB for all claims."""
    outputs = {}
    for claim in claims:
        cid = claim["claim_id"]
        a0 = get_agent_output(conn, cid, "agent_a", 0)
        a1 = get_agent_output(conn, cid, "agent_a", 1)
        b0 = get_agent_output(conn, cid, "agent_b", 0)
        b1 = get_agent_output(conn, cid, "agent_b", 1)
        if a0 and a1 and b0 and b1:
            outputs[cid] = {
                "agent_a": {0: a0, 1: a1},
                "agent_b": {0: b0, 1: b1},
            }
    return outputs


async def main():
    init_db(str(DB_PATH))
    conn = get_db_conn()

    queries          = json.loads(QUERIES_FILE.read_text())
    query_chunks_map = {
        q["query_id"]: q.get("chunks", [])
        for q in json.loads(QUERY_CHUNKS_FILE.read_text())
    }

    pending = [q for q in queries if query_has_debate(conn, q["query_id"])]
    # Filter out already judged (check judge_verdicts table)
    already_judged = set(
        r[0] for r in conn.execute(
            "SELECT DISTINCT c.query_id FROM judge_verdicts jv "
            "JOIN claims c ON jv.claim_id=c.claim_id"
        ).fetchall()
    )
    pending = [q for q in pending if q["query_id"] not in already_judged]
    print(f"Step 5: {len(pending)} queries need judging")

    if not pending:
        print("All queries already judged. Nothing to do.")
        return

    proc = launch_vllm_server(
        JUDGE_MODEL, JUDGE_PORT,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )
    if not wait_for_server(JUDGE_PORT):
        shutdown_server(proc)
        return

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as ckpt:
            graph = _build_graph(ckpt)
            for q in pending:
                qid    = q["query_id"]
                claims = get_claims_for_query(conn, qid)

                claim_chunks = {}
                for claim in claims:
                    cc = get_claim_chunks(conn, claim["claim_id"])
                    if cc:
                        claim_chunks[claim["claim_id"]] = cc

                agent_outputs = _load_agent_outputs(conn, claims)

                initial: MADState = {
                    "query_id":       qid,
                    "run_id":         "v4",
                    "user_query":     q["user_query"],
                    "baseline_answer": None,
                    "claims":         claims,
                    "query_chunks":   query_chunks_map.get(qid, []),
                    "claim_chunks":   claim_chunks,
                    "agent_outputs":  agent_outputs,
                    "judge_verdicts": {},
                    "errors":         [],
                }
                cfg = {"configurable": {"thread_id": f"s5_{qid}"}}
                print(f"\n[{qid}] Judging {len(claims)} claims...")
                try:
                    await graph.ainvoke(initial, config=cfg)
                except Exception as e:
                    print(f"  [ERROR] {qid}: {e}")
    finally:
        shutdown_server(proc)

    total_verdicts = conn.execute("SELECT COUNT(*) FROM judge_verdicts").fetchone()[0]
    print(f"\nStep 5 complete: {total_verdicts} total verdicts in DB.")


if __name__ == "__main__":
    asyncio.run(main())
