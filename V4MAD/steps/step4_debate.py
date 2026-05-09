"""
Step 4: Agent debate — Round 0 then Round 1.

Launches ONE shared 14B vLLM server.
By default both agents call the same base model with different prompts + temperatures.
For GRPO validation, set USE_GRPO_LORA_AGENTS=1 so the same vLLM server
registers Agent A/B LoRA adapters once at startup and the debate nodes call
model="agent_a" and model="agent_b".

Agent A: AGENT_A_SYSTEM, temp=0.4, chunks [0,1,2]  — strict verifier
Agent B: AGENT_B_SYSTEM, temp=0.85, chunks [0,2,3] — inverted-burden skeptic

LangGraph state: R0 outputs stored in state before R1 node reads them.
LangGraph checkpoint: if Colab crashes mid-query, resume from last checkpoint.

Run: python steps/step4_debate.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph

from config import (AGENT_A_LORA_PATH, AGENT_A_MODEL, AGENT_B_LORA_PATH,
                    AGENT_B_MODEL, AGENTS_MODEL, AGENTS_PORT, CHECKPOINT_DB,
                    DB_PATH, QUERIES_FILE, QUERY_CHUNKS_FILE,
                    USE_GRPO_LORA_AGENTS)
from db import (get_claim_chunks, get_claims_for_query, get_db_conn,
                init_db, query_has_claim_chunks, query_has_debate)
from nodes.debate_r0_node import debate_r0_node
from nodes.debate_r1_node import debate_r1_node
from schemas import MADState
from server_utils import launch_vllm_server, shutdown_server, wait_for_server


def _build_graph(checkpointer):
    wf = StateGraph(MADState)
    wf.add_node("debate_r0", debate_r0_node)
    wf.add_node("debate_r1", debate_r1_node)
    wf.add_edge(START,      "debate_r0")
    wf.add_edge("debate_r0", "debate_r1")
    wf.add_edge("debate_r1", END)
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
        if query_has_claim_chunks(conn, q["query_id"])
        and not query_has_debate(conn, q["query_id"])
    ]
    print(f"Step 4: {len(pending)} queries need debate (R0 + R1)")

    if not pending:
        print("All queries already debated. Nothing to do.")
        return

    lora_modules = None
    if USE_GRPO_LORA_AGENTS:
        lora_modules = {
            AGENT_A_MODEL: str(AGENT_A_LORA_PATH),
            AGENT_B_MODEL: str(AGENT_B_LORA_PATH),
        }
        print("[GRPO LoRA] Registering adapters once at vLLM startup:")
        for name, path in lora_modules.items():
            print(f"  {name} -> {path}")

    proc = launch_vllm_server(
        AGENTS_MODEL, AGENTS_PORT,
        gpu_memory_utilization=0.88,
        max_model_len=4096,
        enable_prefix_caching=True,
        lora_modules=lora_modules,
        max_lora_rank=32 if lora_modules else None,
    )
    if not wait_for_server(AGENTS_PORT):
        shutdown_server(proc)
        return

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as ckpt:
            graph = _build_graph(ckpt)
            for q in pending:
                qid    = q["query_id"]
                claims = get_claims_for_query(conn, qid)

                # Load claim chunks from DB into state
                claim_chunks = {}
                for claim in claims:
                    cc = get_claim_chunks(conn, claim["claim_id"])
                    if cc:
                        claim_chunks[claim["claim_id"]] = cc

                initial: MADState = {
                    "query_id":       qid,
                    "run_id":         "v4",
                    "user_query":     q["user_query"],
                    "baseline_answer": None,
                    "claims":         claims,
                    "query_chunks":   query_chunks_map.get(qid, []),
                    "claim_chunks":   claim_chunks,
                    "agent_outputs":  {},
                    "judge_verdicts": {},
                    "errors":         [],
                }
                cfg = {"configurable": {"thread_id": f"s4_{qid}"}}
                print(f"\n[{qid}] Debating {len(claims)} claims...")
                try:
                    await graph.ainvoke(initial, config=cfg)
                except Exception as e:
                    print(f"  [ERROR] {qid}: {e}")
    finally:
        shutdown_server(proc)

    done = sum(1 for q in queries if query_has_debate(conn, q["query_id"]))
    print(f"\nStep 4 complete: {done}/{len(queries)} queries fully debated.")


if __name__ == "__main__":
    asyncio.run(main())
