"""
LangGraph node: Per-claim chunk assignment (no LLM).

Reranks the top-5 query-level stored chunks against each specific claim text
using TF-IDF cosine similarity, then assigns:
  Agent A: ranked positions [0, 1, 2]  — highest relevance, supporting
  Agent B: ranked positions [0, 3, 4]  — anchor + alternatives (edge cases)
  Judge:   all 5 chunks                — complete evidence pool

No GPU needed. Runs on CPU in seconds for all claims across all queries.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db_conn, insert_claim_chunks
from rag_rerank import assign_chunks
from schemas import MADState


async def claim_rag_node(state: MADState) -> dict:
    claims       = state.get("claims", [])
    query_chunks = state.get("query_chunks", [])
    user_query   = state["user_query"]
    conn         = get_db_conn()

    claim_chunks: dict[str, dict] = {}

    for claim in claims:
        claim_id   = claim["claim_id"]
        claim_text = claim["claim_text"]

        assignment = assign_chunks(claim_text, user_query, query_chunks)
        claim_chunks[claim_id] = assignment
        insert_claim_chunks(conn, claim_id, assignment)

    print(f"  [claim_rag] {state['query_id']}: "
          f"assigned chunks for {len(claims)} claims "
          f"(A:{len(next(iter(claim_chunks.values()), {}).get('agent_a', []))} "
          f"B:{len(next(iter(claim_chunks.values()), {}).get('agent_b', []))} "
          f"J:{len(next(iter(claim_chunks.values()), {}).get('judge', []))})")

    return {"claim_chunks": claim_chunks}
