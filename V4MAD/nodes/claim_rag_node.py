"""
LangGraph node: Per-claim NewRAG evidence retrieval.

For every decomposed claim, live NewRAG v2 retrieves with:
    claim + original_query

NewRAG performs Qdrant Cloud dense retrieval + local BM25 + RRF + BGE rerank,
then this node assigns:
  Agent A: ranked chunks [1, 2, 3]
  Agent B: ranked chunks [1, 3, 4]
  Judge:   ranked chunks [1, 2, 3, 4, 5]
"""

import sys
from pathlib import Path

_V4MAD_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _V4MAD_ROOT.parent
sys.path.insert(0, str(_V4MAD_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from config import MAD_USE_LIVE_RAG
from db import get_db_conn, insert_claim_chunks
from rag_rerank import assign_chunks, assign_ranked_chunks
from schemas import MADState


async def claim_rag_node(state: MADState) -> dict:
    claims       = state.get("claims", [])
    query_chunks = state.get("query_chunks", [])
    user_query   = state["user_query"]
    conn         = get_db_conn()

    claim_chunks: dict[str, dict] = {}

    rag_service = None
    if MAD_USE_LIVE_RAG:
        from rag_v2.rag_service import get_service
        rag_service = get_service(verbose=False)

    for claim in claims:
        claim_id   = claim["claim_id"]
        claim_text = claim["claim_text"]

        if rag_service is not None:
            retrieved = rag_service.retrieve_for_cod(
                query=f"{claim_text}. Context: {user_query}",
                session_id=state["query_id"],
                agent_id="shared",
                round_num=0,
                k=5,
            )
            assignment = assign_ranked_chunks(retrieved["chunks"])
        else:
            assignment = assign_chunks(claim_text, user_query, query_chunks)

        claim_chunks[claim_id] = assignment
        insert_claim_chunks(conn, claim_id, assignment)

    print(f"  [claim_rag] {state['query_id']}: "
          f"assigned chunks for {len(claims)} claims "
          f"(A:{len(next(iter(claim_chunks.values()), {}).get('agent_a', []))} "
          f"B:{len(next(iter(claim_chunks.values()), {}).get('agent_b', []))} "
          f"J:{len(next(iter(claim_chunks.values()), {}).get('judge', []))})")

    return {"claim_chunks": claim_chunks}
