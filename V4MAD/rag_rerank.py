"""
Per-claim chunk assignment.

Takes the top-5 query-level chunks (stored from original retrieval) and
re-ranks them against the specific claim text using TF-IDF cosine similarity.

Agent A gets ranks [0,1,2]  — highest relevance, supporting evidence
Agent B gets ranks [0,3,4]  — anchor chunk + lower-ranked alternatives (edge cases, exceptions)
Judge   gets ranks [0,1,2,3,4] — complete evidence pool

This runs on CPU (sklearn TF-IDF), no GPU needed. Fast for 5 candidates.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import AGENT_A_CHUNK_INDICES, AGENT_B_CHUNK_INDICES, JUDGE_CHUNK_INDICES


def _rerank(claim_text: str, user_query: str, chunks: list[dict]) -> list[int]:
    """
    Returns chunk indices sorted by relevance to the claim (descending).
    Uses TF-IDF cosine similarity — no GPU, instant for 5 candidates.
    """
    if len(chunks) <= 1:
        return list(range(len(chunks)))

    query = f"{user_query} — {claim_text}"
    texts = [query] + [c.get("text", "") for c in chunks]

    try:
        tfidf  = TfidfVectorizer(stop_words="english", min_df=1)
        matrix = tfidf.fit_transform(texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return list(np.argsort(scores)[::-1])
    except Exception:
        return list(range(len(chunks)))


def assign_chunks(claim_text: str, user_query: str, query_chunks: list[dict]) -> dict:
    """
    Assign per-claim evidence chunks to each agent and judge.

    Returns:
        {
          "agent_a": [chunk, chunk, chunk],   # ranked positions 0,1,2
          "agent_b": [chunk, chunk, chunk],   # ranked positions 0,3,4
          "judge":   [chunk, chunk, chunk, chunk, chunk],
        }
    """
    n = len(query_chunks)

    if n == 0:
        return {"agent_a": [], "agent_b": [], "judge": []}

    if n < 5:
        # Fewer than 5 chunks — give all to everyone (graceful degradation)
        return {"agent_a": query_chunks, "agent_b": query_chunks, "judge": query_chunks}

    ranked = _rerank(claim_text, user_query, query_chunks)

    def _pick(indices: list[int]) -> list[dict]:
        return [query_chunks[ranked[i]] for i in indices if i < len(ranked)]

    return {
        "agent_a": _pick(AGENT_A_CHUNK_INDICES),
        "agent_b": _pick(AGENT_B_CHUNK_INDICES),
        "judge":   _pick(JUDGE_CHUNK_INDICES),
    }
