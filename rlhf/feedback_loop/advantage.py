"""GRPO group-relative advantage from per-rollout totals (same query_id)."""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def compute_grpo_advantages(con: Any) -> int:
    """
    For each query_id, compare rollout totals (sum of final_reward for clean material claims).
    grpo_advantage for every row in that rollout = rollout_total - mean(rollout_total).
    Single rollout → advantage 0 for all its rows.
    Returns number of (query_id, rollout_id) groups updated.
    """
    qids = [r[0] for r in con.execute("SELECT DISTINCT query_id FROM rewards").fetchall()]
    groups = 0
    for qid in qids:
        r_rows = con.execute(
            "SELECT DISTINCT rollout_id FROM rewards WHERE query_id = ?",
            (qid,),
        ).fetchall()
        rids = [r[0] for r in r_rows]
        if not rids:
            continue
        totals: list[tuple[str, float]] = []
        for rid in rids:
            row = con.execute(
                """
                SELECT COALESCE(SUM(final_reward), 0) AS s
                FROM rewards
                WHERE query_id = ? AND rollout_id = ?
                  AND is_clean = 1 AND is_material = 1
                  AND final_reward IS NOT NULL
                """,
                (qid, rid),
            ).fetchone()
            totals.append((rid, float(row["s"])))
        mean_t = sum(t for _, t in totals) / len(totals)
        for rid, tot in totals:
            adv = tot - mean_t
            con.execute(
                """
                UPDATE rewards SET grpo_advantage = ?
                WHERE query_id = ? AND rollout_id = ?
                """,
                (adv, qid, rid),
            )
            groups += 1
    log.info("GRPO advantage written for %s query-rollout groups", groups)
    return groups
