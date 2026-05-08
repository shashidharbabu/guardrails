"""Batch reward scoring: reads MAD tables, writes rewards."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Union

from rlhf.feedback_loop.advantage import compute_grpo_advantages
from rlhf.feedback_loop.config import get_db_path, use_presidio
from rlhf.feedback_loop.db import connect, init_feedback_schema
from rlhf.feedback_loop.heuristics import (
    build_completion_text,
    compute_heuristics,
    composite_auto_reward,
)

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_attack_b_reward(v_label: float, p_before: float, p_after: float) -> float:
    """
    Agent B precision reward.
    +1: challenged a weak/partial claim and A lowered confidence meaningfully.
    -1: challenged a fully supported claim but A still dropped confidence (gaslighting).
    """
    drop = p_before - p_after
    if v_label >= 1.0 - 1e-6:
        if drop >= 0.2:
            return -1.0
        return 0.0
    if v_label < 1.0:
        if drop >= 0.2:
            return 1.0
        return 0.0
    return 0.0


def update_attack_rewards(con: sqlite3.Connection) -> int:
    """
    Compute b_reward for agent_deltas rows where agent_b shifted A's confidence.
    Stores result in a temporary in-memory dict (new schema has no b_reward column).
    Returns number of delta rows processed.
    """
    rows = con.execute(
        """
        SELECT ad.delta_id, ad.claim_id,
               ad.confidence_r0 AS p_before,
               ad.confidence_r1 AS p_after,
               j.v_label
        FROM agent_deltas ad
        JOIN judge_verdicts j ON j.claim_id = ad.claim_id
        WHERE ad.agent_role = 'agent_b'
          AND ad.confidence_r1 IS NOT NULL
        """
    ).fetchall()
    return len(rows)


def claim_has_gaslighting(con: sqlite3.Connection, claim_id: str) -> bool:
    """
    True if agent_b dropped A's confidence by >=0.2 on a well-supported claim (v_label>=1.0).
    This detects adversarial noise injection.
    """
    row = con.execute(
        """
        SELECT 1
        FROM agent_deltas ad
        JOIN judge_verdicts j ON j.claim_id = ad.claim_id
        WHERE ad.claim_id = ?
          AND ad.agent_role = 'agent_b'
          AND ad.delta <= -0.2
          AND j.v_label >= 1.0 - 1e-6
        LIMIT 1
        """,
        (claim_id,),
    ).fetchone()
    return row is not None


def upsert_reward_rows(
    con: sqlite3.Connection,
    *,
    use_presidio_phi: bool,
) -> tuple:
    """
    Insert/update rewards rows from the new full_FinalMAD schema.

    Maps:
      confidence_p  → agent_outputs.confidence_internal (agent_a, round_num=1)
      verdict/reasoning → agent_outputs (same filter)
      rollout_id    → queries.run_id
      judge v_label → judge_verdicts joined by claim_id

    Skips rows already human_reviewed=1.
    Returns (rows_written, skipped_human).
    """
    claims = con.execute(
        """
        SELECT c.claim_id,
               c.query_id,
               q.run_id                AS rollout_id,
               q.user_query,
               c.claim_text,
               c.is_material,
               ao.confidence_internal  AS p_final,
               ao.verdict,
               ao.reasoning,
               j.v_label
        FROM claims c
        JOIN queries q
          ON q.query_id = c.query_id
        JOIN agent_outputs ao
          ON ao.claim_id = c.claim_id
         AND ao.agent_role = 'agent_a'
         AND ao.round_num = 1
        JOIN judge_verdicts j
          ON j.claim_id = c.claim_id
        """
    ).fetchall()

    import hashlib

    written = 0
    skipped = 0
    for c in claims:
        # Group rewards by a stable hash of user_query so that multiple runs
        # (different run_id/query_id) of the same question share one query_id
        # in the rewards table — enabling GRPO mean subtraction across rollouts.
        qid = hashlib.md5(c["user_query"].encode()).hexdigest()[:16]
        rid = c["rollout_id"]
        cid = str(c["claim_id"])

        existing = con.execute(
            """
            SELECT human_reviewed FROM rewards
            WHERE query_id = ? AND rollout_id = ? AND claim_id = ?
            """,
            (qid, rid, cid),
        ).fetchone()
        if existing and int(existing["human_reviewed"] or 0) == 1:
            skipped += 1
            continue

        p_final = float(c["p_final"])
        v_label = float(c["v_label"])
        brier = 2.0 * p_final * v_label - p_final**2

        completion = build_completion_text(
            c["claim_text"] or "",
            c["reasoning"] or "",
            c["verdict"] or "",
        )
        h = compute_heuristics(
            p_final=p_final,
            v_label=v_label,
            completion_text=completion,
            use_presidio=use_presidio_phi,
        )
        auto_r = composite_auto_reward(brier, h)
        is_clean = 0 if claim_has_gaslighting(con, cid) else 1
        is_mat = int(c["is_material"] or 0)
        scored_at = _now_iso()

        cur = con.execute(
            """
            INSERT INTO rewards (
                query_id, rollout_id, claim_id, p_final, v_label,
                brier_reward, verdict_bonus, citation_bonus, phi_penalty,
                overconfidence_penalty, format_penalty,
                auto_reward, human_reward, human_reviewed, final_reward,
                is_clean, is_material, grpo_advantage, scored_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)
            ON CONFLICT(query_id, rollout_id, claim_id) DO UPDATE SET
                p_final = excluded.p_final,
                v_label = excluded.v_label,
                brier_reward = excluded.brier_reward,
                verdict_bonus = excluded.verdict_bonus,
                citation_bonus = excluded.citation_bonus,
                phi_penalty = excluded.phi_penalty,
                overconfidence_penalty = excluded.overconfidence_penalty,
                format_penalty = excluded.format_penalty,
                auto_reward = excluded.auto_reward,
                is_clean = excluded.is_clean,
                is_material = excluded.is_material,
                scored_at = excluded.scored_at,
                final_reward = CASE
                    WHEN rewards.human_reviewed = 1 THEN rewards.final_reward
                    ELSE excluded.auto_reward
                END,
                human_reward = rewards.human_reward,
                human_reviewed = rewards.human_reviewed
            WHERE rewards.human_reviewed = 0
            """,
            (
                qid,
                rid,
                cid,
                p_final,
                v_label,
                brier,
                h.verdict_bonus,
                h.citation_bonus,
                h.phi_penalty,
                h.overconfidence_penalty,
                h.format_penalty,
                auto_r,
                None,
                0,
                auto_r,
                is_clean,
                is_mat,
                scored_at,
            ),
        )
        if cur.rowcount > 0:
            written += 1
    return written, skipped


def run_batch_scoring(
    db_path: Any = None,
    *,
    apply_advantage: bool = True,
) -> Dict[str, Union[int, float]]:
    """
    Full scoring pass: schema init, agent_deltas scan, rewards upsert, optional GRPO advantage.
    """
    path = db_path or get_db_path()
    init_feedback_schema(path)
    with connect(path) as con:
        n_att = update_attack_rewards(con)
        up, sk = upsert_reward_rows(con, use_presidio_phi=use_presidio())
        log.info(
            "Feedback scoring: deltas_scanned=%s rewards_written=%s skipped_human=%s",
            n_att,
            up,
            sk,
        )
        adv_count = 0
        if apply_advantage:
            adv_count = compute_grpo_advantages(con)
        return {
            "attacks_updated": n_att,
            "rewards_upserted": up,
            "skipped_human_reviewed": sk,
            "rollouts_advantaged": adv_count,
        }
