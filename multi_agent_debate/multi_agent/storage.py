"""
storage.py — SQLite storage layer for the MAD pipeline.
==========================================================

WHY THIS EXISTS
---------------
The MAD pipeline needs to store detailed data at every step so the
feedback loop (GRPO fine-tuning) can consume it later.

Your friend's spec defines exactly 4 tables that the MAD code must write.
The feedback loop (separate codebase) then reads these tables and writes
reward columns back. The MAD code NEVER touches reward columns.

THE 4 TABLES (MAD writes these)
--------------------------------

  1. queries        — one row per query when it enters the pipeline
  2. claims         — three rows per claim (post_step_A, post_cycle1, post_cycle2)
                      NEVER update existing rows — always INSERT new ones
                      Stores agent_a_prompt — the GRPO training input
  3. attacks        — one row per Agent B challenge per cycle
                      MAD writes p_before_attack at challenge time
                      MAD updates p_after_attack after Agent A revises
  4. judge_verdicts — one row per claim after Judge runs

THE REWARD TABLES (feedback loop writes these — NOT MAD)
---------------------------------------------------------
  attacks.b_reward      — written by feedback loop scoring script
  rewards table         — created and owned by feedback loop
                          contains: brier_reward, is_clean, grpo_advantage

THE AGENT_A_PROMPT COLUMN — WHY IT'S CRITICAL
----------------------------------------------
When TRL trains Agent A using GRPO, it needs:
  INPUT:  the exact prompt Agent A received at each checkpoint
  REWARD: the Brier score that resulted from Agent A's output

Without storing agent_a_prompt, the TRL GRPO trainer has no input to
train on. It cannot reconstruct what Agent A was responding to from the
confidence number alone. This column IS the training data for GRPO.

At each checkpoint, agent_a_prompt contains progressively more context:
  post_step_A:   system + query + RAG chunks + claim text
  post_cycle1:   above + Agent B's cycle 1 challenge
  post_cycle2:   above + Agent B's cycle 2 challenge

DATA FLOW (in order, per your friend's spec)
--------------------------------------------
1. Query enters  → INSERT queries
2. Agent A step A → INSERT claims (post_step_A)
3. Agent B cycle 1 → INSERT attacks (cycle=1, p_after_attack=NULL)
4. Agent A revise → INSERT claims (post_cycle1) + UPDATE attacks (p_after_attack)
5. Agent B cycle 2 → INSERT attacks (cycle=2, p_after_attack=NULL)
6. Agent A final  → INSERT claims (post_cycle2) + UPDATE attacks (p_after_attack)
7. Judge runs     → INSERT judge_verdicts
8. CSE runs (later) → UPDATE queries (final_cse_score, routing_decision)
9. Feedback loop  → UPDATE attacks.b_reward + INSERT rewards
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from multi_agent.config import DB_PATH
from multi_agent.models import Claim, Challenge, JudgeVerdict


# ── Connection helper ──────────────────────────────────────────────────────────

@contextmanager
def _conn():
    """Context manager: open connection, commit on exit, always close."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Schema creation ────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all 4 MAD tables if they don't exist.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.

    The rewards table is NOT created here — that belongs to the feedback loop.
    The b_reward column in attacks is left NULL by MAD — feedback loop fills it.
    The agent_b_prompt column in attacks is NULL for now — Phase 2 work.
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    with _conn() as con:
        # ── TABLE 1: queries ──────────────────────────────────────────────────
        # One row per query entering the MAD pipeline.
        # final_cse_score and routing_decision are NULL until CSE runs.
        con.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                query_id         TEXT NOT NULL,
                rollout_id       TEXT NOT NULL,
                query_text       TEXT NOT NULL,
                llm_answer       TEXT NOT NULL,
                rag_chunk_ids    TEXT,           -- JSON array of chunk_ids in evidence pool
                timestamp        TEXT NOT NULL,
                final_cse_score  REAL,           -- NULL: CSE writes this later
                routing_decision TEXT,           -- NULL: CSE writes this later
                cse_f_llm        REAL,           -- DeepEval FaithfulnessMetric score
                cse_h_llm        REAL,           -- DeepEval HallucinationMetric score
                cse_relevancy    REAL,           -- DeepEval ContextualRelevancyMetric score
                cse_judge_eval   REAL,           -- MAD judge aggregate (min-mean)
                cse_version      TEXT,           -- "v2.0" or "v0.1" (fallback)
                PRIMARY KEY (query_id, rollout_id)
            )
        """)

        # Add CSE columns to existing tables (idempotent — safe to run repeatedly)
        _add_column_if_missing(con, "queries", "cse_f_llm",      "REAL")
        _add_column_if_missing(con, "queries", "cse_h_llm",      "REAL")
        _add_column_if_missing(con, "queries", "cse_relevancy",  "REAL")
        _add_column_if_missing(con, "queries", "cse_judge_eval", "REAL")
        _add_column_if_missing(con, "queries", "cse_version",    "TEXT")

        # ── TABLE 2: claims ───────────────────────────────────────────────────
        # Three rows per claim: post_step_A, post_cycle1, post_cycle2.
        # NEVER UPDATE existing rows — always INSERT new ones at each checkpoint.
        # This preserves the full confidence trajectory for GRPO training.
        #
        # agent_a_prompt is the GRPO training input — the exact text Agent A
        # received before outputting its confidence score at this checkpoint.
        con.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                record_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id       TEXT NOT NULL,
                rollout_id     TEXT NOT NULL,
                claim_id       TEXT NOT NULL,
                claim_text     TEXT NOT NULL,    -- locked after post_step_A, never changes
                is_material    INTEGER NOT NULL, -- 1 or 0
                confidence_p   REAL NOT NULL,    -- Agent A's confidence AT THIS checkpoint
                verdict        TEXT NOT NULL,    -- SUPPORTED/PARTIAL/NOT_SUPPORTED/IDK
                checkpoint     TEXT NOT NULL,    -- "post_step_A" / "post_cycle1" / "post_cycle2"
                agent_a_prompt TEXT NOT NULL,    -- GRPO training input: full context Agent A saw
                evidence_chunks TEXT,            -- JSON array of chunk_ids Agent A cited
                reasoning      TEXT,
                timestamp      TEXT NOT NULL
            )
        """)

        # ── TABLE 3: attacks ──────────────────────────────────────────────────
        # One row per Agent B challenge per cycle.
        # p_after_attack starts NULL — filled after Agent A revises in step C.
        # b_reward is NULL — feedback loop writes this.
        # agent_b_prompt is NULL — Phase 2 (Agent B fine-tuning).
        con.execute("""
            CREATE TABLE IF NOT EXISTS attacks (
                attack_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id         TEXT NOT NULL,
                rollout_id       TEXT NOT NULL,
                claim_id         TEXT NOT NULL,
                cycle            INTEGER NOT NULL,  -- 1 or 2
                b_critique_text  TEXT NOT NULL,     -- what Agent B actually said
                b_challenge_type TEXT NOT NULL,     -- CHUNK_CURRENCY/JURISDICTION_SCOPE/etc.
                p_before_attack  REAL NOT NULL,     -- Agent A's confidence BEFORE this attack
                p_after_attack   REAL,              -- NULL until Agent A revises in step C
                agent_b_prompt   TEXT,              -- NULL: Phase 2 work
                timestamp        TEXT NOT NULL,
                b_reward         REAL               -- NULL: feedback loop writes this
            )
        """)

        # ── TABLE 4: judge_verdicts ───────────────────────────────────────────
        # One row per claim after Judge runs. Last thing MAD writes.
        con.execute("""
            CREATE TABLE IF NOT EXISTS judge_verdicts (
                query_id          TEXT NOT NULL,
                rollout_id        TEXT NOT NULL,
                claim_id          TEXT NOT NULL,
                v_label           REAL NOT NULL,    -- 1.0 / 0.5 / 0.0
                judge_reasoning   TEXT,
                evidence_chunk_ids TEXT,            -- JSON array of chunks Judge used
                timestamp         TEXT NOT NULL,
                PRIMARY KEY (query_id, rollout_id, claim_id)
            )
        """)

    print(f"[Storage] Database initialised at {DB_PATH}")


def _add_column_if_missing(con: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    """Add a column to an existing table only if it doesn't already exist."""
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {r["name"] for r in rows}
    if column not in existing:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


# ── Public write functions ─────────────────────────────────────────────────────

def new_rollout_id() -> str:
    """Generate a unique rollout ID for one MAD pipeline run."""
    return str(uuid.uuid4())


def write_query(
    query_id:   str,
    rollout_id: str,
    query_text: str,
    llm_answer: str,
    chunk_ids:  List[str],
) -> None:
    """
    TABLE 1 — Write one row when the query enters the MAD pipeline.
    Called FIRST, before anything else runs.
    final_cse_score and routing_decision are left NULL — CSE fills them.
    """
    with _conn() as con:
        con.execute(
            """INSERT OR IGNORE INTO queries
               (query_id, rollout_id, query_text, llm_answer, rag_chunk_ids, timestamp)
               VALUES (?,?,?,?,?,?)""",
            (query_id, rollout_id, query_text, llm_answer,
             json.dumps(chunk_ids), _now())
        )


def write_claims_checkpoint(
    query_id:          str,
    rollout_id:        str,
    claims:            List[Claim],
    checkpoint:        str,                    # "post_step_A" / "post_cycle1" / "post_cycle2"
    per_claim_prompts: "dict[int, str] | str", # per-claim prompts dict OR single shared string
) -> None:
    """
    TABLE 2 — Insert one NEW row per claim at each checkpoint.

    NEVER update existing rows. The three checkpoints build up a trajectory:
      post_step_A   → Agent A's initial confidence after first RAG verification
      post_cycle1   → Agent A's confidence after revising based on B's cycle 1 challenges
      post_cycle2   → Agent A's final confidence after revising based on B's cycle 2 challenges

    The full trajectory is what GRPO uses: it shows how Agent A's confidence
    changed in response to challenges, and the Brier reward scores whether
    those confidence changes were well-calibrated.

    agent_a_prompt at each checkpoint:
      post_step_A:  system + query + RAG chunks + claim text
      post_cycle1:  above + Agent B's cycle 1 challenge for this claim
      post_cycle2:  above + Agent B's cycle 2 challenge for this claim
    """
    rows = []
    for c in claims:
        # Resolve per-claim prompt: use dict lookup if available, else shared string
        if isinstance(per_claim_prompts, dict):
            prompt = per_claim_prompts.get(c.claim_id, "")
        else:
            prompt = per_claim_prompts  # backward compat — single shared string

        rows.append((
            query_id,
            rollout_id,
            str(c.claim_id),
            c.claim_text,
            1 if c.is_material else 0,
            float(c.confidence),
            c.verdict.value if c.verdict else "IDK",
            checkpoint,
            prompt,
            json.dumps(c.evidence_chunks),
            c.reasoning,
            _now(),
        ))

    with _conn() as con:
        con.executemany(
            """INSERT INTO claims
               (query_id, rollout_id, claim_id, claim_text, is_material,
                confidence_p, verdict, checkpoint, agent_a_prompt,
                evidence_chunks, reasoning, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows
        )


def write_attacks(
    query_id:        str,
    rollout_id:      str,
    challenges:      List[Challenge],
    claims_before:   List[Claim],    # Agent A's confidence BEFORE this cycle's revision
    cycle:           int,
) -> None:
    """
    TABLE 3 — Insert one row per challenge after Agent B generates them.

    p_before_attack is set now (Agent A's current confidence).
    p_after_attack is NULL — filled after Agent A revises in step C.
    b_reward is NULL — feedback loop fills this later.

    The feedback loop uses p_before_attack and p_after_attack to compute
    Agent B's precision reward:
      +1 if B attacked a wrong claim (v=0 or 0.5) AND delta_p >= 0.2
      -1 if B attacked a correct claim (v=1.0) — gaslighting penalty
       0 if attack had no meaningful effect
    """
    confidence_map = {c.claim_id: c.confidence for c in claims_before}

    rows = []
    for ch in challenges:
        rows.append((
            query_id,
            rollout_id,
            str(ch.claim_id),
            cycle,
            ch.challenge_text,
            ch.challenge_type.value,
            float(confidence_map.get(ch.claim_id, 0.5)),  # p_before_attack
            None,    # p_after_attack — filled in update_attack_p_after
            None,    # agent_b_prompt — Phase 2
            _now(),
            None,    # b_reward — feedback loop writes this
        ))

    with _conn() as con:
        con.executemany(
            """INSERT INTO attacks
               (query_id, rollout_id, claim_id, cycle, b_critique_text,
                b_challenge_type, p_before_attack, p_after_attack,
                agent_b_prompt, timestamp, b_reward)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows
        )


def update_attack_p_after(
    query_id:      str,
    rollout_id:    str,
    revised_claims: List[Claim],
    cycle:         int,
) -> None:
    """
    TABLE 3 — Update p_after_attack for all attacks in this cycle.

    Called AFTER Agent A revises verdicts in step C.
    This is the ONLY time MAD updates an existing row (per friend's spec).
    """
    with _conn() as con:
        for claim in revised_claims:
            con.execute(
                """UPDATE attacks
                   SET p_after_attack = ?
                   WHERE query_id = ? AND rollout_id = ?
                     AND claim_id = ? AND cycle = ?
                     AND p_after_attack IS NULL""",
                (float(claim.confidence), query_id, rollout_id,
                 str(claim.claim_id), cycle)
            )


def write_judge_verdicts(
    query_id:       str,
    rollout_id:     str,
    judge_verdicts: List[JudgeVerdict],
    evidence_pool_ids: List[str],
) -> None:
    """
    TABLE 4 — Insert one row per claim after Judge runs.
    This is the LAST thing MAD writes. After this, data passes to CSE.

    v_label (1.0/0.5/0.0) is what the feedback loop uses as the ground
    truth signal when computing Agent A's Brier reward:
      brier_reward = 2 * p_final * v_label - p_final^2
    """
    rows = []
    for jv in judge_verdicts:
        rows.append((
            query_id,
            rollout_id,
            str(jv.claim_id),
            float(jv.score),
            jv.reasoning,
            json.dumps(evidence_pool_ids),
            _now(),
        ))

    with _conn() as con:
        con.executemany(
            """INSERT OR REPLACE INTO judge_verdicts
               (query_id, rollout_id, claim_id, v_label,
                judge_reasoning, evidence_chunk_ids, timestamp)
               VALUES (?,?,?,?,?,?,?)""",
            rows
        )


def update_query_cse(
    query_id:         str,
    rollout_id:       str,
    cse_score:        float,
    routing_decision: str,
    cse_components:   Optional[dict] = None,
    cse_version:      str = "v0.1",
) -> None:
    """
    TABLE 1 — Update final_cse_score, routing_decision, and CSE component scores.
    Called by CSE after it computes the final score.
    MAD pipeline calls this at the end of mad_pipeline.run_mad().

    cse_components should be a dict with keys: f_llm, h_llm, relevancy, judge_eval.
    """
    comp = cse_components or {}
    with _conn() as con:
        con.execute(
            """UPDATE queries
               SET final_cse_score = ?,
                   routing_decision = ?,
                   cse_f_llm        = ?,
                   cse_h_llm        = ?,
                   cse_relevancy    = ?,
                   cse_judge_eval   = ?,
                   cse_version      = ?
               WHERE query_id = ? AND rollout_id = ?""",
            (
                cse_score,
                routing_decision,
                comp.get("f_llm"),
                comp.get("h_llm"),
                comp.get("relevancy"),
                comp.get("judge_eval"),
                cse_version,
                query_id,
                rollout_id,
            )
        )


# ── Prompt builders ────────────────────────────────────────────────────────────
# These functions build the agent_a_prompt string that gets stored in the
# claims table at each checkpoint. This is critical — it's what TRL GRPO
# uses as training input.

def build_agent_a_prompt_step_a(
    system_prompt: str,
    query:         str,
    rag_chunks:    list,
    claim_text:    str,
) -> str:
    """
    Builds agent_a_prompt for checkpoint post_step_A.

    Contains: system instruction + query + RAG chunks + claim text.
    This is the full context Agent A saw when it first produced its
    initial confidence score for this claim.
    """
    chunks_text = "\n".join(
        f"[{c.get('chunk_id', i)}] (tier {c.get('tier', '?')}): {c.get('text', '')[:300]}"
        for i, c in enumerate(rag_chunks)
    )
    return (
        f"System: {system_prompt}\n\n"
        f"User query: {query}\n\n"
        f"Retrieved regulatory evidence:\n{chunks_text}\n\n"
        f"Claim to verify: {claim_text}\n\n"
        f"Output your verdict (SUPPORTED/PARTIAL/NOT_SUPPORTED/IDK) "
        f"and confidence (0.0 to 1.0):"
    )


def build_agent_a_prompt_post_cycle(
    base_prompt:     str,
    cycle:           int,
    b_challenge_text: str,
) -> str:
    """
    Builds agent_a_prompt for checkpoint post_cycle1 or post_cycle2.

    Takes the previous prompt and APPENDS Agent B's challenge for this claim.
    This means the prompt grows at each checkpoint — Agent A sees the full
    history of challenges it has faced and how it responded.
    """
    return (
        f"{base_prompt}\n\n"
        f"--- Agent B Cycle {cycle} Challenge ---\n"
        f"{b_challenge_text}\n\n"
        f"Review B's challenge against your evidence. "
        f"If B identified a genuine regulatory gap or exception, lower your confidence. "
        f"If B is attacking a well-supported claim without new evidence, maintain your position. "
        f"Output your revised verdict and confidence:"
    )
