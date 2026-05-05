"""
models.py — Pydantic schemas for the MAD pipeline.
These are the canonical data contracts between every module.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    SUPPORTED     = "SUPPORTED"
    PARTIAL       = "PARTIAL"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    IDK           = "IDK"


class ChallengeType(str, Enum):
    CHUNK_CURRENCY     = "CHUNK_CURRENCY"       # Is the cited chunk current / not superseded?
    JURISDICTION_SCOPE = "JURISDICTION_SCOPE"   # Does this regulation apply to this jurisdiction?
    EXCEPTION_EXISTENCE= "EXCEPTION_EXISTENCE"  # Does an exception / carve-out apply?
    GAP_FINDING        = "GAP_FINDING"          # Missing regulatory context Agent A overlooked


class RoutingDecision(str, Enum):
    DELIVER      = "DELIVER"       # aggregate > 0.8, all material v=1.0
    RETRY        = "RETRY"         # aggregate 0.4–0.8 → send correction to LLM
    HARD_BLOCK   = "HARD_BLOCK"    # any is_material claim with judge score=0.0
    HUMAN_REVIEW = "HUMAN_REVIEW"  # aggregate < 0.4


# ── Core schemas ──────────────────────────────────────────────────────────────

class Claim(BaseModel):
    """Atomic verifiable statement extracted from the LLM answer."""
    claim_id:       int
    claim_text:     str
    is_material:    bool = True          # True = regulatory obligation / penalty / threshold
    confidence:     float = Field(default=0.7, ge=0.0, le=1.0)  # Agent A's calibrated belief
    verdict:        Optional[Verdict] = None
    evidence_chunks: List[str] = []     # chunk_ids used as evidence
    reasoning:      str = ""


class EvidenceChunk(BaseModel):
    """A single regulatory document chunk from the RAG pipeline."""
    chunk_id: str
    text:     str
    source:   str
    tier:     int = 1   # 1=primary law, 2=guidance, 3=framework, 4=commentary


class Challenge(BaseModel):
    """A single challenge raised by Agent B against a specific claim."""
    claim_id:        int
    challenge_type:  ChallengeType
    challenge_text:  str
    evidence_chunks: List[str] = []          # chunk_ids B retrieved for this challenge
    suggested_verdict: Optional[Verdict] = None


class DebateCycle(BaseModel):
    """Full record of one challenge-revision cycle."""
    cycle_number:       int
    agent_a_report:     List[Claim]       # A's verdicts at start of cycle
    agent_b_challenges: List[Challenge]   # B's challenges
    agent_a_revised:    List[Claim]       # A's updated verdicts after B's challenges
    confidence_signal:  float             # min aggregation of material claim confidences


class JudgeVerdict(BaseModel):
    """Judge's independent score for a single claim."""
    claim_id:   int
    claim_text: str
    is_material: bool
    score:      float   # 1.0 = fully supported, 0.5 = partial, 0.0 = unsupported
    reasoning:  str


class MADOutput(BaseModel):
    """Full output of the MAD pipeline — returned to caller / API."""
    query:               str
    llm_answer:          str
    claims:              List[Claim]
    debate_cycles:       List[DebateCycle]
    evidence_pool:       List[EvidenceChunk]
    judge_verdicts:      List[JudgeVerdict]
    correction_signal:   Optional[str]       # Sent to LLM on retry
    routing_decision:    str                 # RoutingDecision value
    aggregate_confidence: float
    debate_transcript:   str                 # Human-readable full transcript
    # Storage IDs — needed for feedback loop to join tables
    query_id:            str = ""            # UUID — joins all 4 tables
    rollout_id:          str = ""            # UUID — for GRPO multi-rollout comparison
    # CSE full breakdown (None if CSE import failed completely)
    cse_result:          Optional[dict] = None   # CSEResult.as_dict() or None

    class Config:
        arbitrary_types_allowed = True
