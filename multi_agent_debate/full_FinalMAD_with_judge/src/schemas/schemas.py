from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AgentVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    IDK = "IDK"


class JudgeVerdict(float, Enum):
    SUPPORTED = 1.0
    PARTIAL = 0.5
    NOT_SUPPORTED = 0.0


class AgentRole(str, Enum):
    AGENT_A = "agent_a"
    AGENT_B = "agent_b"


class Claim(BaseModel):
    claim_id: str
    claim_text: str
    claim_index: int
    is_material: bool
    is_critical: bool
    confidence_prior: float = Field(ge=0.50, le=0.90)


class DecomposerOutput(BaseModel):
    claims: List[Claim]
    coverage_check_passed: bool
    coverage_ratio: float
    coverage_warning: Optional[str] = None


class EvidenceCitation(BaseModel):
    chunk_id: str
    relevant_quote: str


class AgentOutputFull(BaseModel):
    """Full output, written to DB. Never passed to another agent or judge."""

    agent_role: AgentRole
    round_num: int
    verdict: AgentVerdict
    reasoning: str
    evidence_cited: List[EvidenceCitation]
    confidence_internal: float = Field(ge=0.0, le=1.0)


class AgentOutputStripped(BaseModel):
    """Stripped output for inter-agent passing and judge input. No confidence."""

    debater_label: str
    round_num: int
    verdict: AgentVerdict
    reasoning: str
    evidence_cited: List[EvidenceCitation]


def strip_for_peer(output: AgentOutputFull, debater_label: str) -> AgentOutputStripped:
    """Convert full agent output into the anonymized peer-view schema."""
    reasoning_words = output.reasoning.split()
    evidence_cited = [
        EvidenceCitation(
            chunk_id=item.chunk_id,
            relevant_quote=item.relevant_quote[:160],
        )
        for item in output.evidence_cited[:1]
    ]
    return AgentOutputStripped(
        debater_label=debater_label,
        round_num=output.round_num,
        verdict=output.verdict,
        reasoning=" ".join(reasoning_words[:120]),
        evidence_cited=evidence_cited,
    )


class JudgeOutput(BaseModel):
    v_label: JudgeVerdict
    judge_confidence: float = Field(ge=0.0, le=1.0)
    judge_reasoning: str
    evidence_chunk_ids: List[str]


class PipelineState(BaseModel):
    query_id: str
    run_id: str
    user_query: str
    rag_chunks: List[dict]
    baseline_answer: Optional[str] = None
    claims: List[Claim] = Field(default_factory=list)
    agent_outputs_by_claim: dict = Field(default_factory=dict)
    judge_verdicts_by_claim: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
