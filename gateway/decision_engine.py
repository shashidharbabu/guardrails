"""
Decision Engine — combines PII_Score + JB_Score + PI_Score into Gateway Score.
Makes final routing decision: PASS → MAD pipeline, ESCALATE → flag+log, BLOCK → error.

Thresholds:
  Gateway Score < 0.3   → PASS
  0.3 ≤ score ≤ 0.7    → ESCALATE
  Gateway Score > 0.7   → BLOCK

Hard overrides (checked BEFORE composite scoring):
  jb_score  >= 0.7  → immediate BLOCK
  pi_score  >= 0.7  → immediate BLOCK
  pii_score >= 0.9  → immediate BLOCK

Weights:
  PII: 0.30   JB: 0.40   PI: 0.30
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from gateway import telemetry as gw_telemetry


class Decision(str, Enum):
    PASS = "PASS"
    ESCALATE = "ESCALATE"
    BLOCK = "BLOCK"


@dataclass
class GatewayResult:
    """Full result object returned by the gateway for every request."""

    decision: Decision
    gateway_score: float
    pii_score: float
    jb_score: float
    pi_score: float
    pii_entities: List[Dict] = field(default_factory=list)
    threat_types: List[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    raw_input: str = ""

    @property
    def is_allowed(self) -> bool:
        """True if request should proceed (PASS or ESCALATE)."""
        return self.decision != Decision.BLOCK

    def to_dict(self) -> Dict:
        return {
            "decision": self.decision.value,
            "is_allowed": self.is_allowed,
            "gateway_score": round(self.gateway_score, 4),
            "scores": {
                "pii": round(self.pii_score, 4),
                "jailbreak": round(self.jb_score, 4),
                "prompt_injection": round(self.pi_score, 4),
            },
            "pii_entities": self.pii_entities,
            "threat_types": self.threat_types,
            "blocked_reason": self.blocked_reason,
        }


class DecisionEngine:
    """
    Computes weighted composite score and makes routing decision.

    Constructor params:
      pass_threshold      — composite score below this → PASS (default 0.3)
      block_threshold     — composite score above this → BLOCK (default 0.7)
      pii_weight          — weight for PII score (default 0.30)
      jb_weight           — weight for JB score (default 0.40)
      pi_weight           — weight for PI score (default 0.30)
      jb_override_threshold  — single JB score at/above this → immediate BLOCK (default 0.7)
      pi_override_threshold  — single PI score at/above this → immediate BLOCK (default 0.7)
      pii_override_threshold — single PII score at/above this → immediate BLOCK (default 0.9)
    """

    def __init__(
        self,
        pass_threshold: float = 0.3,
        block_threshold: float = 0.7,
        pii_weight: float = 0.30,
        jb_weight: float = 0.40,
        pi_weight: float = 0.30,
        jb_override_threshold: float = 0.7,
        pi_override_threshold: float = 0.7,
        pii_override_threshold: float = 0.9,
    ):
        assert abs(pii_weight + jb_weight + pi_weight - 1.0) < 1e-6, (
            "Weights must sum to 1.0"
        )
        self.pass_threshold = pass_threshold
        self.block_threshold = block_threshold
        self.pii_weight = pii_weight
        self.jb_weight = jb_weight
        self.pi_weight = pi_weight
        self.jb_override_threshold = jb_override_threshold
        self.pi_override_threshold = pi_override_threshold
        self.pii_override_threshold = pii_override_threshold

    def decide(
        self,
        raw_input: str,
        pii_score: float,
        jb_score: float,
        pi_score: float,
        pii_entities: List[Dict] = None,
    ) -> GatewayResult:
        pii_entities = pii_entities or []

        # Build human-readable threat descriptions
        threat_types = []
        if jb_score >= 0.4:
            threat_types.append(f"JAILBREAK ({jb_score:.3f})")
        if pi_score >= 0.4:
            threat_types.append(f"PROMPT_INJECTION ({pi_score:.3f})")
        if pii_entities:
            types = list({e["entity_type"] for e in pii_entities})
            threat_types.append(f"PII[{','.join(types[:3])}] ({pii_score:.3f})")

        # Hard overrides — single high-confidence signal → immediate BLOCK
        if jb_score >= self.jb_override_threshold:
            res = GatewayResult(
                decision=Decision.BLOCK,
                gateway_score=min(1.0, jb_score),
                pii_score=pii_score,
                jb_score=jb_score,
                pi_score=pi_score,
                pii_entities=pii_entities,
                threat_types=threat_types,
                blocked_reason=(
                    f"Blocked — high-confidence jailbreak (jb_score={jb_score:.3f}). "
                    f"Threats: {', '.join(threat_types)}"
                ),
                raw_input=raw_input,
            )
            gw_telemetry.tag_current_span(
                decision="BLOCK",
                override_branch="jb_hard",
                gateway_score=res.gateway_score,
            )
            return res

        if pi_score >= self.pi_override_threshold:
            res = GatewayResult(
                decision=Decision.BLOCK,
                gateway_score=min(1.0, pi_score),
                pii_score=pii_score,
                jb_score=jb_score,
                pi_score=pi_score,
                pii_entities=pii_entities,
                threat_types=threat_types,
                blocked_reason=(
                    f"Blocked — high-confidence prompt injection (pi_score={pi_score:.3f}). "
                    f"Threats: {', '.join(threat_types)}"
                ),
                raw_input=raw_input,
            )
            gw_telemetry.tag_current_span(
                decision="BLOCK",
                override_branch="pi_hard",
                gateway_score=res.gateway_score,
            )
            return res

        if pii_score >= self.pii_override_threshold:
            res = GatewayResult(
                decision=Decision.BLOCK,
                gateway_score=min(1.0, pii_score),
                pii_score=pii_score,
                jb_score=jb_score,
                pi_score=pi_score,
                pii_entities=pii_entities,
                threat_types=threat_types,
                blocked_reason=(
                    f"Blocked — high-confidence PII exfiltration (pii_score={pii_score:.3f}). "
                    f"Threats: {', '.join(threat_types)}"
                ),
                raw_input=raw_input,
            )
            gw_telemetry.tag_current_span(
                decision="BLOCK",
                override_branch="pii_hard",
                gateway_score=res.gateway_score,
            )
            return res

        # Composite weighted score
        gateway_score = (
            self.pii_weight * pii_score
            + self.jb_weight * jb_score
            + self.pi_weight * pi_score
        )
        gateway_score = min(1.0, max(0.0, gateway_score))

        if gateway_score < self.pass_threshold:
            decision = Decision.PASS
            reason = None
        elif gateway_score <= self.block_threshold:
            decision = Decision.ESCALATE
            reason = (
                f"Flagged for review (gateway_score={gateway_score:.3f}). "
                f"Threats: {', '.join(threat_types) if threat_types else 'borderline score'}"
            )
        else:
            decision = Decision.BLOCK
            reason = (
                f"Blocked (gateway_score={gateway_score:.3f}). "
                f"Threats: {', '.join(threat_types)}"
            )

        res = GatewayResult(
            decision=decision,
            gateway_score=gateway_score,
            pii_score=pii_score,
            jb_score=jb_score,
            pi_score=pi_score,
            pii_entities=pii_entities,
            threat_types=threat_types,
            blocked_reason=reason,
            raw_input=raw_input,
        )
        gw_telemetry.tag_current_span(
            decision=decision.value,
            override_branch="composite",
            gateway_score=gateway_score,
        )
        return res
