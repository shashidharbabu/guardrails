"""Heuristic reward shaping aligned with docs/RLHF_IMPLEMENTATION.md (MAD uses numeric p)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

CITATION_PATTERNS = [
    re.compile(r"\b(HIPAA|45\s*CFR|42\s*CFR|21\s*CFR)\b", re.I),
    re.compile(r"\b(HL7|FHIR|ICD-?10|CPT|SNOMED)\b", re.I),
    re.compile(r"\b(FDA|DEA|CMS|HHS|NIH)\b", re.I),
    re.compile(r"\b(ISO\s*13485|EU\s*MDR|21st\s*Century\s*Cures)\b", re.I),
    re.compile(r"§\s*\d+\.\d+", re.I),
]

# Presidio entity list from RLHF doc (subset)
PHI_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "MEDICAL_LICENSE",
    "US_BANK_NUMBER",
]


@dataclass
class HeuristicComponents:
    verdict_bonus: float
    citation_bonus: float
    phi_penalty: float
    overconfidence_penalty: float
    format_penalty: float


def has_citation(text: str) -> bool:
    return any(p.search(text) for p in CITATION_PATTERNS)


def verdict_bonus(v_label: float) -> float:
    return 0.10 if v_label > 0.7 else 0.0


def citation_bonus_value(text: str) -> float:
    return 0.15 if has_citation(text) else 0.0


_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is not None:
        return _analyzer
    from presidio_analyzer import AnalyzerEngine

    _analyzer = AnalyzerEngine()
    return _analyzer


def phi_penalty_value(text: str, use_presidio: bool) -> float:
    if not use_presidio:
        return 0.0
    try:
        analyzer = _get_analyzer()
    except ImportError:
        log.warning("presidio-analyzer not installed; skipping PHI penalty")
        return 0.0
    try:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=PHI_ENTITIES,
        )
        if any(r.score > 0.75 for r in results):
            return 0.40
    except Exception as exc:  # noqa: BLE001
        log.warning("Presidio analyze failed (%s); skipping PHI penalty", exc)
        return 0.0
    return 0.0


def overconfidence_penalty_value(confidence: float, text: str) -> float:
    if confidence > 0.85 and not has_citation(text):
        return 0.30
    return 0.0


def format_penalty_value(confidence: Optional[float]) -> float:
    if confidence is None:
        return 0.20
    return 0.0


def build_completion_text(claim_text: str, reasoning: str, verdict: str) -> str:
    parts = [claim_text.strip(), reasoning.strip(), f"Verdict: {verdict}"]
    return "\n".join(p for p in parts if p)


def compute_heuristics(
    *,
    p_final: float,
    v_label: float,
    completion_text: str,
    use_presidio: bool,
) -> HeuristicComponents:
    vb = verdict_bonus(v_label)
    cb = citation_bonus_value(completion_text)
    ph = phi_penalty_value(completion_text, use_presidio)
    oc = overconfidence_penalty_value(p_final, completion_text)
    fp = format_penalty_value(p_final if p_final is not None else None)
    return HeuristicComponents(
        verdict_bonus=vb,
        citation_bonus=cb,
        phi_penalty=ph,
        overconfidence_penalty=oc,
        format_penalty=fp,
    )


def composite_auto_reward(brier_reward: float, h: HeuristicComponents) -> float:
    return (
        brier_reward
        + h.verdict_bonus
        + h.citation_bonus
        - h.phi_penalty
        - h.overconfidence_penalty
        - h.format_penalty
    )
