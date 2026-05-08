"""Standalone healthcare reward shaping for Colab / TRL GRPO (completion + judge v).

Duplicate of repo `rlhf/feedback_loop/heuristics.py` logic so this file can be
copied to `/content/reward_fn/` in Colab without the full tree.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

log = logging.getLogger(__name__)

CITATION_PATTERNS = [
    re.compile(r"\b(HIPAA|45\s*CFR|42\s*CFR|21\s*CFR)\b", re.I),
    re.compile(r"\b(HL7|FHIR|ICD-?10|CPT|SNOMED)\b", re.I),
    re.compile(r"\b(FDA|DEA|CMS|HHS|NIH)\b", re.I),
    re.compile(r"\b(ISO\s*13485|EU\s*MDR|21st\s*Century\s*Cures)\b", re.I),
    re.compile(r"§\s*\d+\.\d+", re.I),
]

PHI_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "MEDICAL_LICENSE",
    "US_BANK_NUMBER",
]


def has_citation(text: str) -> bool:
    return any(p.search(text) for p in CITATION_PATTERNS)


def extract_confidence(completion: str) -> Optional[float]:
    m = re.search(r"CONFIDENCE:\s*([\d.]+)", completion, re.I)
    if not m:
        return None
    try:
        p = float(m.group(1))
        return max(0.0, min(1.0, p))
    except ValueError:
        return None


def _vb(v: float) -> float:
    return 0.10 if v > 0.7 else 0.0


def _cb(text: str) -> float:
    return 0.15 if has_citation(text) else 0.0


def _phi(text: str) -> float:
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        return 0.0
    try:
        eng = AnalyzerEngine()
        hits = eng.analyze(text=text, language="en", entities=PHI_ENTITIES)
        if any(h.score > 0.75 for h in hits):
            return 0.40
    except Exception as exc:  # noqa: BLE001
        log.debug("presidio skip: %s", exc)
    return 0.0


def scalar_reward(
    completion: str,
    judge_verdict: float,
    *,
    use_presidio: bool = True,
) -> float:
    """
    Full shaped reward for one completion given judge probability v in [0,1].
    Uses Brier-style term 2*p*v - p^2 when confidence line present; else -0.20 format penalty only path.
    """
    p = extract_confidence(completion)
    if p is None:
        return -0.20
    brier = 2.0 * p * judge_verdict - p**2
    r = (
        brier
        + _vb(judge_verdict)
        + _cb(completion)
        - (_phi(completion) if use_presidio else 0.0)
    )
    if p > 0.85 and not has_citation(completion):
        r -= 0.30
    return r


def reward_batch_for_trl(
    completions: list[str],
    judge_verdicts: Iterable[float],
    *,
    use_presidio: bool = True,
) -> list[float]:
    """Return one float reward per completion (TRL GRPO batch hook pattern)."""
    vs = list(judge_verdicts)
    if len(vs) != len(completions):
        raise ValueError("judge_verdicts must match completions length")
    return [
        scalar_reward(c, float(v), use_presidio=use_presidio)
        for c, v in zip(completions, vs)
    ]
