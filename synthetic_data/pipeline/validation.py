from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")


def normalize_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = _NON_ALNUM_RE.sub(" ", q)
    q = _WS_RE.sub(" ", q).strip()
    return q


def word_count(q: str) -> int:
    qn = normalize_query(q)
    return 0 if not qn else len(qn.split(" "))


def char_ngrams(q: str, n: int = 5) -> set[str]:
    qn = normalize_query(q)
    if len(qn) < n:
        return {qn} if qn else set()
    return {qn[i : i + n] for i in range(0, len(qn) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def semantic_similarity(a_text: str, b_text: str) -> float:
    """
    Lightweight semantic proxy based on normalized word overlap.
    This avoids external model dependencies while still measuring topical closeness.
    """
    a = set(normalize_query(a_text).split())
    b = set(normalize_query(b_text).split())
    return jaccard(a, b)


def is_compliance_question(q: str) -> bool:
    qn = (q or "").strip().lower()
    bad_prefixes = ("what does", "how does", "according to")
    return any(qn.startswith(p) for p in bad_prefixes)


def validate_embedding_triplets(
    triplets: List[dict],
    *,
    min_similarity: float = 0.08,
    max_similarity: float = 0.75,
) -> Tuple[List[dict], List[dict]]:
    kept: List[dict] = []
    rejected: List[dict] = []

    for t in triplets:
        reasons: List[str] = []
        pos = t.get("positive", {}) or {}
        neg = t.get("hard_negative", {}) or {}
        pos_text = str(pos.get("text", "") or "")
        neg_text = str(neg.get("text", "") or "")
        if not pos_text or not neg_text:
            reasons.append("missing_positive_or_negative_text")
        else:
            sim = semantic_similarity(pos_text, neg_text)
            t.setdefault("validation", {})
            t["validation"]["semantic_similarity"] = sim
            if sim < min_similarity:
                reasons.append("negative_not_semantically_related")
            if sim > max_similarity:
                reasons.append("negative_too_similar_to_positive")

        if reasons:
            out = dict(t)
            out["reject_reasons"] = reasons
            rejected.append(out)
        else:
            kept.append(t)

    return kept, rejected


def _response_get(d: dict, path: List[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def validate_agent_pairs(
    pairs: List[dict],
    *,
    target_distribution: Dict[str, float] | None = None,
    allowed_distribution_drift: float = 0.05,
) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    kept: List[dict] = []
    rejected: List[dict] = []

    for p in pairs:
        reasons: List[str] = []

        decision = (p.get("metadata", {}) or {}).get("decision") or _response_get(p, ["response", "decision"])
        if decision:
            decision = str(decision).upper()

        user_query = _response_get(p, ["instruction", "user_query"]) or ""
        if is_compliance_question(str(user_query)):
            reasons.append("query_not_realistic")

        resp = p.get("response")
        if not isinstance(resp, dict):
            reasons.append("response_not_object")
        else:
            # Decision consistency checks
            policy_violations = resp.get("policy_violations", [])
            if decision == "BLOCK" and (not isinstance(policy_violations, list) or len(policy_violations) == 0):
                reasons.append("block_missing_policy_violations")
            if decision == "ALLOW":
                try:
                    conf = float(resp.get("confidence", 0.0))
                    if conf <= 0.7:
                        reasons.append("allow_low_confidence")
                except Exception:
                    reasons.append("allow_confidence_not_float")

            # Policy citation validity: each violation policy string should mention one retrieved_context source doc_id
            ctx_sources = {
                str(c.get("source"))
                for c in (_response_get(p, ["instruction", "retrieved_context"]) or [])
                if isinstance(c, dict) and c.get("source")
            }
            if ctx_sources and isinstance(policy_violations, list):
                for v in policy_violations:
                    if not isinstance(v, dict):
                        continue
                    pol = str(v.get("policy", "") or "")
                    if pol and not any(src in pol for src in ctx_sources):
                        reasons.append("policy_citation_not_in_context")
                        break

        if reasons:
            out = dict(p)
            out["reject_reasons"] = reasons
            rejected.append(out)
        else:
            kept.append(p)

    stats: Dict[str, Any] = {}
    decisions = Counter((x.get("metadata", {}) or {}).get("decision") for x in kept)
    stats["decision_counts"] = dict(decisions)

    if target_distribution and kept:
        total = len(kept)
        dist = {k: (decisions.get(k, 0) / total) for k in target_distribution.keys()}
        stats["decision_distribution"] = dist
        stats["distribution_within_tolerance"] = all(
            abs(dist.get(k, 0.0) - target_distribution[k]) <= allowed_distribution_drift for k in target_distribution
        )

    return kept, rejected, stats

