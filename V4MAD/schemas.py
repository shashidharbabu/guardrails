import json
import re
from typing import TypedDict, Optional, Any


# ── LangGraph state (flows through all nodes) ──────────────────────────────────

class MADState(TypedDict, total=False):
    query_id:        str
    run_id:          str
    user_query:      str
    baseline_answer: Optional[str]
    claims:          list[dict]     # [{claim_id, claim_text, claim_index, is_material, is_critical, confidence_prior}]
    query_chunks:    list[dict]     # top-5 chunks stored for this query
    claim_chunks:    dict[str, dict]  # {claim_id: {agent_a:[...], agent_b:[...], judge:[...]}}
    agent_outputs:   dict[str, dict]  # {claim_id: {agent_a:{0:{...},1:{...}}, agent_b:{...}}}
    judge_verdicts:  dict[str, dict]  # {claim_id: {v_label, judge_confidence, judge_reasoning, ...}}
    errors:          list[str]


# ── JSON parsing (robust 3-pass) ───────────────────────────────────────────────

_MD_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_agent_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _MD_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ── Agent output helpers ───────────────────────────────────────────────────────

VALID_VERDICTS = {"SUPPORTED", "PARTIAL", "NOT_SUPPORTED", "IDK"}


def validate_agent_output(parsed: dict) -> tuple[bool, str]:
    if "verdict" not in parsed:
        return False, "missing verdict"
    if parsed["verdict"] not in VALID_VERDICTS:
        return False, f"invalid verdict: {parsed['verdict']}"
    if "confidence_internal" not in parsed:
        return False, "missing confidence_internal"
    try:
        c = float(parsed["confidence_internal"])
        if not (0.0 <= c <= 1.0):
            return False, f"confidence out of range: {c}"
    except (TypeError, ValueError):
        return False, "confidence not numeric"
    return True, "ok"


def strip_for_peer(output: dict, label: str) -> dict:
    """Remove confidence_internal before showing to the other agent. Truncate to save tokens."""
    reasoning = output.get("reasoning", "")
    words = reasoning.split()
    if len(words) > 120:
        reasoning = " ".join(words[:120]) + "..."

    evidence = output.get("evidence_cited", [])
    if evidence:
        first = evidence[0]
        quote = first.get("relevant_quote", "")[:160]
        evidence = [{"chunk_id": first.get("chunk_id", ""), "relevant_quote": quote}]

    return {
        "debater_label":  label,
        "round_num":      output.get("round_num", 0),
        "verdict":        output.get("verdict", "IDK"),
        "reasoning":      reasoning,
        "evidence_cited": evidence,
    }
