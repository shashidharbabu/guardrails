import json
import re

from src.schemas.schemas import (
    AgentOutputFull,
    AgentRole,
    AgentVerdict,
    EvidenceCitation,
)


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_json_object(raw: str) -> str:
    text = _strip_markdown_fences(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No complete JSON object found in agent response: {raw[:200]}")
    candidate = text[start : end + 1]
    return "".join(
        char for char in candidate if char in "\n\r\t" or ord(char) >= 32
    )


def _extract_string_field(text: str, field: str) -> str | None:
    match = re.search(rf'"{field}"\s*:\s*"(.*?)"', text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_float_field(text: str, field: str) -> float | None:
    match = re.search(rf'"{field}"\s*:\s*([0-9]*\.?[0-9]+)', text)
    return float(match.group(1)) if match else None


def _fallback_agent_payload(text: str) -> dict | None:
    verdict = _extract_string_field(text, "verdict")
    reasoning = _extract_string_field(text, "reasoning")
    confidence = _extract_float_field(text, "confidence_internal")
    if verdict is None or reasoning is None or confidence is None:
        return None
    evidence_cited = []
    for chunk_id in re.findall(r'"chunk_id"\s*:\s*"(.*?)"', text, flags=re.DOTALL)[:2]:
        evidence_cited.append({"chunk_id": chunk_id.strip(), "relevant_quote": ""})
    return {
        "verdict": verdict,
        "reasoning": reasoning,
        "evidence_cited": evidence_cited,
        "confidence_internal": confidence,
    }


def _payload_to_agent_output(
    payload: dict,
    role: AgentRole,
    round_num: int,
    raw: str,
) -> AgentOutputFull:
    from pydantic import ValidationError
    if not isinstance(payload, dict):
        raise ValueError(f"Agent response must be a JSON object: {raw[:200]}")
    try:
        verdict = AgentVerdict(payload["verdict"])
        evidence_cited = [
            EvidenceCitation(
                chunk_id=item["chunk_id"],
                relevant_quote=str(item.get("relevant_quote", ""))[:240],
            )
            for item in payload.get("evidence_cited", [])[:2]
        ]
        return AgentOutputFull(
            agent_role=role,
            round_num=round_num,
            verdict=verdict,
            reasoning=payload["reasoning"],
            evidence_cited=evidence_cited,
            confidence_internal=payload["confidence_internal"],
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ValueError(f"Invalid agent response for {role.value}: {raw[:200]}") from exc


def parse_agent_output_strict(raw: str, role: AgentRole, round_num: int) -> AgentOutputFull:
    payload = json.loads(_extract_json_object(raw))
    return _payload_to_agent_output(payload, role, round_num, raw)


def parse_agent_output(raw: str, role: AgentRole, round_num: int) -> AgentOutputFull:
    json_text = _extract_json_object(raw)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        payload = _fallback_agent_payload(json_text)
        if payload is None:
            raise ValueError(f"Could not parse agent JSON response: {raw[:200]}") from exc
    return _payload_to_agent_output(payload, role, round_num, raw)
