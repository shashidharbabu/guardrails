import json
import logging
import re
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from configs import config
from src.db.db import connect, insert_claim
from src.schemas.schemas import Claim, DecomposerOutput, PipelineState
from src.utils.prompts import DECOMPOSER_SYSTEM, decomposer_user
from src.utils.vllm_client import make_decomposer_client


logger = logging.getLogger(__name__)


def claim_coverage_check(
    baseline_answer: str,
    claims: list[Claim],
    threshold: float = 0.6,
) -> tuple[bool, float, str | None]:
    answer_words = set(w.lower() for w in re.findall(r"\b[a-zA-Z0-9]{4,}\b", baseline_answer))
    if not answer_words:
        return True, 1.0, None
    claim_words = set()
    for claim in claims:
        claim_words.update(w.lower() for w in re.findall(r"\b[a-zA-Z0-9]{4,}\b", claim.claim_text))
    overlap = answer_words & claim_words
    coverage_ratio = len(overlap) / len(answer_words)
    if coverage_ratio < threshold:
        warning = (
            f"Decomposer covered only {coverage_ratio:.0%} of original answer "
            f"(threshold {threshold:.0%}). Missing words: {list(answer_words - claim_words)[:10]}"
        )
        return False, coverage_ratio, warning
    return True, coverage_ratio, None


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)


def _strip_markdown_fence(raw: str) -> str:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def _extract_json_candidate(raw: str) -> str:
    text = _strip_markdown_fence(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _repair_redundant_claim_wrappers(candidate: str) -> str:
    return re.sub(
        (
            r"\{\s*(\{\s*\"claim_text\"\s*:\s*\"[^\"]*\"\s*,\s*"
            r"\"claim_index\"\s*:\s*\d+\s*,\s*"
            r"\"is_material\"\s*:\s*(?:true|false)\s*,\s*"
            r"\"is_critical\"\s*:\s*(?:true|false)\s*,\s*"
            r"\"confidence_prior\"\s*:\s*[0-9.]+\s*\})\s*\}"
        ),
        r"\1",
        candidate,
        flags=re.DOTALL,
    )


def _parse_decomposer_output(raw_response: str, query_id: str) -> list[Claim]:
    candidate = _extract_json_candidate(raw_response)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        repaired = _repair_redundant_claim_wrappers(candidate)
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError:
            raise ValueError(
                f"Could not parse decomposer JSON for {query_id}: {raw_response[:200]}"
            ) from exc
    raw_claims = payload.get("claims") if isinstance(payload, dict) else payload
    if not isinstance(raw_claims, list):
        raise ValueError(f"Decomposer response missing claims list: {raw_response[:200]}")
    claims = []
    for index, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid claim object in decomposer response: {raw_response[:200]}")
        claim_payload = {
            **item,
            "claim_id": str(uuid4()),
            "claim_index": item.get("claim_index", index),
        }
        try:
            claims.append(Claim(**claim_payload))
        except ValidationError as exc:
            raise ValueError(f"Invalid decomposer claim for {query_id}: {raw_response[:200]}") from exc
    return claims


async def decompose_node(state: PipelineState) -> PipelineState:
    if not state.baseline_answer:
        raise ValueError("decompose_node requires state.baseline_answer")
    client = make_decomposer_client()
    message = await client.ainvoke(
        [
            SystemMessage(content=DECOMPOSER_SYSTEM),
            HumanMessage(content=decomposer_user(state.user_query, state.baseline_answer)),
        ]
    )
    raw_response = _message_text(message)
    claims = _parse_decomposer_output(raw_response, state.query_id)
    coverage_check, coverage_ratio, coverage_warning = claim_coverage_check(
        state.baseline_answer, claims, threshold=0.6,
    )
    decomposer_output = DecomposerOutput(
        claims=claims,
        coverage_check_passed=coverage_check,
        coverage_ratio=coverage_ratio,
        coverage_warning=coverage_warning,
    )
    with connect(config.SQLITE_DB_PATH) as conn:
        for claim in decomposer_output.claims:
            insert_claim(conn, claim, state.query_id, coverage_check, coverage_ratio)
    if not coverage_check:
        logger.warning(coverage_warning)
    state.claims = decomposer_output.claims
    return state
