from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from configs import config
from src.db.db import connect, insert_judge_verdict
from src.schemas.schemas import AgentOutputFull, JudgeOutput, JudgeVerdict, PipelineState, strip_for_peer
from src.utils.prompts import JUDGE_SYSTEM, judge_user
from src.utils.vllm_client import make_judge_client


logger = logging.getLogger(__name__)


def _make_langfuse_client():
    try:
        from langfuse import Langfuse
    except Exception:
        logger.info("Langfuse SDK unavailable; running without trace emission")
        return None
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        logger.info("Langfuse keys missing; running without trace emission")
        return None
    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
    )


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


def _parse_judge_output(raw: str) -> JudgeOutput:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse judge JSON response: {raw[:200]}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Judge response must be a JSON object: {raw[:200]}")
    try:
        v_label = JudgeVerdict(float(payload["v_label"]))
        return JudgeOutput(
            v_label=v_label,
            judge_confidence=payload["judge_confidence"],
            judge_reasoning=payload["judge_reasoning"],
            evidence_chunk_ids=list(payload.get("evidence_chunk_ids", [])),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ValueError(f"Invalid judge response: {raw[:200]}") from exc


def _round_output(
    outputs_by_role: dict,
    role_name: str,
    round_num: int,
    claim_id: str,
) -> AgentOutputFull | None:
    role_outputs = outputs_by_role.get(role_name)
    if not isinstance(role_outputs, dict):
        logger.error("Missing outputs for %s on claim %s", role_name, claim_id)
        return None
    output = role_outputs.get(round_num)
    if not isinstance(output, AgentOutputFull):
        logger.error("Missing round %s output for %s on claim %s", round_num, role_name, claim_id)
        return None
    return output


def _usage_counts(message) -> tuple[int, int]:
    usage_metadata = getattr(message, "usage_metadata", None) or {}
    if usage_metadata:
        return int(usage_metadata.get("input_tokens") or 0), int(usage_metadata.get("output_tokens") or 0)
    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or {}
    return int(token_usage.get("prompt_tokens") or 0), int(token_usage.get("completion_tokens") or 0)


async def judge_node(state: PipelineState, langfuse=None) -> PipelineState:
    if langfuse is None:
        langfuse = _make_langfuse_client()
    client = make_judge_client()

    for claim in state.claims:
        outputs_by_role = state.agent_outputs_by_claim.get(claim.claim_id)
        if not isinstance(outputs_by_role, dict):
            logger.error("Missing agent outputs for claim %s", claim.claim_id)
            continue

        a_r0 = _round_output(outputs_by_role, "agent_a", 0, claim.claim_id)
        a_r1 = _round_output(outputs_by_role, "agent_a", 1, claim.claim_id)
        b_r0 = _round_output(outputs_by_role, "agent_b", 0, claim.claim_id)
        b_r1 = _round_output(outputs_by_role, "agent_b", 1, claim.claim_id)
        if any(output is None for output in (a_r0, a_r1, b_r0, b_r1)):
            continue

        d1_r0 = strip_for_peer(a_r0, "Debater 1")
        d1_r1 = strip_for_peer(a_r1, "Debater 1")
        d2_r0 = strip_for_peer(b_r0, "Debater 2")
        d2_r1 = strip_for_peer(b_r1, "Debater 2")
        prompt = judge_user(claim, state.rag_chunks, state.user_query, d1_r0, d1_r1, d2_r0, d2_r1)

        trace = None
        trace_id = None
        if langfuse:
            trace = langfuse.trace(
                name="mad-judge-claim",
                input={
                    "run_id": state.run_id,
                    "query_id": state.query_id,
                    "claim_id": claim.claim_id,
                    "claim_text": claim.claim_text,
                    "user_query": state.user_query,
                },
                metadata={"run_id": state.run_id, "judge_model": config.JUDGE_MODEL_NAME},
            )
            trace_id = trace.id

        start_time = datetime.now(timezone.utc)
        start = time.perf_counter()
        message = await client.ainvoke(
            [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)]
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        raw_response = _message_text(message)
        tokens_in, tokens_out = _usage_counts(message)

        parse_ok = True
        try:
            verdict = _parse_judge_output(raw_response)
        except ValueError:
            logger.exception("Judge parse failed for claim %s", claim.claim_id)
            parse_ok = False
            verdict = JudgeOutput(
                v_label=JudgeVerdict.PARTIAL,
                judge_confidence=0.0,
                judge_reasoning="Judge response was unparseable; defaulted to uncertain.",
                evidence_chunk_ids=[],
            )

        if langfuse and trace_id:
            try:
                langfuse.generation(
                    trace_id=trace_id,
                    name="judge-llm-call",
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    model=config.JUDGE_MODEL_NAME,
                    model_parameters={"temperature": 0.0, "max_tokens": 1024},
                    input=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    output=raw_response,
                    metadata={
                        "run_id": state.run_id,
                        "query_id": state.query_id,
                        "claim_id": claim.claim_id,
                        "parse_ok": parse_ok,
                        "latency_ms": latency_ms,
                        "v_label": float(verdict.v_label.value),
                        "judge_confidence": verdict.judge_confidence,
                    },
                    usage_details={"input": tokens_in, "output": tokens_out, "total": tokens_in + tokens_out},
                )
                trace.update(
                    output={
                        "v_label": float(verdict.v_label.value),
                        "judge_confidence": verdict.judge_confidence,
                        "judge_reasoning": verdict.judge_reasoning,
                        "latency_ms": latency_ms,
                        "parse_ok": parse_ok,
                    },
                )
                langfuse.score(trace_id=trace_id, name="v_label", value=float(verdict.v_label.value))
                langfuse.score(trace_id=trace_id, name="judge_confidence", value=verdict.judge_confidence)
            except Exception:
                logger.warning("Langfuse logging failed for claim %s", claim.claim_id)

        with connect(config.SQLITE_DB_PATH) as conn:
            insert_judge_verdict(
                conn, verdict, claim.claim_id, config.JUDGE_MODEL_NAME, raw_response, latency_ms
            )
        state.judge_verdicts_by_claim[claim.claim_id] = verdict

    if langfuse:
        try:
            langfuse.flush()
        except Exception:
            pass
    return state
