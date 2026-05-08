from __future__ import annotations
import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from configs import config
from src.agents.parser import parse_agent_output, parse_agent_output_strict
from src.db.db import connect, insert_agent_output
from src.schemas.schemas import AgentRole, Claim, PipelineState
from src.utils.prompts import AGENT_A_SYSTEM, AGENT_B_SYSTEM, round0_user
from src.utils.vllm_client import call_agent, make_agent_client


logger = logging.getLogger(__name__)

REPAIR_SYSTEM = (
    "Return ONLY valid JSON matching the required schema. No markdown. "
    "Quote all keys. Use an array for evidence_cited. "
    "Do not include comments or numbered object keys."
)


async def _call_with_retry(
    prompt,
    role: str,
    role_enum: AgentRole,
    round_num: int,
    client,
):
    raw, latency_ms, tokens_in, tokens_out = await call_agent(prompt, role, client)
    try:
        parsed = parse_agent_output_strict(raw, role_enum, round_num)
        return parsed, raw, latency_ms, tokens_in, tokens_out
    except Exception:
        pass

    repair_prompt = [
        SystemMessage(content=REPAIR_SYSTEM),
        HumanMessage(
            content=(
                "Repair this model response into valid JSON matching the schema. "
                "Preserve the intended verdict, reasoning, evidence_cited, and confidence_internal.\n\n"
                f"RAW RESPONSE:\n{raw}"
            )
        ),
    ]
    raw_retry, latency_retry, tokens_in_retry, tokens_out_retry = await call_agent(
        repair_prompt, role, client
    )
    try:
        parsed = parse_agent_output_strict(raw_retry, role_enum, round_num)
        return parsed, raw_retry, latency_retry, tokens_in_retry, tokens_out_retry
    except Exception:
        parsed = parse_agent_output(raw_retry, role_enum, round_num)
        return parsed, raw_retry, latency_retry, tokens_in_retry, tokens_out_retry


async def debate_round0_node(state: PipelineState) -> PipelineState:
    semaphore = asyncio.Semaphore(config.CLAIM_CONCURRENCY)
    agent_a_client = make_agent_client("agent_a")
    agent_b_client = make_agent_client("agent_b")

    async def run_claim(claim: Claim) -> None:
        async with semaphore:
            user_prompt = round0_user(claim, state.rag_chunks, state.user_query)
            a_prompt = [SystemMessage(content=AGENT_A_SYSTEM), HumanMessage(content=user_prompt)]
            b_prompt = [SystemMessage(content=AGENT_B_SYSTEM), HumanMessage(content=user_prompt)]
            try:
                a_result, b_result = await asyncio.gather(
                    _call_with_retry(a_prompt, "agent_a", AgentRole.AGENT_A, 0, agent_a_client),
                    _call_with_retry(b_prompt, "agent_b", AgentRole.AGENT_B, 0, agent_b_client),
                )
                a_out, a_raw, a_latency, a_in, a_out_tokens = a_result
                b_out, b_raw, b_latency, b_in, b_out_tokens = b_result
            except ValueError:
                logger.exception("Skipping claim %s after invalid agent JSON", claim.claim_id)
                return

            with connect(config.SQLITE_DB_PATH) as conn:
                insert_agent_output(conn, a_out, claim.claim_id, a_raw, a_latency, a_in, a_out_tokens)
                insert_agent_output(conn, b_out, claim.claim_id, b_raw, b_latency, b_in, b_out_tokens)

            state.agent_outputs_by_claim[claim.claim_id] = {
                "agent_a": {0: a_out},
                "agent_b": {0: b_out},
            }

    await asyncio.gather(*(run_claim(claim) for claim in state.claims))
    return state
