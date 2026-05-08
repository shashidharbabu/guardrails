from __future__ import annotations
import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from configs import config
from src.agents.parser import parse_agent_output, parse_agent_output_strict
from src.db.db import connect, insert_agent_delta, insert_agent_output
from src.schemas.schemas import AgentOutputFull, AgentRole, Claim, PipelineState, strip_for_peer
from src.utils.prompts import AGENT_A_SYSTEM, AGENT_B_SYSTEM, round1_user
from src.utils.vllm_client import call_agent, make_agent_client


logger = logging.getLogger(__name__)

REPAIR_SYSTEM = (
    "Return ONLY valid JSON matching the required schema. No markdown. "
    "Quote all keys. Use an array for evidence_cited."
)


async def _call_with_retry(prompt, role: str, role_enum: AgentRole, round_num: int, client):
    raw, latency_ms, tokens_in, tokens_out = await call_agent(prompt, role, client)
    try:
        parsed = parse_agent_output_strict(raw, role_enum, round_num)
        return parsed, raw, latency_ms, tokens_in, tokens_out
    except Exception:
        pass
    repair_prompt = [
        SystemMessage(content=REPAIR_SYSTEM),
        HumanMessage(content=f"Repair this into valid JSON:\n{raw}"),
    ]
    raw_retry, latency_retry, t_in, t_out = await call_agent(repair_prompt, role, client)
    parsed = parse_agent_output(raw_retry, role_enum, round_num)
    return parsed, raw_retry, latency_retry, t_in, t_out


def _round0_output(outputs_by_role: dict, role_name: str, claim_id: str) -> AgentOutputFull | None:
    role_outputs = outputs_by_role.get(role_name)
    if not isinstance(role_outputs, dict):
        return None
    output = role_outputs.get(0)
    if not isinstance(output, AgentOutputFull):
        return None
    return output


async def debate_round1_node(state: PipelineState) -> PipelineState:
    semaphore = asyncio.Semaphore(config.CLAIM_CONCURRENCY)
    agent_a_client = make_agent_client("agent_a")
    agent_b_client = make_agent_client("agent_b")

    async def run_claim(claim: Claim) -> None:
        async with semaphore:
            outputs_by_role = state.agent_outputs_by_claim.get(claim.claim_id)
            if not isinstance(outputs_by_role, dict):
                logger.error("Missing Round 0 outputs for claim %s", claim.claim_id)
                return
            a_r0 = _round0_output(outputs_by_role, "agent_a", claim.claim_id)
            b_r0 = _round0_output(outputs_by_role, "agent_b", claim.claim_id)
            if a_r0 is None or b_r0 is None:
                return

            a_stripped = strip_for_peer(a_r0, "Debater 1")
            b_stripped = strip_for_peer(b_r0, "Debater 2")
            a_prompt = [
                SystemMessage(content=AGENT_A_SYSTEM),
                HumanMessage(content=round1_user(claim, state.rag_chunks, state.user_query, b_stripped)),
            ]
            b_prompt = [
                SystemMessage(content=AGENT_B_SYSTEM),
                HumanMessage(content=round1_user(claim, state.rag_chunks, state.user_query, a_stripped)),
            ]
            try:
                a_result, b_result = await asyncio.gather(
                    _call_with_retry(a_prompt, "agent_a", AgentRole.AGENT_A, 1, agent_a_client),
                    _call_with_retry(b_prompt, "agent_b", AgentRole.AGENT_B, 1, agent_b_client),
                )
                a_out, a_raw, a_latency, a_in, a_out_tokens = a_result
                b_out, b_raw, b_latency, b_in, b_out_tokens = b_result
            except ValueError:
                logger.exception("Skipping claim %s after invalid Round 1 agent JSON", claim.claim_id)
                return

            with connect(config.SQLITE_DB_PATH) as conn:
                insert_agent_output(conn, a_out, claim.claim_id, a_raw, a_latency, a_in, a_out_tokens)
                insert_agent_output(conn, b_out, claim.claim_id, b_raw, b_latency, b_in, b_out_tokens)
                insert_agent_delta(conn, claim.claim_id, AgentRole.AGENT_A, a_r0, a_out)
                insert_agent_delta(conn, claim.claim_id, AgentRole.AGENT_B, b_r0, b_out)

            outputs_by_role["agent_a"][1] = a_out
            outputs_by_role["agent_b"][1] = b_out

    await asyncio.gather(*(run_claim(claim) for claim in state.claims))
    return state
