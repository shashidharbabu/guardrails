import os
import time
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


AgentRoleName = Literal["agent_a", "agent_b"]


def _endpoint(env_name: str, default: str) -> str:
    return os.getenv(env_name, default)


def _make_client(
    *,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int = 1024,
) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=base_url,
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def make_baseline_client() -> ChatOpenAI:
    from configs import config
    return _make_client(
        base_url=_endpoint("VLLM_BASELINE_URL", config.VLLM_BASELINE_URL),
        model=config.BASELINE_MODEL_NAME,
        temperature=0.0,
    )


def make_decomposer_client() -> ChatOpenAI:
    from configs import config
    return _make_client(
        base_url=_endpoint("VLLM_DECOMPOSER_URL", config.VLLM_DECOMPOSER_URL),
        model=config.DECOMPOSER_MODEL_NAME,
        temperature=0.0,
        max_tokens=2048,
    )


def make_agent_client(role: AgentRoleName = "agent_a", max_tokens: int = 384) -> ChatOpenAI:
    if role not in ("agent_a", "agent_b"):
        raise ValueError("role must be 'agent_a' or 'agent_b'")
    from configs import config
    return _make_client(
        base_url=_endpoint("VLLM_AGENTS_URL", config.VLLM_AGENTS_URL),
        model=config.AGENTS_MODEL_NAME,
        temperature=0.4 if role == "agent_a" else 0.7,
        max_tokens=max_tokens,
    )


def make_judge_client() -> ChatOpenAI:
    from configs import config
    return _make_client(
        base_url=_endpoint("VLLM_JUDGE_URL", config.VLLM_JUDGE_URL),
        model=config.JUDGE_MODEL_NAME,
        temperature=0.0,
    )


def _message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts = []
        for item in message.content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(message.content)


def _usage_counts(message: AIMessage) -> tuple[int, int]:
    usage_metadata = getattr(message, "usage_metadata", None) or {}
    if usage_metadata:
        return (
            int(usage_metadata.get("input_tokens") or 0),
            int(usage_metadata.get("output_tokens") or 0),
        )
    response_metadata: dict[str, Any] = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or {}
    return (
        int(token_usage.get("prompt_tokens") or 0),
        int(token_usage.get("completion_tokens") or 0),
    )


async def call_agent(
    prompt,
    role: AgentRoleName,
    client: ChatOpenAI,
) -> tuple[str, int, int, int]:
    if role not in ("agent_a", "agent_b"):
        raise ValueError("role must be 'agent_a' or 'agent_b'")
    start = time.perf_counter()
    message = await client.ainvoke(prompt, config={"cache": False})
    latency_ms = int((time.perf_counter() - start) * 1000)
    tokens_in, tokens_out = _usage_counts(message)
    return _message_text(message), latency_ms, tokens_in, tokens_out
