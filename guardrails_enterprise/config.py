"""
guardrails_enterprise.config — SDK configuration dataclass.

All settings can be overridden via environment variables or constructor args.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class SDKConfig:
    """
    Central configuration for the Guardrails Enterprise SDK.

    Usage::

        from guardrails_enterprise import SDKConfig, GuardrailPipeline

        cfg = SDKConfig(llm_model="qwen2.5:14b", raise_on_block=True)
        pipeline = GuardrailPipeline(config=cfg)
    """

    # ── Gateway ─────────────────────────────────────────────────────────────────
    gateway_url: str = field(
        default_factory=lambda: os.getenv("GATEWAY_URL", "http://localhost:8080")
    )

    # ── LLM (Ollama OpenAI-compat endpoint) ─────────────────────────────────────
    ollama_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("AGENT_MODEL", "qwen2.5:7b")
    )
    llm_system_prompt: str = (
        "You are an enterprise compliance assistant. Answer questions about regulatory "
        "requirements accurately and concisely based on your knowledge of healthcare, "
        "banking, and legal regulations. Be concise — 2-4 sentences maximum."
    )
    llm_max_tokens: int = 512
    llm_timeout_s: float = 300.0

    # ── Gateway timeouts ────────────────────────────────────────────────────────
    gateway_timeout_s: float = 60.0

    # ── MAD ─────────────────────────────────────────────────────────────────────
    run_mad: bool = True          # set False to skip output verification
    mad_in_background: bool = True  # True = MAD runs async after returning LLM answer

    # ── RAG (retrieval + SLM verification) ───────────────────────────────────────
    # When run_rag=True, pipeline.retrieve() and pipeline.retrieve_and_verify()
    # use the real Qdrant + Nemotron-8B + SLM verifier pipeline.
    # When False, falls back to TF-IDF stub (no Qdrant required).
    run_rag: bool = True
    rag_top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_RETRIEVE", "7"))
    )
    # Qdrant connection — pulled from env vars by default (never hardcode keys)
    qdrant_url: str = field(
        default_factory=lambda: os.getenv(
            "QDRANT_URL",
            "https://e2e7b7d2-4927-4c61-a78d-61f9c4e024bb.us-east4-0.gcp.cloud.qdrant.io",
        )
    )
    qdrant_api_key: str = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY", "")
    )
    # SLM verifier model (via Ollama — same instance as LLM)
    verifier_model: str = field(
        default_factory=lambda: os.getenv("VERIFIER_MODEL", "qwen2.5:7b")
    )

    # ── Behaviour ────────────────────────────────────────────────────────────────
    raise_on_block: bool = False  # if True, raises GatewayBlockedError instead of returning result
