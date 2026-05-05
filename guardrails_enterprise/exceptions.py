"""
guardrails_enterprise.exceptions — SDK exception hierarchy.
"""


class GuardrailError(Exception):
    """Base exception for all guardrails_enterprise errors."""


class GatewayBlockedError(GuardrailError):
    """Raised when the input gateway blocks a request."""

    def __init__(self, decision: str, score: float, reason: str | None = None):
        self.decision = decision
        self.score = score
        self.reason = reason
        super().__init__(
            f"Gateway {decision} (score={score:.3f})"
            + (f": {reason}" if reason else "")
        )


class MADError(GuardrailError):
    """Raised when the MAD pipeline encounters an unrecoverable error."""


class LLMError(GuardrailError):
    """Raised when the downstream LLM call fails."""


class GatewayConnectionError(GuardrailError):
    """Raised when the gateway service is unreachable."""


class RAGError(GuardrailError):
    """Raised when the RAG retrieval or SLM verification pipeline fails."""
