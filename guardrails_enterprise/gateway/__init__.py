from __future__ import annotations


def __getattr__(name: str):
    if name == "Gateway" or name == "GuardrailGateway":
        from gateway.gateway import GuardrailGateway
        return GuardrailGateway
    if name == "DecisionEngine":
        from gateway.decision_engine import DecisionEngine
        return DecisionEngine
    if name == "GatewayResult":
        from gateway.decision_engine import GatewayResult
        return GatewayResult
    raise AttributeError(f"module 'guardrails_enterprise.gateway' has no attribute {name!r}")


__all__ = ["Gateway", "GuardrailGateway", "DecisionEngine", "GatewayResult"]
