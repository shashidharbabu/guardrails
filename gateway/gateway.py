"""
GuardrailGateway - main entry point for all requests.

Wires together:
  1. Three guardrails-ai validators (PII, JailBreak, Prompt Injection)
  2. DecisionEngine (composite scoring + hard overrides + routing)
  3. Logger (SQLite audit trail)

Models are lazy-loaded on first .process() call.
Environment variables (loaded from .env at repo root):
  PII_MODEL_PATH               — NER PII model
  THREAT_MODEL_PATH            — JailBreak binary classifier
  PROMPT_INJECTION_MODEL_PATH  — Llama Prompt Guard 2 86M
  HF_TOKEN                     — HuggingFace auth token (required for Llama Prompt Guard)
"""

import os

from dotenv import load_dotenv
from guardrails import Guard

from gateway import logger as event_logger
from gateway.decision_engine import Decision, DecisionEngine, GatewayResult
from gateway.validators.pi_validator import CustomPIValidator
from gateway.validators.pii_validator import CustomPIIValidator
from gateway.validators.threat_validator import CustomThreatValidator

# Load .env from repo root (parent of this file's directory)
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(_env_path)


class GuardrailGateway:
    """
    Enterprise AI gateway — scans all user inputs before they reach downstream systems.

    Three validators run sequentially:
      CustomPIIValidator     — NER-based PII detection
      CustomThreatValidator  — binary jailbreak classifier
      CustomPIValidator      — Llama Prompt Guard 2 (prompt injection / malicious intent)
    """

    def __init__(
        self,
        pii_model_path: str = None,
        threat_model_path: str = None,
        pi_model_path: str = None,
        pii_threshold: float = 0.5,
        jb_threshold: float = 0.4,
        pi_threshold: float = 0.4,
        pass_threshold: float = 0.3,
        block_threshold: float = 0.7,
        jb_override_threshold: float = 0.7,
        pi_override_threshold: float = 0.7,
        pii_override_threshold: float = 0.9,
    ):
        base_dir = os.path.dirname(__file__)
        pii_model_path = pii_model_path or os.environ.get(
            "PII_MODEL_PATH", os.path.join(base_dir, "models", "pii_ner_model")
        )
        threat_model_path = threat_model_path or os.environ.get(
            "THREAT_MODEL_PATH",
            os.path.join(base_dir, "models", "threat_classifier_model"),
        )
        pi_model_path = pi_model_path or os.environ.get(
            "PROMPT_INJECTION_MODEL_PATH",
            "meta-llama/Llama-Prompt-Guard-2-86M",
        )

        self._pii_validator = CustomPIIValidator(
            model_path=pii_model_path,
            threshold=pii_threshold,
            on_fail="noop",
        )
        self._threat_validator = CustomThreatValidator(
            model_path=threat_model_path,
            jb_threshold=jb_threshold,
            on_fail="noop",
        )
        self._pi_validator = CustomPIValidator(
            model_path=pi_model_path,
            pi_threshold=pi_threshold,
            on_fail="noop",
        )

        self._guard = (
            Guard()
            .use(self._pii_validator)
            .use(self._threat_validator)
            .use(self._pi_validator)
        )

        self._engine = DecisionEngine(
            pass_threshold=pass_threshold,
            block_threshold=block_threshold,
            jb_override_threshold=jb_override_threshold,
            pi_override_threshold=pi_override_threshold,
            pii_override_threshold=pii_override_threshold,
        )

        event_logger.init_db()

    def process(self, user_input: str) -> GatewayResult:
        """
        Run user input through the full gateway pipeline.
        Calls each validator directly to collect metadata reliably.
        Always returns a GatewayResult — never raises.
        """
        pii_score = 0.0
        jb_score = 0.0
        pi_score = 0.0
        pii_entities = []
        validation_error = None

        # Call each validator directly — Guard history is unreliable for multi-validator PassResult metadata
        try:
            pii_result = self._pii_validator.validate(user_input, {})
            pii_meta = getattr(pii_result, "metadata", {}) or {}
            pii_score = float(pii_meta.get("pii_score", 0.0))
            pii_entities = pii_meta.get("pii_entities", [])
        except Exception as exc:
            print(f"[Gateway] PII validator error: {exc}")
            validation_error = str(exc)

        try:
            jb_result = self._threat_validator.validate(user_input, {})
            jb_meta = getattr(jb_result, "metadata", {}) or {}
            jb_score = float(jb_meta.get("jb_score", 0.0))
        except Exception as exc:
            print(f"[Gateway] JB validator error: {exc}")
            if not validation_error:
                validation_error = str(exc)

        try:
            pi_result = self._pi_validator.validate(user_input, {})
            pi_meta = getattr(pi_result, "metadata", {}) or {}
            pi_score = float(pi_meta.get("pi_score", 0.0))
        except Exception as exc:
            print(f"[Gateway] PI validator error: {exc}")
            if not validation_error:
                validation_error = str(exc)

        result = self._engine.decide(
            raw_input=user_input,
            pii_score=pii_score,
            jb_score=jb_score,
            pi_score=pi_score,
            pii_entities=pii_entities,
        )

        if validation_error and result.decision == Decision.PASS:
            result = GatewayResult(
                decision=Decision.ESCALATE,
                gateway_score=max(result.gateway_score, self._engine.pass_threshold),
                pii_score=pii_score,
                jb_score=jb_score,
                pi_score=pi_score,
                pii_entities=pii_entities,
                threat_types=["VALIDATOR_ERROR"],
                blocked_reason=(
                    "Gateway validator unavailable; routed to analyst review instead of passing. "
                    f"Error: {validation_error[:240]}"
                ),
                raw_input=user_input,
            )

        event_logger.log_event(result)
        return result

    def summary(self, result: GatewayResult) -> str:
        """One-line human-readable summary for logs and demos."""
        markers = {
            Decision.PASS: "[PASS]",
            Decision.ESCALATE: "[ESCALATE]",
            Decision.BLOCK: "[BLOCK]",
        }
        marker = markers[result.decision]
        return (
            f"{marker} {result.decision.value} "
            f"[score={result.gateway_score:.3f} | "
            f"pii={result.pii_score:.2f} jb={result.jb_score:.2f} pi={result.pi_score:.2f}]"
            + (f" - {result.blocked_reason}" if result.blocked_reason else "")
        )
