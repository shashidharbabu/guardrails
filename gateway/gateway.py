"""
GuardrailGateway - main entry point for all requests.

Wires together:
  1. Three guardrails-ai validators (PII, JailBreak, Prompt Injection)
  2. DecisionEngine (composite scoring + hard overrides + routing)
  3. Logger (SQLite audit trail)

Models are lazy-loaded on first .process() call.
Environment variables (loaded from .env at repo root):
  PII_MODEL_PATH               — NER PII model (or LoRA adapter id if PII_USE_PEFT=true)
  THREAT_MODEL_PATH            — JailBreak classifier (or adapter if THREAT_USE_PEFT=true)
  PROMPT_INJECTION_MODEL_PATH — Prompt-injection model or adapter (see PI_USE_PEFT / PI_PEFT_ARCH)
  HF_TOKEN                     — HuggingFace auth token for gated / private repos
  PII_USE_PEFT, THREAT_USE_PEFT, PI_USE_PEFT — load base+adapter via PEFT (see .env.example)
"""

import os
from typing import Optional

from dotenv import load_dotenv
from guardrails import Guard

from gateway import logger as event_logger
from gateway import telemetry as gw_telemetry
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

        # guardrails-ai 0.6+: use Guard.for_string + parse(); .use() no longer accepts multiple validators.
        self._guard = Guard.for_string(
            [
                self._pii_validator,
                self._threat_validator,
                self._pi_validator,
            ],
        )

        self._engine = DecisionEngine(
            pass_threshold=pass_threshold,
            block_threshold=block_threshold,
            jb_override_threshold=jb_override_threshold,
            pi_override_threshold=pi_override_threshold,
            pii_override_threshold=pii_override_threshold,
        )

        event_logger.init_db()

    def process(self, user_input: str, trace_id: Optional[str] = None) -> GatewayResult:
        """
        Run user input through the full gateway pipeline.
        Always returns a GatewayResult — never raises.

        trace_id — optional correlator propagated from HTTP/session for Datadog/JSON logs.
        """
        tid = trace_id
        with gw_telemetry.span("gateway.process", trace_id=tid, text_len=len(user_input)):
            pii_score = 0.0
            jb_score = 0.0
            pi_score = 0.0
            pii_entities = []

            # Local/docker smoke tests should not depend on downloading large HF models.
            # When enabled, we skip guardrails validators and emit deterministic scores.
            if os.getenv("GATEWAY_STUB_VALIDATORS", "false").lower() in ("1", "true", "yes"):
                text = user_input.lower()
                jb_score = 0.9 if "jailbreak" in text or "ignore previous" in text else 0.05
                pi_score = 0.85 if "prompt injection" in text or "system prompt" in text else 0.05
                pii_score = 0.95 if "ssn" in text or "social security" in text else 0.02
                pii_entities = [{"entity_type": "SSN", "text": "stub", "confidence": 0.95}] if pii_score >= 0.9 else []
                gw_telemetry.tag_current_span(
                    stub_mode=True,
                    pii_score=pii_score,
                    jb_score=jb_score,
                    pi_score=pi_score,
                    pii_entity_count=len(pii_entities),
                )
                result = self._engine.decide(
                    raw_input=user_input,
                    pii_score=pii_score,
                    jb_score=jb_score,
                    pi_score=pi_score,
                    pii_entities=pii_entities,
                )
                event_logger.log_event(result)
                return result

            with gw_telemetry.span("gateway.validators", trace_id=tid):
                try:
                    self._guard.parse(user_input)
                except Exception as exc:
                    print(f"[Gateway] Warning: validation error: {exc}")
                    gw_telemetry.tag_current_span(validator_error=str(exc))

            with gw_telemetry.span("gateway.score_extraction", trace_id=tid):
                try:
                    last_call = self._guard.history[-1]
                    validator_logs = getattr(last_call, "validator_logs", None)
                    if validator_logs is None:
                        validator_logs = getattr(last_call.inputs, "validator_logs", [])

                    for vlog in validator_logs:
                        vname = (getattr(vlog, "validator_name", "") or "").lower()
                        result = getattr(vlog, "validation_result", None)
                        meta = getattr(result, "metadata", {}) or {}
                        validator_id = str(meta.get("validator", "")).lower()

                        if (
                            "custompiivalidator" in vname
                            or "custom-pii-ner" in vname
                            or validator_id == "custom-pii-ner"
                        ):
                            pii_score = float(meta.get("pii_score", 0.0))
                            pii_entities = meta.get("pii_entities", [])

                        elif (
                            "customthreatvalidator" in vname
                            or "custom-threat-classifier" in vname
                            or validator_id == "custom-threat-classifier"
                        ):
                            jb_score = float(meta.get("jb_score", 0.0))

                        elif (
                            "custompivalidator" in vname
                            or "custom-pi-classifier" in vname
                            or validator_id == "custom-pi-classifier"
                        ):
                            pi_score = float(meta.get("pi_score", 0.0))

                except (IndexError, AttributeError) as exc:
                    print(f"[Gateway] Warning: could not read validator scores: {exc}")
                    gw_telemetry.tag_current_span(score_extraction_error=str(exc))

                gw_telemetry.tag_current_span(
                    pii_score=pii_score,
                    jb_score=jb_score,
                    pi_score=pi_score,
                    pii_entity_count=len(pii_entities),
                )

            with gw_telemetry.span("gateway.decision_engine", trace_id=tid):
                result = self._engine.decide(
                    raw_input=user_input,
                    pii_score=pii_score,
                    jb_score=jb_score,
                    pi_score=pi_score,
                    pii_entities=pii_entities,
                )

            with gw_telemetry.span("gateway.audit_log", trace_id=tid):
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
