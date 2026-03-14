"""
Custom JailBreak Validator - wraps shashidharbabu/roberta-jailbreak-guardrails.

Model is RobertaForSequenceClassification with two labels:
  benign    (0) — safe input
  jailbreak (1) — jailbreak attempt

Returns jb_score = probability of jailbreak class (raw model confidence, 0.0–1.0).
Prompt injection is handled separately by CustomPIValidator (pi_validator.py).

Model path: gateway/models/threat_classifier_model/
  OR set env var: THREAT_MODEL_PATH (or HuggingFace Hub ID)
"""

import os
from typing import Any, Callable, ClassVar, Dict, Optional

from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from transformers import pipeline


@register_validator(name="custom-threat-classifier", data_type="string")
class CustomThreatValidator(Validator):
    """
    Wraps our finetuned JailBreak classifier as a guardrails-ai Validator.

    Returns jb_score regardless of pass/fail — DecisionEngine reads it for
    composite scoring and hard-override checks.

    LABEL_MAP covers all known label formats from this model family.
    """

    LABEL_MAP = {
        "BENIGN": "safe",
        "SAFE": "safe",
        "LABEL_0": "safe",
        "JAILBREAK": "jailbreak",
        "LABEL_1": "jailbreak",
        "JB": "jailbreak",
        "MALICIOUS": "jailbreak",
    }

    _PIPELINE_CACHE: ClassVar[Dict[str, Any]] = {}

    def __init__(
        self,
        model_path: str = None,
        jb_threshold: float = 0.4,
        on_fail: Optional[Callable] = None,
    ):
        if model_path is None:
            model_path = os.environ.get(
                "THREAT_MODEL_PATH",
                os.path.join(
                    os.path.dirname(__file__), "..", "models", "threat_classifier_model"
                ),
            )
        super().__init__(
            on_fail=on_fail,
            model_path=model_path,
            jb_threshold=jb_threshold,
        )
        self.model_path = model_path
        self.jb_threshold = jb_threshold

    def _load_model(self):
        if self.model_path not in CustomThreatValidator._PIPELINE_CACHE:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            print(f"[JB Validator] Loading model from: {self.model_path}")
            CustomThreatValidator._PIPELINE_CACHE[self.model_path] = pipeline(
                task="text-classification",
                model=self.model_path,
                tokenizer=self.model_path,
                top_k=None,
                device="cpu",
                truncation=True,
                max_length=512,
                token=hf_token,
            )
            print("[JB Validator] Model ready")

    @property
    def _pipe(self):
        return CustomThreatValidator._PIPELINE_CACHE.get(self.model_path)

    def _get_jb_score(self, pipeline_output: list) -> float:
        """Extract jailbreak class probability from pipeline output."""
        raw = pipeline_output[0] if isinstance(pipeline_output[0], list) else pipeline_output
        for item in raw:
            internal = self.LABEL_MAP.get(item["label"].upper())
            if internal == "jailbreak":
                return float(item["score"])
        return 0.0

    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        self._load_model()

        output = self._pipe(value)
        jb_score = self._get_jb_score(output)
        safe_score = max(0.0, 1.0 - jb_score)

        if jb_score < self.jb_threshold:
            return PassResult(
                metadata={
                    "jb_score": round(jb_score, 4),
                    "safe_score": round(safe_score, 4),
                    "validator": "custom-threat-classifier",
                }
            )

        return FailResult(
            error_message=f"Jailbreak detected (confidence={jb_score:.3f})",
            fix_value=value,
            metadata={
                "jb_score": round(jb_score, 4),
                "safe_score": round(safe_score, 4),
                "validator": "custom-threat-classifier",
            },
        )
