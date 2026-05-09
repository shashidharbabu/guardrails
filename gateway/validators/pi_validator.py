"""
Custom Prompt Injection Validator - wraps meta-llama/Llama-Prompt-Guard-2-86M.

Model is mDeBERTa-v3 SequenceClassifier (86M params) with two labels:
  BENIGN    (0) — safe input
  MALICIOUS (1) — jailbreak or prompt injection attempt

Returns pi_score = probability of MALICIOUS class (raw model confidence, 0.0–1.0).

Requires HuggingFace authentication:
  Set HF_TOKEN env var (or in .env at repo root).

Model path: set env var PROMPT_INJECTION_MODEL_PATH  (or HuggingFace Hub ID)
  Default: meta-llama/Llama-Prompt-Guard-2-86M
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


@register_validator(name="custom-pi-classifier", data_type="string")
class CustomPIValidator(Validator):
    """
    Wraps Llama Prompt Guard 2 86M as a guardrails-ai Validator.

    Detects both prompt injection AND jailbreak attempts (model labels both MALICIOUS).
    Returns pi_score regardless of pass/fail — DecisionEngine reads it for composite
    scoring and hard-override checks.
    """

    LABEL_MAP = {
        "BENIGN": "safe",
        "SAFE": "safe",
        "LABEL_0": "safe",
        "MALICIOUS": "prompt_injection",
        "INJECTION": "prompt_injection",
        "LABEL_1": "prompt_injection",
    }

    _PIPELINE_CACHE: ClassVar[Dict[str, Any]] = {}

    def __init__(
        self,
        model_path: str = None,
        pi_threshold: float = 0.4,
        on_fail: Optional[Callable] = None,
    ):
        if model_path is None:
            model_path = os.environ.get(
                "PROMPT_INJECTION_MODEL_PATH",
                "meta-llama/Llama-Prompt-Guard-2-86M",
            )
        super().__init__(
            on_fail=on_fail,
            model_path=model_path,
            pi_threshold=pi_threshold,
        )
        self.model_path = model_path
        self.pi_threshold = pi_threshold

    def _load_model(self):
        if self.model_path not in CustomPIValidator._PIPELINE_CACHE:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            print(f"[PI Validator] Loading model from: {self.model_path}")
            CustomPIValidator._PIPELINE_CACHE[self.model_path] = pipeline(
                task="text-classification",
                model=self.model_path,
                tokenizer=self.model_path,
                top_k=None,
                device="cpu",
                truncation=True,
                max_length=512,
                token=hf_token,
            )
            print("[PI Validator] Model ready")

    @property
    def _pipe(self):
        return CustomPIValidator._PIPELINE_CACHE.get(self.model_path)

    def _get_pi_score(self, pipeline_output: list) -> float:
        """Extract prompt injection / malicious class probability."""
        raw = pipeline_output[0] if isinstance(pipeline_output[0], list) else pipeline_output
        for item in raw:
            internal = self.LABEL_MAP.get(item["label"].upper())
            if internal == "prompt_injection":
                return float(item["score"])
        return 0.0

    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        self._load_model()

        output = self._pipe(value)
        pi_score = self._get_pi_score(output)
        safe_score = max(0.0, 1.0 - pi_score)

        # Always return PassResult so guardrails doesn't short-circuit the chain.
        # DecisionEngine reads scores from metadata and makes the final routing decision.
        return PassResult(
            metadata={
                "pi_score":   round(pi_score, 4),
                "safe_score": round(safe_score, 4),
                "validator":  "custom-pi-classifier",
            }
        )
