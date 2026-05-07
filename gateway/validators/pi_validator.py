"""
Custom Prompt Injection Validator.

Default: Hugging Face text-classification (e.g. Llama-Prompt-Guard) via pipeline.

PEFT modes (set PI_USE_PEFT=true):
  - PI_PEFT_ARCH=classifier — base + LoRA as sequence classification (same as JB path).
  - PI_PEFT_ARCH=causal_lm — AutoModelForCausalLM + PeftModel (e.g. Qwen2.5-7B-Instruct +
    harshitasayala/pi-qwen25-7b). Scoring uses short generation + PI_CAUSAL_PROMPT_TEMPLATE.

Requires HF_TOKEN for gated models.
"""

import os
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple

import torch
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from transformers import pipeline

from gateway.hf_peft_loader import (
    load_causal_lm_peft,
    load_text_classification_peft_pipeline,
    score_pi_with_causal_peft,
)


@register_validator(name="custom-pi-classifier", data_type="string")
class CustomPIValidator(Validator):
    """
    Prompt injection / malicious-intent score for the decision engine.

    pi_score is always returned for composite scoring and hard-override checks.
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
    _CAUSAL_CACHE: ClassVar[Dict[str, Tuple[Any, Any]]] = {}

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
        self._use_peft = os.getenv("PI_USE_PEFT", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        self._peft_base = os.environ.get("PI_PEFT_BASE", "").strip() or None
        self._peft_arch = os.getenv("PI_PEFT_ARCH", "classifier").strip().lower()
        self._cache_key = (
            f"peft:pi:{self._peft_arch}:{model_path}:{self._peft_base or ''}"
            if self._use_peft
            else model_path
        )

    def _load_model(self) -> None:
        if self._use_peft and self._peft_arch == "causal_lm":
            if self._cache_key not in CustomPIValidator._CAUSAL_CACHE:
                print(f"[PI Validator] Loading causal PEFT: {self.model_path!r}")
                model, tokenizer = load_causal_lm_peft(
                    self.model_path,
                    explicit_base=self._peft_base,
                )
                CustomPIValidator._CAUSAL_CACHE[self._cache_key] = (model, tokenizer)
                print("[PI Validator] Causal PEFT model ready")
            return

        if self._cache_key not in CustomPIValidator._PIPELINE_CACHE:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            print(f"[PI Validator] Loading model from: {self.model_path}")
            if self._use_peft and self._peft_arch == "classifier":
                CustomPIValidator._PIPELINE_CACHE[self._cache_key] = (
                    load_text_classification_peft_pipeline(
                        self.model_path,
                        explicit_base=self._peft_base,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                )
            else:
                CustomPIValidator._PIPELINE_CACHE[self._cache_key] = pipeline(
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
        return CustomPIValidator._PIPELINE_CACHE.get(self._cache_key)

    def _get_pi_score(self, pipeline_output: list) -> float:
        """Extract prompt injection / malicious class probability."""
        raw = pipeline_output[0] if isinstance(pipeline_output[0], list) else pipeline_output
        for item in raw:
            internal = self.LABEL_MAP.get(item["label"].upper())
            if internal == "prompt_injection":
                return float(item["score"])
        return 0.0

    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        self._load_model()

        if self._use_peft and self._peft_arch == "causal_lm":
            model, tokenizer = CustomPIValidator._CAUSAL_CACHE[self._cache_key]
            pi_score = float(score_pi_with_causal_peft(model, tokenizer, value))
        else:
            output = self._pipe(value)
            pi_score = self._get_pi_score(output)

        safe_score = max(0.0, 1.0 - pi_score)

        if pi_score < self.pi_threshold:
            return PassResult(
                metadata={
                    "pi_score": round(pi_score, 4),
                    "safe_score": round(safe_score, 4),
                    "validator": "custom-pi-classifier",
                }
            )

        return FailResult(
            error_message=f"Prompt injection / malicious intent detected (confidence={pi_score:.3f})",
            fix_value=value,
            metadata={
                "pi_score": round(pi_score, 4),
                "safe_score": round(safe_score, 4),
                "validator": "custom-pi-classifier",
            },
        )
