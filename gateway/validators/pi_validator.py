"""
Prompt Injection Validator — calls HF Serverless Inference API (no local model loading).

Model: protectai/deberta-v3-base-prompt-injection-v2
Task:  text-classification — returns [{label: INJECTION|SAFE, score: float}].

Set PROMPT_INJECTION_MODEL_PATH to override the model repo ID.
Set HF_TOKEN for authenticated requests (higher rate limits).
"""

import logging
import os
from typing import Callable, Dict, List, Optional

import httpx
from guardrails.validators import PassResult, ValidationResult, Validator, register_validator

logger = logging.getLogger(__name__)

_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"
_DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"

_INJECTION_LABELS = {"INJECTION", "MALICIOUS", "JAILBREAK", "LABEL_1", "PROMPT_INJECTION"}


@register_validator(name="custom-pi-classifier", data_type="string")
class CustomPIValidator(Validator):
    """
    Calls HF Serverless Inference API for prompt injection classification.
    No local model, no torch — pure HTTP.
    """

    def __init__(
        self,
        model_path: str = None,
        pi_threshold: float = 0.4,
        on_fail: Optional[Callable] = None,
    ):
        model_path = model_path or os.environ.get("PROMPT_INJECTION_MODEL_PATH", _DEFAULT_MODEL)
        super().__init__(on_fail=on_fail, model_path=model_path, pi_threshold=pi_threshold)
        self.model_path = model_path
        self.pi_threshold = pi_threshold
        self._hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
        self._url = f"{_HF_API_BASE}/{self.model_path}"

    def _call_api(self, text: str) -> List[Dict]:
        headers = {"Content-Type": "application/json"}
        if self._hf_token:
            headers["Authorization"] = f"Bearer {self._hf_token}"
        try:
            resp = httpx.post(
                self._url,
                json={"inputs": text},
                headers=headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data and isinstance(data[0], list):
                return data[0]
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("[PI Validator] API call failed: %s", exc)
            return []

    def _get_pi_score(self, results: List[Dict]) -> float:
        for item in results:
            if item.get("label", "").upper() in _INJECTION_LABELS:
                return float(item["score"])
        return 0.0

    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        results = self._call_api(value)
        pi_score = self._get_pi_score(results)

        return PassResult(
            metadata={
                "pi_score": round(pi_score, 4),
                "safe_score": round(max(0.0, 1.0 - pi_score), 4),
                "validator": "custom-pi-classifier",
            }
        )
