"""
Threat (Jailbreak) Validator — calls HF Serverless Inference API (no local model loading).

Model: jackhhao/jailbreak-classifier
Task:  text-classification — returns [{label: jailbreak|benign, score: float}].

Set THREAT_MODEL_PATH to override the model repo ID.
Set HF_TOKEN for authenticated requests (higher rate limits).
"""

import logging
import os
from typing import Callable, Dict, List, Optional

import httpx
from guardrails.validators import PassResult, ValidationResult, Validator, register_validator

logger = logging.getLogger(__name__)

_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"
_DEFAULT_MODEL = "jackhhao/jailbreak-classifier"

_JAILBREAK_LABELS = {"JAILBREAK", "INJECTION", "MALICIOUS", "LABEL_1", "JB"}


@register_validator(name="custom-threat-classifier", data_type="string")
class CustomThreatValidator(Validator):
    """
    Calls HF Serverless Inference API for jailbreak classification.
    No local model, no torch — pure HTTP.
    """

    def __init__(
        self,
        model_path: str = None,
        jb_threshold: float = 0.4,
        on_fail: Optional[Callable] = None,
    ):
        model_path = model_path or os.environ.get("THREAT_MODEL_PATH", _DEFAULT_MODEL)
        super().__init__(on_fail=on_fail, model_path=model_path, jb_threshold=jb_threshold)
        self.model_path = model_path
        self.jb_threshold = jb_threshold
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
            logger.warning("[Threat Validator] API call failed: %s", exc)
            raise RuntimeError("threat validator unavailable") from exc

    def _get_jb_score(self, results: List[Dict]) -> float:
        for item in results:
            if item.get("label", "").upper() in _JAILBREAK_LABELS:
                return float(item["score"])
        return 0.0

    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        results = self._call_api(value)
        jb_score = self._get_jb_score(results)

        return PassResult(
            metadata={
                "jb_score": round(jb_score, 4),
                "safe_score": round(max(0.0, 1.0 - jb_score), 4),
                "validator": "custom-threat-classifier",
            }
        )
