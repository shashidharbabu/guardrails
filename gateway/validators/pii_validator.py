"""
PII Validator — calls HF Serverless Inference API (no local model loading).

Model: iiiorg/piiranha-v1-detect-personal-information
Task:  token-classification — returns entity spans with entity_group + score.

Set PII_MODEL_PATH to override the model repo ID.
Set HF_TOKEN for authenticated requests (higher rate limits).
"""

import logging
import os
from typing import Callable, Dict, List, Optional

import httpx
from guardrails.validators import PassResult, ValidationResult, Validator, register_validator

logger = logging.getLogger(__name__)

_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"
_DEFAULT_MODEL = "iiiorg/piiranha-v1-detect-personal-information"

_PII_ENTITY_GROUPS = {
    "FIRSTNAME", "LASTNAME", "NAME", "EMAIL", "PHONE", "SSN", "SOCIALNUMBER",
    "CREDITCARDNUMBER", "DATEOFBIRTH", "AGE", "ADDRESS", "CITY", "STATE",
    "ZIPCODE", "COUNTRY", "USERNAME", "PASSWORD", "IDNUMBER", "ACCOUNTNUMBER",
    "TAXNUMBER", "IPV4ADDRESS", "IPV6ADDRESS", "URL", "DRIVERLICENSE",
    "PASSPORT", "MEDICALNUMBER", "PER", "LOC", "ORG",
}


@register_validator(name="custom-pii-ner", data_type="string")
class CustomPIIValidator(Validator):
    """
    Calls HF Serverless Inference API for PII token-classification.
    No local model, no torch — pure HTTP.
    """

    def __init__(
        self,
        model_path: str = None,
        threshold: float = 0.5,
        on_fail: Optional[Callable] = None,
    ):
        model_path = model_path or os.environ.get("PII_MODEL_PATH", _DEFAULT_MODEL)
        super().__init__(on_fail=on_fail, model_path=model_path, threshold=threshold)
        self.model_path = model_path
        self.threshold = threshold
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
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("[PII Validator] API call failed: %s", exc)
            return []

    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        raw_entities = self._call_api(value)
        pii_entities: List[Dict] = []

        for ent in raw_entities:
            group = (ent.get("entity_group") or ent.get("entity") or "").upper()
            score = float(ent.get("score", 0.0))
            word = ent.get("word", "")
            if not word or score < self.threshold:
                continue
            if group not in _PII_ENTITY_GROUPS:
                continue
            pii_entities.append({
                "entity_type": group,
                "text": word,
                "confidence": round(score, 4),
                "start": ent.get("start"),
                "end": ent.get("end"),
            })

        pii_score = 1.0 if pii_entities else 0.0

        return PassResult(
            metadata={
                "pii_score": round(pii_score, 4),
                "pii_entities": pii_entities,
                "validator": "custom-pii-ner",
            }
        )
