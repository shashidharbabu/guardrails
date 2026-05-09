"""
Custom PII Validator — supports two modes controlled by PII_INFERENCE_URL env var:

  LOCAL mode (default, dev):
    Loads vineeth453/qwen25-7b-pii-detection-lora or any HF NER model in-process.
    Set PII_MODEL_PATH to override the model.

  REMOTE mode (production):
    Set PII_INFERENCE_URL=http://<host>:<port> to call a HuggingFace TEI or
    custom inference server's /token-classification endpoint instead.
    No model is loaded locally — the container stays lean.

HuggingFace TEI server command (on GPU EC2):
  docker run -p 3000:80 ghcr.io/huggingface/text-embeddings-inference:latest \
    --model-id vineeth453/qwen25-7b-pii-detection-lora \
    --dtype float16
"""

import os
from typing import Any, Callable, ClassVar, Dict, List, Optional

import requests
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from transformers import pipeline

_PII_INFERENCE_URL = os.environ.get("PII_INFERENCE_URL", "").rstrip("/")


@register_validator(name="custom-pii-ner", data_type="string")
class CustomPIIValidator(Validator):
    """
    Wraps our finetuned NER model as a guardrails-ai Validator.

    In REMOTE mode: calls PII_INFERENCE_URL/token-classification (no local model).
    In LOCAL mode: loads model in-process (dev/fallback only).
    """

    _PIPELINE_CACHE: ClassVar[Dict[str, Any]] = {}

    def __init__(
        self,
        model_path: str = None,
        threshold: float = 0.5,
        on_fail: Optional[Callable] = None,
    ):
        if model_path is None:
            model_path = os.environ.get(
                "PII_MODEL_PATH",
                "vineeth453/qwen25-7b-pii-detection-lora",
            )
        super().__init__(on_fail=on_fail, model_path=model_path, threshold=threshold)
        self.model_path = model_path
        self.threshold = threshold
        self._remote_url = _PII_INFERENCE_URL

    def _load_model(self):
        if self.model_path not in CustomPIIValidator._PIPELINE_CACHE:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            print(f"[PII Validator] Loading model locally: {self.model_path}")
            CustomPIIValidator._PIPELINE_CACHE[self.model_path] = pipeline(
                task="ner",
                model=self.model_path,
                tokenizer=self.model_path,
                aggregation_strategy="simple",
                device="cpu",
                token=hf_token,
            )
            print("[PII Validator] Model ready")

    def _call_remote(self, text: str) -> List[Dict]:
        """Call TEI /token-classification endpoint and return raw entity list."""
        resp = requests.post(
            f"{self._remote_url}/token-classification",
            json={"inputs": text, "aggregation_strategy": "simple"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _parse_entities(self, raw_entities: List[Dict]) -> tuple:
        pii_entities = []
        for entity in raw_entities:
            label = entity.get("entity_group", entity.get("entity", "UNKNOWN"))
            score = float(entity.get("score", 0.0))
            word = entity.get("word", "")
            if label.upper() == "O":
                continue
            if score >= self.threshold:
                pii_entities.append({
                    "entity_type": label,
                    "text": word,
                    "confidence": round(score, 4),
                    "start": entity.get("start"),
                    "end": entity.get("end"),
                })
        pii_score = max((e["confidence"] for e in pii_entities), default=0.0)
        return pii_entities, pii_score

    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        if self._remote_url:
            raw = self._call_remote(value)
        else:
            self._load_model()
            raw = CustomPIIValidator._PIPELINE_CACHE[self.model_path](value)

        pii_entities, pii_score = self._parse_entities(raw)

        return PassResult(
            metadata={
                "pii_score":    round(pii_score, 4),
                "pii_entities": pii_entities,
                "validator":    "custom-pii-ner",
            }
        )
