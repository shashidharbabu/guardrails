"""
Custom PII Validator - wraps our finetuned NER model (shashidharbabu/deberta-pii-guardrails).

Model is RobertaForTokenClassification with 112 IOB2 PII entity labels.
Every non-O entity the model predicts IS a PII entity — no label whitelist needed.

Model path: gateway/models/pii_ner_model/
  OR set env var: PII_MODEL_PATH=path/to/model  (or HuggingFace Hub ID)

HuggingFace authentication: set HF_TOKEN env var (or in .env) for gated model access.
"""

import os
from typing import Any, Callable, ClassVar, Dict, List, Optional

from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from transformers import pipeline


@register_validator(name="custom-pii-ner", data_type="string")
class CustomPIIValidator(Validator):
    """
    Wraps our finetuned NER model as a guardrails-ai Validator.

    Accepts ALL non-O entity labels as PII — the model was trained specifically on
    PII entity types so every prediction is meaningful.

    pii_score = max confidence across all detected PII entities (0.0 if none).
    """

    # Class-level cache shared across all instances (guardrails copies validator
    # instances internally, so instance-level caching causes repeated model loads).
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
                os.path.join(os.path.dirname(__file__), "..", "models", "pii_ner_model"),
            )
        super().__init__(on_fail=on_fail, model_path=model_path, threshold=threshold)
        self.model_path = model_path
        self.threshold = threshold

    def _load_model(self):
        """Load model once per unique path; subsequent calls hit the class-level cache."""
        if self.model_path not in CustomPIIValidator._PIPELINE_CACHE:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            print(f"[PII Validator] Loading model from: {self.model_path}")
            CustomPIIValidator._PIPELINE_CACHE[self.model_path] = pipeline(
                task="ner",
                model=self.model_path,
                tokenizer=self.model_path,
                aggregation_strategy="simple",
                device="cpu",
                token=hf_token,
            )
            print("[PII Validator] Model ready")

    @property
    def _pipe(self):
        return CustomPIIValidator._PIPELINE_CACHE.get(self.model_path)

    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        self._load_model()

        ner_results = self._pipe(value)
        pii_entities: List[Dict] = []

        for entity in ner_results:
            label = entity.get("entity_group", entity.get("entity", "UNKNOWN"))
            score = float(entity.get("score", 0.0))
            word = entity.get("word", "")

            # Skip the "O" (outside) label — every other prediction is a PII entity
            if label.upper() == "O":
                continue

            if score >= self.threshold:
                pii_entities.append(
                    {
                        "entity_type": label,
                        "text": word,
                        "confidence": round(score, 4),
                        "start": entity.get("start"),
                        "end": entity.get("end"),
                    }
                )

        pii_score = max((e["confidence"] for e in pii_entities), default=0.0)

        # Always return PassResult so guardrails doesn't short-circuit the chain.
        # DecisionEngine reads scores from metadata and makes the final routing decision.
        return PassResult(
            metadata={
                "pii_score":    round(pii_score, 4),
                "pii_entities": pii_entities,
                "validator":    "custom-pii-ner",
            }
        )
