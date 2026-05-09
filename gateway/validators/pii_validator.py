"""
Custom PII Validator - wraps vineeth453/qwen25-7b-pii-detection-lora (QLoRA adapter).

Model: Qwen2.5-7B-Instruct fine-tuned on ai4privacy/pii-masking-200k (56 PII entity types,
4 languages: EN/FR/DE/IT). Returns structured JSON — completely different from the previous
NER token-classifier; this is a causal LM that generates {"entities":[{"text":"...","label":"..."}]}.

Requirements: GPU with ~5 GB VRAM for 4-bit NF4 quantization, or Apple MPS.
Model path env: PII_MODEL_PATH (adapter repo ID or local path, defaults to HF hub ID).
Base model env: PII_BASE_MODEL (defaults to Qwen/Qwen2.5-7B-Instruct).
HF auth: HF_TOKEN or HUGGING_FACE_HUB_TOKEN.
"""

import json
import logging
import os
import re
from typing import Any, Callable, ClassVar, Dict, List, Optional

from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a PII detection system. Extract all personally identifiable information.\n"
    'Return ONLY valid JSON: {"entities":[{"text":"...","label":"..."}]}'
)


def _build_prompt(text: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{_SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@register_validator(name="custom-pii-ner", data_type="string")
class CustomPIIValidator(Validator):
    """
    Wraps vineeth453/qwen25-7b-pii-detection-lora as a guardrails-ai Validator.

    Loads Qwen2.5-7B-Instruct in 4-bit NF4 quantization and applies the PII LoRA adapter.
    The model returns JSON {"entities":[{"text":"..","label":".."}]} which is parsed into
    the same metadata shape the rest of the gateway expects (pii_score, pii_entities).

    pii_score is set to 1.0 when any entity is detected (the model is a binary detector
    per entity, not a calibrated classifier — there is no per-entity confidence score).
    """

    _MODEL_CACHE: ClassVar[Dict[str, Any]] = {}

    _DEFAULT_ADAPTER = "vineeth453/qwen25-7b-pii-detection-lora"
    _DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(
        self,
        model_path: str = None,
        threshold: float = 0.5,
        on_fail: Optional[Callable] = None,
    ):
        adapter_path = model_path or os.environ.get("PII_MODEL_PATH", self._DEFAULT_ADAPTER)
        super().__init__(on_fail=on_fail, model_path=adapter_path, threshold=threshold)
        self.model_path = adapter_path
        self.threshold = threshold

    def _load_model(self):
        if self.model_path in CustomPIIValidator._MODEL_CACHE:
            return

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        base_model_id = os.environ.get("PII_BASE_MODEL", self._DEFAULT_BASE)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        logger.info("[PII Validator] Loading base model %s with 4-bit NF4 quant", base_model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
        )

        logger.info("[PII Validator] Applying LoRA adapter from %s", self.model_path)
        model = PeftModel.from_pretrained(base, self.model_path, token=hf_token)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, token=hf_token)

        CustomPIIValidator._MODEL_CACHE[self.model_path] = (model, tokenizer)
        logger.info("[PII Validator] Model ready")

    @property
    def _model_and_tokenizer(self):
        return CustomPIIValidator._MODEL_CACHE.get(self.model_path)

    def _run_inference(self, text: str) -> List[Dict]:
        """Generate PII entities for the given text and return a list of entity dicts."""
        import torch

        model, tokenizer = self._model_and_tokenizer
        prompt = _build_prompt(text)

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip().replace("<|im_end|>", "").strip()

        try:
            data = json.loads(response)
            return data.get("entities", [])
        except (json.JSONDecodeError, AttributeError):
            # Try to extract JSON substring if the model wrapped it in extra text
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    return data.get("entities", [])
                except (json.JSONDecodeError, AttributeError):
                    pass
            logger.warning("[PII Validator] Failed to parse model output: %r", response[:200])
            return []

    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        self._load_model()

        raw_entities = self._run_inference(value)
        pii_entities: List[Dict] = []

        for ent in raw_entities:
            label = ent.get("label", "UNKNOWN")
            word = ent.get("text", "")
            if not word:
                continue
            pii_entities.append(
                {
                    "entity_type": label,
                    "text": word,
                    # Model does not emit per-entity confidence; use 1.0 for detected entities
                    "confidence": 1.0,
                    "start": None,
                    "end": None,
                }
            )

        # pii_score: 1.0 if any entity detected, 0.0 otherwise — threshold comparison still works
        pii_score = 1.0 if pii_entities else 0.0

        return PassResult(
            metadata={
                "pii_score":    round(pii_score, 4),
                "pii_entities": pii_entities,
                "validator":    "custom-pii-ner",
            }
        )
