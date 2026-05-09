"""
Custom PII Validator — three modes via env vars:

  SAGEMAKER GENERATIVE mode (production):
    Set PII_SAGEMAKER_ENDPOINT=spartanguard-pii
    → DJL LMI vLLM endpoint (Qwen2.5-7B-Instruct + vineeth453 PII LoRA)
    → Uses invoke-endpoint with OpenAI-compat payload via boto3
    → Extracts detected PII entities from generative output

  REMOTE mode (fallback production):
    Set PII_INFERENCE_URL=http://<host>:<port>
    → calls /token-classification endpoint

  LOCAL mode (dev):
    Leave both unset → loads model in-process (slow, dev only)
"""

import json
import os
import re
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
_PII_SAGEMAKER_ENDPOINT = os.environ.get("PII_SAGEMAKER_ENDPOINT", "").strip()
_PII_LORA_ADAPTER = os.environ.get("PII_LORA_ADAPTER", "vineeth453/qwen25-7b-pii-detection-lora")
_AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("DEPLOYMENT_REGION", "us-west-2"))

_PII_SYSTEM_PROMPT = (
    "You are a PII detection system. Analyze the user message and identify any Personally "
    "Identifiable Information (PII) such as: names, email addresses, phone numbers, SSNs, "
    "credit card numbers, addresses, dates of birth, IP addresses, or other sensitive personal data. "
    "Respond in JSON format: {\"pii_found\": true/false, \"entities\": [{\"type\": \"<TYPE>\", \"text\": \"<VALUE>\", \"confidence\": 0.95}]}. "
    "If no PII is found, respond: {\"pii_found\": false, \"entities\": []}. "
    "Respond with ONLY valid JSON, no other text."
)


def _build_qwen_prompt(system_prompt: str, user_text: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _invoke_sagemaker_generative(endpoint_name: str, text: str, system_prompt: str, region: str, lora_adapter: str = None) -> str:
    import boto3
    client = boto3.client("sagemaker-runtime", region_name=region)
    payload = {
        "inputs": _build_qwen_prompt(system_prompt, text),
        "parameters": {
            "max_new_tokens": 256,
            "do_sample": False,
            "temperature": 1.0,
        },
    }
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read())
    if "generated_text" in result:
        return result["generated_text"]
    if isinstance(result, list) and result:
        return result[0].get("generated_text", str(result[0]))
    return str(result)


def _invoke_sagemaker_classifier(endpoint_name: str, payload: dict, region: str) -> list:
    import boto3
    client = boto3.client("sagemaker-runtime", region_name=region)
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read())


@register_validator(name="custom-pii-ner", data_type="string")
class CustomPIIValidator(Validator):
    """
    Wraps PII detection as a guardrails-ai Validator.

    Priority: SageMaker (Qwen2.5-7B generative) > REMOTE URL > LOCAL model
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
                "dslim/bert-base-NER",
            )
        super().__init__(on_fail=on_fail, model_path=model_path, threshold=threshold)
        self.model_path = model_path
        self.threshold = threshold
        self._remote_url = _PII_INFERENCE_URL
        self._sm_endpoint = _PII_SAGEMAKER_ENDPOINT
        self._lora_adapter = _PII_LORA_ADAPTER

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

    def _call_sagemaker_generative(self, text: str) -> List[Dict]:
        raw_output = _invoke_sagemaker_generative(
            self._sm_endpoint, text, _PII_SYSTEM_PROMPT, _AWS_REGION, self._lora_adapter
        )
        return self._parse_generative_output(raw_output)

    def _parse_generative_output(self, output: str) -> List[Dict]:
        try:
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get("entities", [])
                result = []
                for e in entities:
                    result.append({
                        "entity_group": e.get("type", "PII"),
                        "word": e.get("text", ""),
                        "score": float(e.get("confidence", 0.9)),
                        "start": e.get("start", 0),
                        "end": e.get("end", 0),
                    })
                return result
        except Exception as exc:
            print(f"[PII Validator] Failed to parse generative output: {exc} — raw: {output[:200]}")
        return []

    def _call_remote(self, text: str) -> List[Dict]:
        resp = requests.post(
            f"{self._remote_url}/token-classification",
            json={"inputs": text, "aggregation_strategy": "simple"},
            timeout=30,
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
        if self._sm_endpoint:
            raw = self._call_sagemaker_generative(value)
        elif self._remote_url:
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
