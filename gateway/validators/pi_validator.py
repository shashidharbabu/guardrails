"""
Custom Prompt Injection Validator — supports two modes via PI_INFERENCE_URL env var:

  LOCAL mode (default, dev):
    Loads harshitasayala/pi-llama31-8b or meta-llama/Llama-Prompt-Guard-2-86M
    in-process. Heavy (~16GB RAM for LLaMA-3.1-8B).

  REMOTE mode (production):
    Set PI_INFERENCE_URL=http://<host>:<port> to call a HuggingFace TGI or TEI
    /text-classification endpoint. No model loaded locally — container stays lean.

HuggingFace TGI server command (on GPU EC2):
  docker run -p 3001:80 ghcr.io/huggingface/text-generation-inference:latest \
    --model-id harshitasayala/pi-llama31-8b \
    --dtype float16

Labels (BENIGN / MALICIOUS) — both model families use these.
"""

import os
from typing import Any, Callable, ClassVar, Dict, Optional

import requests
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from transformers import pipeline

_PI_INFERENCE_URL = os.environ.get("PI_INFERENCE_URL", "").rstrip("/")


@register_validator(name="custom-pi-classifier", data_type="string")
class CustomPIValidator(Validator):
    """
    Wraps prompt injection classifier as a guardrails-ai Validator.

    REMOTE mode: calls PI_INFERENCE_URL/classify (no local model, fast, GPU-backed).
    LOCAL mode: loads model in-process (dev/fallback only).
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
                "harshitasayala/pi-llama31-8b",
            )
        super().__init__(on_fail=on_fail, model_path=model_path, pi_threshold=pi_threshold)
        self.model_path = model_path
        self.pi_threshold = pi_threshold
        self._remote_url = _PI_INFERENCE_URL

    def _load_model(self):
        if self.model_path not in CustomPIValidator._PIPELINE_CACHE:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            print(f"[PI Validator] Loading model locally: {self.model_path}")
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

    def _call_remote(self, text: str) -> list:
        """Call remote classifier and return raw label scores."""
        resp = requests.post(
            f"{self._remote_url}/classify",
            json={"inputs": text},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_pi_score(self, pipeline_output: list) -> float:
        raw = pipeline_output[0] if isinstance(pipeline_output[0], list) else pipeline_output
        for item in raw:
            internal = self.LABEL_MAP.get(item["label"].upper())
            if internal == "prompt_injection":
                return float(item["score"])
        return 0.0

    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        if self._remote_url:
            output = self._call_remote(value)
        else:
            self._load_model()
            output = CustomPIValidator._PIPELINE_CACHE[self.model_path](value)

        pi_score = self._get_pi_score(output)
        safe_score = max(0.0, 1.0 - pi_score)

        return PassResult(
            metadata={
                "pi_score":   round(pi_score, 4),
                "safe_score": round(safe_score, 4),
                "validator":  "custom-pi-classifier",
            }
        )
