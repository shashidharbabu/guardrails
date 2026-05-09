"""
SageMaker OpenAI-compat proxy for MAD pipeline.

Runs on port 8080 inside the mad-api pod. Translates standard OpenAI
/v1/chat/completions requests into boto3 SageMaker invoke-endpoint calls,
then returns OpenAI-format responses.

This lets vllm_client.py (which uses LangChain ChatOpenAI) work unchanged
against the spartanguard-agents SageMaker endpoint without any public URL.

The `model` field in the request maps to the LoRA adapter name:
  "agent-a" → loads agent-a LoRA adapter
  "agent-b" → loads agent-b LoRA adapter
  anything else → no lora_name sent (base model)
"""

import json
import logging
import os
import re
import time
import uuid

import boto3
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SageMaker OpenAI Proxy", version="1.0.0")

_ENDPOINT = os.environ.get("AGENTS_SAGEMAKER_ENDPOINT", "spartanguard-agents")
_REGION = os.environ.get("AWS_REGION", "us-west-2")
_KNOWN_LORAS = {"agent-a", "agent-b"}

from botocore.config import Config as _BotocoreConfig
_sm_client = boto3.client(
    "sagemaker-runtime",
    region_name=_REGION,
    config=_BotocoreConfig(read_timeout=120, connect_timeout=10, retries={"max_attempts": 1}),
)

_STOP_TOKENS = re.compile(
    r"(<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|<\|end_of_text\|>).*",
    re.DOTALL,
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "agent-a"
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = 512
    stream: bool = False


def _build_qwen_prompt(messages: list[Message]) -> str:
    parts = []
    for msg in messages:
        role = msg.role
        content = msg.content
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant")
    return "\n".join(parts) + "\n"


def _clean_output(text: str) -> str:
    text = _STOP_TOKENS.sub("", text)
    return text.strip()


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "agent-a", "object": "model", "owned_by": "spartanguard"},
            {"id": "agent-b", "object": "model", "owned_by": "spartanguard"},
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported")

    prompt = _build_qwen_prompt(req.messages)
    # No cap — let each call complete fully. CLAIM_CONCURRENCY=1 ensures serial
    # execution so each call finishes within SageMaker's 60s hard timeout.
    payload: dict = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": req.max_tokens,
            "do_sample": req.temperature > 0,
            "temperature": req.temperature if req.temperature > 0 else 1.0,
        },
    }
    if req.model in _KNOWN_LORAS:
        payload["lora_name"] = req.model

    t0 = time.perf_counter()
    last_exc = None
    for attempt in range(3):  # retry up to 3 times on timeout
        try:
            response = _sm_client.invoke_endpoint(
                EndpointName=_ENDPOINT,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            result = json.loads(response["Body"].read())
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("SageMaker invoke attempt %d failed: %s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(2)
    if last_exc is not None:
        logger.error("SageMaker invoke failed after 3 attempts: %s", last_exc)
        raise HTTPException(status_code=502, detail=f"SageMaker error: {last_exc}")

    latency_ms = int((time.perf_counter() - t0) * 1000)

    if isinstance(result, dict) and "generated_text" in result:
        raw = result["generated_text"]
    elif isinstance(result, list) and result:
        raw = result[0].get("generated_text", str(result[0]))
    else:
        raw = str(result)

    content = _clean_output(raw)
    logger.info("model=%s lora=%s latency_ms=%d len=%d",
                req.model, payload.get("lora_name", "none"), latency_ms, len(content))

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(content.split()),
            "total_tokens": len(prompt.split()) + len(content.split()),
        },
    })


@app.get("/health")
def health():
    return {"status": "ok", "endpoint": _ENDPOINT, "region": _REGION}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
