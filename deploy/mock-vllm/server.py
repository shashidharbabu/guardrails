from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _message_text(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(item.get("text", item)) for item in content)
    return "\n".join(parts)


def _completion_for(model: str, messages: list[dict]) -> str:
    prompt = _message_text(messages).lower()

    if "enterprise compliance assistant" in prompt:
        return (
            "HIPAA requires covered entities to protect electronic protected health "
            "information with appropriate administrative, physical, and technical safeguards."
        )

    if model in {"agent_a", "agent_b"} or "evidence_cited" in prompt:
        return json.dumps(
            {
                "verdict": "SUPPORTED",
                "reasoning": "The claim is treated as supported in the local mock vLLM smoke test.",
                "evidence_cited": [],
                "confidence_internal": 0.9,
            }
        )

    if "claims" in prompt and ("decompose" in prompt or "claim" in prompt):
        return json.dumps(
            {
                "claims": [
                    {
                        "claim_text": (
                            "HIPAA requires covered entities to protect electronic protected "
                            "health information with administrative physical and technical safeguards."
                        ),
                        "claim_index": 0,
                        "is_material": True,
                        "is_critical": True,
                        "confidence_prior": 0.9,
                    }
                ]
            }
        )

    if "judge" in prompt or "v_label" in prompt:
        return json.dumps(
            {
                "v_label": 1.0,
                "judge_confidence": 0.9,
                "judge_reasoning": "The claim is supported for local deployment smoke testing.",
                "evidence_chunk_ids": [],
            }
        )

    return "Local mock vLLM response for deployment smoke testing."


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health" or self.path.startswith("/v1/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "Qwen/Qwen2.5-14B-Instruct", "object": "model"},
                        {"id": "agent_a", "object": "model"},
                        {"id": "agent_b", "object": "model"},
                    ],
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self.path.startswith("/v1/chat/completions"):
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        model = payload.get("model", "mock-model")
        content = _completion_for(model, payload.get("messages", []))
        now = int(time.time())
        self._send_json(
            200,
            {
                "id": f"chatcmpl-mock-{now}",
                "object": "chat.completion",
                "created": now,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
