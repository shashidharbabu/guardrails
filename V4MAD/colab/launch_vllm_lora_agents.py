#!/usr/bin/env python3
"""
Colab A100 helper: launch one vLLM server with both GRPO LoRA agents.

Usage in Colab:
  1. Runtime -> Change runtime type -> A100 GPU.
  2. Upload finalA.zip and finalB.zip, or mount Drive and set FINAL_A_DIR / FINAL_B_DIR.
  3. Optional for ngrok:
       export NGROK_AUTHTOKEN=...
  4. Run:
       python launch_vllm_lora_agents.py

The script prints AGENTS_BASE_URL. Put that value in local .env:
  USE_GRPO_LORA_AGENTS=1
  AGENT_A_MODEL=agent_a
  AGENT_B_MODEL=agent_b
  AGENTS_BASE_URL=https://<ngrok-id>.ngrok-free.app/v1
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests


MODEL = os.environ.get("AGENTS_MODEL", "Qwen/Qwen2.5-14B-Instruct")
PORT = int(os.environ.get("AGENTS_PORT", "8003"))
FINAL_A_DIR = Path(os.environ.get("FINAL_A_DIR", "/content/adapters/finalA"))
FINAL_B_DIR = Path(os.environ.get("FINAL_B_DIR", "/content/adapters/finalB"))
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def maybe_unzip(name: str, target: Path) -> None:
    if target.exists():
        return
    zip_path = Path(f"/content/{name}.zip")
    if not zip_path.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target.parent)
    extracted = target.parent / name
    if extracted != target and extracted.exists():
        shutil.move(str(extracted), str(target))


def check_adapter(path: Path) -> None:
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {path}")


def wait_for_server() -> None:
    url = f"http://127.0.0.1:{PORT}/v1/models"
    for i in range(180):
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print("vLLM ready:", resp.json(), flush=True)
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    raise TimeoutError("vLLM did not become ready within 15 minutes")


def start_ngrok() -> str:
    from pyngrok import ngrok

    if NGROK_AUTHTOKEN:
        ngrok.set_auth_token(NGROK_AUTHTOKEN)
    tunnel = ngrok.connect(PORT, "http")
    public_url = tunnel.public_url
    if not public_url.startswith("https://"):
        public_url = public_url.replace("http://", "https://", 1)
    return public_url


def main() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "-U", "vllm", "pyngrok"])

    maybe_unzip("finalA", FINAL_A_DIR)
    maybe_unzip("finalB", FINAL_B_DIR)
    check_adapter(FINAL_A_DIR)
    check_adapter(FINAL_B_DIR)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--trust-remote-code",
        "--enable-lora",
        "--lora-modules",
        f"agent_a={FINAL_A_DIR}",
        f"agent_b={FINAL_B_DIR}",
        "--max-lora-rank", os.environ.get("MAX_LORA_RANK", "32"),
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.88"),
        "--dtype", os.environ.get("VLLM_DTYPE", "bfloat16"),
        "--enable-prefix-caching",
    ]

    log_path = Path("/content/vllm_agents.log")
    print("Starting vLLM. Logs:", log_path, flush=True)
    log = log_path.open("w")
    subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    wait_for_server()

    public_url = start_ngrok()
    print("\nPaste these into local .env:", flush=True)
    print("USE_GRPO_LORA_AGENTS=1", flush=True)
    print("AGENT_A_MODEL=agent_a", flush=True)
    print("AGENT_B_MODEL=agent_b", flush=True)
    print(f"AGENTS_BASE_URL={public_url}/v1", flush=True)
    print("AGENTS_API_KEY=EMPTY", flush=True)

    print("\nQuick test:", flush=True)
    resp = requests.post(
        f"{public_url}/v1/chat/completions",
        headers={"Authorization": "Bearer EMPTY"},
        json={
            "model": "agent_a",
            "messages": [{"role": "user", "content": "Return only JSON: {\"ok\": true}"}],
            "max_tokens": 64,
            "temperature": 0,
        },
        timeout=60,
    )
    print(resp.status_code, resp.text[:1000], flush=True)


if __name__ == "__main__":
    main()
