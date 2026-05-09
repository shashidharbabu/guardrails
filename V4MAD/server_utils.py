"""
vLLM server lifecycle management.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

import requests


def launch_vllm_server(
    model: str,
    port: int,
    gpu_memory_utilization: float = 0.88,
    max_model_len: int = 4096,
    enable_prefix_caching: bool = True,
    dtype: str = "bfloat16",
    lora_modules: Mapping[str, str] | None = None,
    max_lora_rank: int | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--dtype", dtype,
        "--trust-remote-code",
    ]

    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    if lora_modules:
        cmd.append("--enable-lora")
        cmd.append("--lora-modules")
        cmd.extend(f"{name}={path}" for name, path in lora_modules.items())
        if max_lora_rank is not None:
            cmd.extend(["--max-lora-rank", str(max_lora_rank)])

    log_path = Path(f"/content/vllm_{port}.log")
    log_file = open(log_path, "w")

    print(f"[vLLM] Launching {model} on port {port}...")
    print(f"[vLLM] Logs: {log_path}")

    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def wait_for_server(port: int, timeout: int = 600) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()

    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print(f"\n[vLLM] Server ready on port {port} ({int(time.time()-start)}s)")
                return True
        except Exception:
            pass

        time.sleep(4)
        print(".", end="", flush=True)

    print(f"\n[vLLM] Server on port {port} did not start within {timeout}s")
    print(f"[vLLM] Check logs: /content/vllm_{port}.log")
    return False


def shutdown_server(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()

    print("[vLLM] Server stopped.")
