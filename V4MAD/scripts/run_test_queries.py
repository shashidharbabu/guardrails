#!/usr/bin/env python3
"""
Run 1-5 local V4MAD test queries through the real data path.

Expected test topology:
  - baseline/decomposer: local OpenAI-compatible endpoint, usually Ollama
  - claim RAG: Qdrant Cloud + local BM25
  - agents: remote Colab vLLM LoRA server, model=agent_a/agent_b
  - judge: Claude Sonnet via Anthropic API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env", override=False)

from V4MAD.api import run_v4mad


DEFAULT_QUERIES = Path("/Users/vineethrayadurgam/Downloads/finalMAD_5090_handoff/data/queries_5_test.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4MAD test queries")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-claims", type=int, default=2)
    parser.add_argument("--baseline-model", default=os.environ.get("DEFAULT_LLM_MODEL", "qwen2.5:7b"))
    parser.add_argument("--baseline-url", default=os.environ.get("LLM_PROVIDER_URL", "http://localhost:11434"))
    return parser.parse_args()


async def baseline_answer(query: str, model: str, base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    client = AsyncOpenAI(base_url=base, api_key="EMPTY")
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enterprise compliance assistant. Answer accurately "
                    "and include specific requirements when relevant."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


async def main() -> None:
    args = parse_args()
    queries_path = Path(args.queries)
    queries = [
        json.loads(line)
        for line in queries_path.read_text().splitlines()
        if line.strip()
    ][: args.limit]

    print(f"queries={queries_path}")
    print(f"limit={len(queries)} max_claims={args.max_claims}")
    print(f"agents_base_url={os.environ.get('AGENTS_BASE_URL', '')}")
    print(f"judge_backend={os.environ.get('JUDGE_BACKEND', 'vllm')}")
    print(f"decomposer_base_url={os.environ.get('DECOMPOSER_BASE_URL', '')}")

    for item in queries:
        qid = item["query_id"]
        query = item["user_query"]
        print("\n" + "=" * 100)
        print(f"{qid}: {query}")

        answer = await baseline_answer(query, args.baseline_model, args.baseline_url)
        print("baseline_answer:", answer[:500].replace("\n", " "))

        result = await run_v4mad(
            query,
            answer,
            query_id=f"local_{qid}",
            rollout_id=f"local_{qid}_rollout",
            max_claims=args.max_claims,
        )
        payload = result.model_dump(mode="json")
        print(
            "mad_result:",
            json.dumps(
                {
                    "routing_decision": payload["routing_decision"],
                    "aggregate_confidence": payload["aggregate_confidence"],
                    "claims": len(payload["claims"]),
                    "judge_verdicts": len(payload["judge_verdicts"]),
                },
                indent=2,
            ),
        )
        for claim in payload["claims"]:
            print(f"- claim={claim['claim_text']}")
            print(f"  verdict={claim.get('verdict')} evidence={claim.get('evidence_chunks')}")


if __name__ == "__main__":
    asyncio.run(main())
