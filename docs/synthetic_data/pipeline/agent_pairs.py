from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .anthropic_utils import call_claude_json
from .io import load_checkpoint, save_checkpoint
from .prompts import AGENT_GENERATION_PROMPT


def generate_agent_pair(
    chunks_subset: List[dict],
    *,
    decision: str,
    threat_category: str,
    client: Any,
    model: str,
    max_tokens_response: int,
    retry_limit: int,
    debug: bool = False,
) -> Optional[dict]:
    if not chunks_subset:
        if debug:
            print("[DEBUG][pairs] empty_chunks_subset")
        return None

    def dbg(msg: str) -> None:
        if debug:
            print(f"[DEBUG][pairs] {msg}")

    context = random.sample(chunks_subset, min(3, len(chunks_subset)))
    context_text = "\n\n".join(
        [
            f'[{c.get("doc_id")} | {c.get("summary", "")}]\n{str(c.get("text", "") or "")[:400]}'
            for c in context
        ]
    )

    prompt = AGENT_GENERATION_PROMPT.format(
        context_chunks=context_text,
        decision=decision,
        threat_category=threat_category,
    )

    try:
        result = call_claude_json(
            client=client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens_response,
            retry_limit=retry_limit,
        )
        if not isinstance(result, dict):
            dbg("model_output_not_object")
            return None
        if "user_query" not in result or "response" not in result:
            dbg("model_output_missing_user_query_or_response")
            return None

        out = {
            "id": f"agent_{decision.lower()}_{threat_category}_{int(time.time())}_{random.randint(1000, 9999)}",
            "instruction": {
                "system": "You are an enterprise AI safety agent. Evaluate the user query using only the retrieved policy context. Return a structured JSON decision.",
                "user_query": result["user_query"],
                "retrieved_context": [
                    {
                        "rank": i + 1,
                        "source": c.get("doc_id"),
                        "chunk_id": c.get("chunk_id"),
                        "text": str(c.get("text", "") or "")[:400],
                    }
                    for i, c in enumerate(context)
                ],
            },
            "response": result["response"],
            "metadata": {
                "decision": decision,
                "threat_category": threat_category,
                "source_docs": [c.get("doc_id") for c in context],
            },
        }
        dbg(f"pair_generated id={out['id']} decision={decision} threat_category={threat_category}")
        return out
    except Exception:
        dbg("pair_generation_exception")
        return None


def _target_counts(
    *,
    agent_target: int,
    decisions: List[str],
    threat_categories: List[str],
    label_distribution: Dict[str, float],
) -> Dict[Tuple[str, str], int]:
    """
    Returns desired counts per (decision, threat_category).

    The spec asks for overall label distribution (50/30/20) AND coverage over threat categories.
    We allocate per decision by distribution, then split evenly across threat categories.
    """
    per_key: Dict[Tuple[str, str], int] = {}
    for decision in decisions:
        decision_total = int(round(agent_target * float(label_distribution.get(decision, 0.0))))
        base = decision_total // max(len(threat_categories), 1)
        rem = decision_total % max(len(threat_categories), 1)
        for idx, tc in enumerate(threat_categories):
            per_key[(decision, tc)] = base + (1 if idx < rem else 0)
    return per_key


def generate_agent_pairs(
    *,
    synthetic_dir: str,
    client: Any,
    model: str,
    max_tokens_response: int,
    retry_limit: int,
    agent_target: int,
    decisions: List[str],
    threat_categories: List[str],
    label_distribution: Dict[str, float],
    sleep_seconds: float = 0.5,
    seed: int = 17,
) -> List[dict]:
    random.seed(seed)

    enriched = load_checkpoint(synthetic_dir, "enriched_chunks.jsonl")
    agent_pairs = load_checkpoint(synthetic_dir, "agent/pairs_progress.jsonl")

    existing_counts = Counter(
        (p.get("metadata", {}).get("decision"), p.get("metadata", {}).get("threat_category"))
        for p in agent_pairs
    )
    targets = _target_counts(
        agent_target=agent_target,
        decisions=decisions,
        threat_categories=threat_categories,
        label_distribution=label_distribution,
    )

    for decision in decisions:
        for threat_cat in threat_categories:
            relevant = [c for c in enriched if threat_cat in (c.get("threat_categories", []) or [])]
            if not relevant:
                continue

            needed = max(targets.get((decision, threat_cat), 0) - existing_counts[(decision, threat_cat)], 0)
            if needed <= 0:
                continue

            for _ in tqdm(range(needed), desc=f"{decision}/{threat_cat}"):
                pair = generate_agent_pair(
                    relevant,
                    decision=decision,
                    threat_category=threat_cat,
                    client=client,
                    model=model,
                    max_tokens_response=max_tokens_response,
                    retry_limit=retry_limit,
                )
                if pair:
                    agent_pairs.append(pair)
                    existing_counts[(decision, threat_cat)] += 1
                time.sleep(sleep_seconds)

            save_checkpoint(agent_pairs, synthetic_dir, "agent/pairs_progress.jsonl")

    save_checkpoint(agent_pairs, synthetic_dir, "agent/pairs_raw.jsonl")
    return agent_pairs

