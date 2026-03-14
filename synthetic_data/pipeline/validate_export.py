from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from .export import export_dataset
from .io import load_checkpoint, save_checkpoint
from .validation import validate_agent_pairs, validate_embedding_triplets


def validate_and_export(
    *,
    synthetic_dir: str,
    exports_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    label_distribution: Dict[str, float],
) -> Dict[str, Any]:
    triplets_raw = load_checkpoint(synthetic_dir, "embedding/triplets_raw.jsonl")
    pairs_raw = load_checkpoint(synthetic_dir, "agent/pairs_raw.jsonl")

    triplets_ok, triplets_rej = validate_embedding_triplets(triplets_raw)
    save_checkpoint(triplets_ok, synthetic_dir, "embedding/triplets_validated.jsonl")
    save_checkpoint(triplets_rej, synthetic_dir, "rejected/embedding_triplets_rejected.jsonl")

    pairs_ok, pairs_rej, agent_stats = validate_agent_pairs(
        pairs_raw,
        target_distribution=label_distribution,
    )
    save_checkpoint(pairs_ok, synthetic_dir, "agent/pairs_validated.jsonl")
    save_checkpoint(pairs_rej, synthetic_dir, "rejected/agent_pairs_rejected.jsonl")

    emb_paths = export_dataset(
        triplets_ok,
        exports_dir=exports_dir,
        name="embedding",
        key_fn=lambda x: (
            (x.get("metadata", {}) or {}).get("threat_categories", []) or ["unknown"]
        )[0],
        train=train_ratio,
        val=val_ratio,
        test=test_ratio,
    )

    agent_paths = export_dataset(
        pairs_ok,
        exports_dir=exports_dir,
        name="agent",
        key_fn=lambda x: (x.get("metadata", {}) or {}).get("decision") or "unknown",
        train=train_ratio,
        val=val_ratio,
        test=test_ratio,
    )

    decision_counts = Counter((p.get("metadata", {}) or {}).get("decision") for p in pairs_ok)
    decision_pct = (
        {k: (v / len(pairs_ok)) for k, v in decision_counts.items()} if pairs_ok else {}
    )

    return {
        "embedding": {
            "raw": len(triplets_raw),
            "validated": len(triplets_ok),
            "rejected": len(triplets_rej),
            "export_paths": emb_paths,
        },
        "agent": {
            "raw": len(pairs_raw),
            "validated": len(pairs_ok),
            "rejected": len(pairs_rej),
            "export_paths": agent_paths,
            "decision_counts": dict(decision_counts),
            "decision_distribution": decision_pct,
            "stats": agent_stats,
        },
    }

