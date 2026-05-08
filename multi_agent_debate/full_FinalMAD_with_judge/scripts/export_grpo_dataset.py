"""
Export debate outputs from SQLite to GRPO-style JSONL.

Each row = one (agent, round, claim) training example.

Output format matches grpo_finetune_package/data/agent_a_grpo.jsonl:
{
  "prompt": "<SYSTEM>\nUSER QUERY: ...\nCLAIM TO VERIFY: ...",
  "completion": "{\"verdict\": ...}",
  "v_label": 0.5,
  "role": "agent_a",
  "round": 0,
  "claim_id": "...",
  "query_id": "q_001",
  "is_critical": false,
  "is_material": true,
  "judge_confidence": 0.85,
  "valid_completion": true
}

Usage:
  python scripts/export_grpo_dataset.py --db mad.db --run-id my-run-01 --out-dir ./grpo_export/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from configs import config
from src.db.db import fetch_all
from src.utils.prompts import (
    AGENT_A_SYSTEM, AGENT_B_SYSTEM, format_chunks, round0_user, round1_user
)
from src.schemas.schemas import AgentRole, AgentVerdict, Claim, EvidenceCitation, AgentOutputFull, strip_for_peer


logger = logging.getLogger(__name__)


def _claim_from_row(row: dict) -> Claim:
    return Claim(
        claim_id=row["claim_id"],
        claim_text=row["claim_text"],
        claim_index=row["claim_index"],
        is_material=bool(row["is_material"]),
        is_critical=bool(row["is_critical"]),
        confidence_prior=row["confidence_prior"],
    )


def _agent_output_from_row(row: dict) -> AgentOutputFull:
    evidence = [EvidenceCitation(**item) for item in json.loads(row["evidence_cited"])]
    return AgentOutputFull(
        agent_role=AgentRole(row["agent_role"]),
        round_num=row["round_num"],
        verdict=AgentVerdict(row["verdict"]),
        reasoning=row["reasoning"],
        evidence_cited=evidence,
        confidence_internal=row["confidence_internal"],
    )


def export_grpo_dataset(run_id: str, db_path: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = fetch_all("SELECT * FROM queries WHERE run_id = ?", (run_id,), db_path)

    rows_a = []
    rows_b = []

    for query_row in queries:
        query_id = query_row["query_id"]
        user_query = query_row["user_query"]
        rag_chunks = json.loads(query_row["rag_chunks"])

        claim_rows = fetch_all(
            "SELECT * FROM claims WHERE query_id = ? ORDER BY claim_index",
            (query_id,),
            db_path,
        )
        for claim_row in claim_rows:
            claim = _claim_from_row(claim_row)
            claim_id = claim.claim_id

            verdict_rows = fetch_all(
                "SELECT * FROM judge_verdicts WHERE claim_id = ? ORDER BY timestamp LIMIT 1",
                (claim_id,),
                db_path,
            )
            v_label = verdict_rows[0]["v_label"] if verdict_rows else None
            judge_confidence = verdict_rows[0]["judge_confidence"] if verdict_rows else None

            output_rows = fetch_all(
                "SELECT * FROM agent_outputs WHERE claim_id = ? ORDER BY round_num, agent_role",
                (claim_id,),
                db_path,
            )
            outputs_by_role_round: dict = {"agent_a": {}, "agent_b": {}}
            for row in output_rows:
                out = _agent_output_from_row(row)
                outputs_by_role_round[row["agent_role"]][row["round_num"]] = (out, row["raw_response"])

            for role, system_prompt in [("agent_a", AGENT_A_SYSTEM), ("agent_b", AGENT_B_SYSTEM)]:
                for round_num in [0, 1]:
                    if round_num not in outputs_by_role_round[role]:
                        continue
                    out, raw_response = outputs_by_role_round[role][round_num]

                    if round_num == 0:
                        prompt_text = f"{system_prompt}\n{round0_user(claim, rag_chunks, user_query)}"
                    else:
                        peer_role = "agent_b" if role == "agent_a" else "agent_a"
                        peer_label = "Debater 2" if role == "agent_a" else "Debater 1"
                        if 0 in outputs_by_role_round[peer_role]:
                            peer_out, _ = outputs_by_role_round[peer_role][0]
                            peer_stripped = strip_for_peer(peer_out, peer_label)
                        else:
                            continue
                        prompt_text = f"{system_prompt}\n{round1_user(claim, rag_chunks, user_query, peer_stripped)}"

                    valid_completion = bool(raw_response and len(raw_response.strip()) > 10)
                    try:
                        json.loads(raw_response)
                    except Exception:
                        valid_completion = False

                    row_data = {
                        "prompt": prompt_text,
                        "completion": raw_response,
                        "valid_completion": valid_completion,
                        "v_label": v_label,
                        "role": role,
                        "round": round_num,
                        "claim_id": claim_id,
                        "query_id": query_id,
                        "is_critical": claim.is_critical,
                        "is_material": claim.is_material,
                        "judge_confidence": judge_confidence,
                    }
                    if role == "agent_a":
                        rows_a.append(row_data)
                    else:
                        rows_b.append(row_data)

    path_a = out_dir / "agent_a_grpo.jsonl"
    path_b = out_dir / "agent_b_grpo.jsonl"
    with open(path_a, "w") as f:
        for row in rows_a:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(path_b, "w") as f:
        for row in rows_b:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "agent_a_rows": len(rows_a),
        "agent_b_rows": len(rows_b),
        "output_dir": str(out_dir),
        "files": [str(path_a), str(path_b)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GRPO dataset from SQLite")
    parser.add_argument("--db", default=None, help="SQLite DB path")
    parser.add_argument("--run-id", required=True, help="Run ID to export")
    parser.add_argument("--out-dir", default="grpo_export", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db_path = args.db or config.SQLITE_DB_PATH
    result = export_grpo_dataset(args.run_id, db_path, Path(args.out_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
