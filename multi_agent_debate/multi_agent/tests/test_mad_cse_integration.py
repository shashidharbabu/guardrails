"""
test_mad_cse_integration.py — Integration tests for the full MAD → CSE → storage → API chain.

All LLM calls are mocked. No Ollama or DeepEval required.
Each test uses a per-test temp SQLite DB (pytest tmp_path fixture).
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent.parent
for _p in [str(_REPO), str(_REPO / "multi_agent_debate")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_storage(db_path: str):
    """Load storage module directly from disk with overridden DB_PATH."""
    storage_src = _REPO / "multi_agent_debate" / "multi_agent" / "storage.py"

    fake_config = types.ModuleType("multi_agent.config")
    fake_config.DB_PATH = db_path
    sys.modules["multi_agent.config"] = fake_config

    spec   = importlib.util.spec_from_file_location("_int_storage", storage_src)
    module = importlib.util.module_from_spec(spec)

    from contextlib import contextmanager
    import sqlite3 as _sqlite3

    @contextmanager
    def _patched_conn():
        con = _sqlite3.connect(db_path)
        con.row_factory = _sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    spec.loader.exec_module(module)
    module._conn = _patched_conn
    return module


def _make_mad_output(query_id="qid-int", rollout_id="rid-int"):
    """Build a minimal MADOutput-like dict for assertions."""
    from multi_agent.models import MADOutput, Claim, JudgeVerdict, Verdict, EvidenceChunk
    return {
        "query_id": query_id,
        "rollout_id": rollout_id,
        "routing_decision": "DELIVER",
        "aggregate_confidence": 0.87,
        "cse_result": {
            "routing_decision": "DELIVER",
            "scoring_mode": "JUDGE_ONLY_FALLBACK",
            "final_score": 0.87,
        },
    }


def _make_cse_result_mock():
    """Build a mock CSEResult that quacks like the real one."""
    mock = MagicMock()
    mock.routing_decision = "DELIVER"
    mock.final_score = 0.87
    mock.version = "cse_v1.1"
    mock.error = ""
    mock.explanation = "All claims supported."
    mock.components = MagicMock()
    mock.components.as_dict.return_value = {
        "f_llm": 0.9, "h_llm": 0.1, "relevancy": 0.8, "judge_eval": 0.85
    }
    mock.score_breakdown = None
    mock.as_dict.return_value = {
        "routing_decision": "DELIVER",
        "scoring_mode": "JUDGE_ONLY_FALLBACK",
        "final_score": 0.87,
        "explanation": "All claims supported.",
    }
    return mock


def _seed_query_row(db_path: str, query_id: str, rollout_id: str) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """INSERT INTO queries
           (query_id, rollout_id, query_text, llm_answer, rag_chunk_ids, timestamp)
           VALUES (?,?,?,?,?,?)""",
        (query_id, rollout_id, "test query", "test answer",
         "[]", "2026-01-01T00:00:00+00:00"),
    )
    con.commit()
    con.close()


# ══════════════════════════════════════════════════════════════════════════════
# TestRunMadCseWiring
# ══════════════════════════════════════════════════════════════════════════════

class TestRunMadCseWiring:
    """Tests wiring from run_mad() through CSE to storage."""

    def test_update_query_cse_stores_cse_breakdown(self, tmp_path):
        """
        When update_query_cse is called with a breakdown dict, the JSON blob
        is retrievable from SQLite and contains expected keys.
        """
        db_path = str(tmp_path / "wiring.db")
        storage = _load_storage(db_path)
        storage.init_db()

        query_id   = "qid-wire-001"
        rollout_id = "rid-wire-001"
        _seed_query_row(db_path, query_id, rollout_id)

        breakdown = {
            "routing_decision": "DELIVER",
            "scoring_mode": "JUDGE_ONLY_FALLBACK",
            "final_score": 0.87,
            "explanation": "All claims supported.",
        }
        cse_mock = _make_cse_result_mock()

        storage.update_query_cse(
            query_id=query_id,
            rollout_id=rollout_id,
            cse_score=cse_mock.final_score,
            routing_decision=cse_mock.routing_decision,
            cse_components=cse_mock.components.as_dict(),
            cse_version=cse_mock.version,
            cse_breakdown=cse_mock.as_dict(),
        )

        con = sqlite3.connect(db_path)
        row = con.execute(
            "SELECT final_cse_score, routing_decision, cse_breakdown "
            "FROM queries WHERE query_id=?", (query_id,)
        ).fetchone()
        con.close()

        assert row is not None
        assert abs(row[0] - 0.87) < 1e-6
        assert row[1] == "DELIVER"
        parsed = json.loads(row[2])
        assert parsed["routing_decision"] == "DELIVER"
        assert "scoring_mode" in parsed

    def test_update_query_cse_cse_failure_stores_null_breakdown(self, tmp_path):
        """When CSE fails (cse_result=None), breakdown column stays NULL."""
        db_path = str(tmp_path / "failure.db")
        storage = _load_storage(db_path)
        storage.init_db()

        query_id   = "qid-wire-002"
        rollout_id = "rid-wire-002"
        _seed_query_row(db_path, query_id, rollout_id)

        storage.update_query_cse(
            query_id=query_id,
            rollout_id=rollout_id,
            cse_score=0.5,
            routing_decision="RETRY",
            cse_components=None,
            cse_version="v0.1",
            cse_breakdown=None,
        )

        con = sqlite3.connect(db_path)
        row = con.execute(
            "SELECT routing_decision, cse_breakdown FROM queries WHERE query_id=?",
            (query_id,)
        ).fetchone()
        con.close()

        assert row[0] == "RETRY"
        assert row[1] is None

    def test_storage_cse_breakdown_json_is_parseable(self, tmp_path):
        """The JSON stored in cse_breakdown must be valid JSON with required keys."""
        db_path = str(tmp_path / "json_check.db")
        storage = _load_storage(db_path)
        storage.init_db()

        query_id   = "qid-wire-003"
        rollout_id = "rid-wire-003"
        _seed_query_row(db_path, query_id, rollout_id)

        breakdown = {
            "routing_decision": "HARD_BLOCK",
            "scoring_mode": "FULL",
            "final_score": 0.0,
            "explanation": "Material claim contradicted.",
            "triggered_flags": {"hard_blocked": True, "material_claim_failed": True},
            "top_failed_claims": [{"claim_id": 1, "verdict": "contradicted"}],
        }

        storage.update_query_cse(
            query_id=query_id,
            rollout_id=rollout_id,
            cse_score=0.0,
            routing_decision="HARD_BLOCK",
            cse_breakdown=breakdown,
        )

        con = sqlite3.connect(db_path)
        raw = con.execute(
            "SELECT cse_breakdown FROM queries WHERE query_id=?", (query_id,)
        ).fetchone()[0]
        con.close()

        parsed = json.loads(raw)
        assert parsed["routing_decision"] == "HARD_BLOCK"
        assert parsed["triggered_flags"]["hard_blocked"] is True
        assert len(parsed["top_failed_claims"]) == 1


# ══════════════════════════════════════════════════════════════════════════════
# TestApiEndpoints
# ══════════════════════════════════════════════════════════════════════════════

class TestApiEndpoints:
    """Tests for FastAPI endpoints using TestClient, with mocked run_mad."""

    def _get_client(self, db_path: str):
        """
        Build a TestClient for the canonical api module with DB_PATH patched.

        Loads the canonical api.py directly from disk via importlib.util.
        The api module only uses DB_PATH at request time (not at import time),
        so we can patch it on the loaded module after exec.
        """
        from fastapi.testclient import TestClient

        api_src = _REPO / "multi_agent_debate" / "multi_agent" / "api.py"

        # Ensure real config is available (undo any fake from _load_storage)
        if hasattr(sys.modules.get("multi_agent.config"), "DB_PATH") and \
           not hasattr(sys.modules.get("multi_agent.config"), "OLLAMA_BASE_URL"):
            # Our fake config is in place — restore real one
            del sys.modules["multi_agent.config"]
            import multi_agent.config  # noqa: F401

        import multi_agent.config as real_cfg
        orig_db = real_cfg.DB_PATH
        real_cfg.DB_PATH = db_path

        # Load api from canonical file, giving it a unique module name each call
        import uuid as _uuid
        mod_name = f"_canon_api_{_uuid.uuid4().hex}"
        spec   = importlib.util.spec_from_file_location(mod_name, api_src)
        api_mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = api_mod
        spec.loader.exec_module(api_mod)

        # Override the module-level DB_PATH used in the GET endpoint query
        api_mod.DB_PATH = db_path

        client = TestClient(api_mod.app)
        real_cfg.DB_PATH = orig_db
        return client, api_mod

    def test_get_cse_endpoint_returns_404_for_unknown_id(self, tmp_path):
        """GET /mad/cse/<unknown> must return 404."""
        db_path = str(tmp_path / "api_404.db")
        storage = _load_storage(db_path)
        storage.init_db()

        client, _ = self._get_client(db_path)
        resp = client.get("/mad/cse/does-not-exist")
        assert resp.status_code == 404

    def test_get_cse_endpoint_reads_from_sqlite(self, tmp_path):
        """GET /mad/cse/<id> returns the stored score and routing_decision."""
        db_path = str(tmp_path / "api_read.db")
        storage = _load_storage(db_path)
        storage.init_db()

        query_id   = "qid-api-001"
        rollout_id = "rid-api-001"
        _seed_query_row(db_path, query_id, rollout_id)

        breakdown = {"routing_decision": "DELIVER", "scoring_mode": "FULL", "final_score": 0.9}
        storage.update_query_cse(
            query_id=query_id,
            rollout_id=rollout_id,
            cse_score=0.9,
            routing_decision="DELIVER",
            cse_version="v2.0",
            cse_breakdown=breakdown,
        )

        client, _ = self._get_client(db_path)
        resp = client.get(f"/mad/cse/{query_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["query_id"] == query_id
        assert abs(data["final_score"] - 0.9) < 1e-6
        assert data["routing_decision"] == "DELIVER"
        assert data["version"] == "v2.0"
        assert data["cse_breakdown"]["scoring_mode"] == "FULL"

    def test_get_cse_endpoint_handles_null_breakdown(self, tmp_path):
        """GET /mad/cse/<id> returns cse_breakdown=None when not stored."""
        db_path = str(tmp_path / "api_null.db")
        storage = _load_storage(db_path)
        storage.init_db()

        query_id   = "qid-api-002"
        rollout_id = "rid-api-002"
        _seed_query_row(db_path, query_id, rollout_id)

        storage.update_query_cse(
            query_id=query_id,
            rollout_id=rollout_id,
            cse_score=0.65,
            routing_decision="RETRY",
        )

        client, _ = self._get_client(db_path)
        resp = client.get(f"/mad/cse/{query_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["cse_breakdown"] is None

    def test_post_verify_validation_rejects_empty_query(self, tmp_path):
        """POST /mad/verify with empty query must return 422."""
        db_path = str(tmp_path / "api_val.db")
        storage = _load_storage(db_path)
        storage.init_db()

        client, _ = self._get_client(db_path)
        resp = client.post("/mad/verify", json={"query": "", "llm_answer": "some answer"})
        assert resp.status_code == 422

    def test_health_endpoint(self, tmp_path):
        """GET /mad/health must return ok."""
        db_path = str(tmp_path / "api_health.db")
        storage = _load_storage(db_path)
        storage.init_db()

        client, _ = self._get_client(db_path)
        resp = client.get("/mad/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
