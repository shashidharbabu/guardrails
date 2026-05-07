"""
test_storage.py — Unit tests for the MAD SQLite storage layer.

Tests:
  - update_query_cse stores and retrieves the cse_breakdown JSON blob
  - update_query_cse is backward-compatible when cse_breakdown is omitted
  - init_db is idempotent (safe to call multiple times)
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

# Absolute path to the canonical storage module on disk
_STORAGE_SRC = Path(__file__).resolve().parent.parent / "storage.py"
_CONFIG_SRC  = Path(__file__).resolve().parent.parent / "config.py"
_MODELS_SRC  = Path(__file__).resolve().parent.parent / "models.py"


def _load_storage(db_path: str):
    """
    Load storage.py from disk with DB_PATH overridden to db_path.
    Uses importlib.util so the package-dir mapping in pyproject.toml
    does not interfere — we load the file directly.
    """
    # 1. Load models into a fresh namespace (needed by storage imports)
    _load_module_from_file("_test_multi_agent_models", _MODELS_SRC)

    # 2. Create a fake config module with our temp DB_PATH
    fake_config = types.ModuleType("_test_multi_agent_config")
    fake_config.DB_PATH = db_path
    sys.modules["_test_multi_agent_config"] = fake_config

    # 3. Load storage.py, patching its imports at load time
    spec   = importlib.util.spec_from_file_location("_test_storage", _STORAGE_SRC)
    module = importlib.util.module_from_spec(spec)

    # Inject our override modules before exec so storage.py's imports resolve
    sys.modules["multi_agent.config"] = fake_config
    # Ensure models is available under the name storage.py uses
    sys.modules["multi_agent.models"] = sys.modules["_test_multi_agent_models"]

    spec.loader.exec_module(module)
    # Override the DB_PATH constant that was captured at module level
    module.DB_PATH = db_path  # type: ignore[attr-defined]

    # Patch the _conn context manager to use our db_path
    import sqlite3 as _sqlite3
    from contextlib import contextmanager

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

    module._conn = _patched_conn  # type: ignore[attr-defined]

    return module


def _load_module_from_file(name: str, path: Path):
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def storage(tmp_path):
    """Yield a storage module wired to a fresh temp SQLite DB."""
    db_file = str(tmp_path / "test_mad.db")
    mod = _load_storage(db_file)
    mod.init_db()
    yield mod, db_file


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestInitDb:
    def test_init_db_idempotent(self, tmp_path):
        """Calling init_db() twice must not raise and must produce a valid schema."""
        db_file = str(tmp_path / "idempotent.db")
        mod = _load_storage(db_file)
        mod.init_db()   # first call
        mod.init_db()   # second call — must not raise

        con = sqlite3.connect(db_file)
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        con.close()

        assert "queries" in tables
        assert "claims" in tables
        assert "attacks" in tables
        assert "judge_verdicts" in tables

    def test_init_db_creates_cse_breakdown_column(self, tmp_path):
        """init_db must create the cse_breakdown column in the queries table."""
        db_file = str(tmp_path / "breakdown_col.db")
        mod = _load_storage(db_file)
        mod.init_db()

        con = sqlite3.connect(db_file)
        cols = {r[1] for r in con.execute("PRAGMA table_info(queries)").fetchall()}
        con.close()

        assert "cse_breakdown" in cols


class TestUpdateQueryCse:
    def _seed_query(self, db_file: str, query_id: str, rollout_id: str) -> None:
        """Insert a minimal query row so update_query_cse has a row to update."""
        con = sqlite3.connect(db_file)
        con.execute(
            """INSERT INTO queries
               (query_id, rollout_id, query_text, llm_answer, rag_chunk_ids, timestamp)
               VALUES (?,?,?,?,?,?)""",
            (query_id, rollout_id, "test query", "test answer",
             "[]", "2026-01-01T00:00:00+00:00"),
        )
        con.commit()
        con.close()

    def test_update_query_cse_stores_breakdown(self, storage):
        """update_query_cse must persist the full cse_breakdown JSON blob."""
        mod, db_file = storage
        query_id   = "qid-001"
        rollout_id = "rid-001"
        self._seed_query(db_file, query_id, rollout_id)

        breakdown = {
            "routing_decision": "DELIVER",
            "scoring_mode": "FULL",
            "aggregate_score": 0.87,
            "explanation": "All claims well-supported.",
            "triggered_flags": {"hard_blocked": False},
            "top_failed_claims": [],
        }
        components = {"f_llm": 0.9, "h_llm": 0.1, "relevancy": 0.8, "judge_eval": 0.85}

        mod.update_query_cse(
            query_id=query_id,
            rollout_id=rollout_id,
            cse_score=0.87,
            routing_decision="DELIVER",
            cse_components=components,
            cse_version="v2.0",
            cse_breakdown=breakdown,
        )

        con = sqlite3.connect(db_file)
        row = con.execute(
            "SELECT final_cse_score, routing_decision, cse_version, cse_breakdown "
            "FROM queries WHERE query_id=? AND rollout_id=?",
            (query_id, rollout_id),
        ).fetchone()
        con.close()

        assert row is not None
        assert abs(row[0] - 0.87) < 1e-6
        assert row[1] == "DELIVER"
        assert row[2] == "v2.0"

        parsed = json.loads(row[3])
        assert parsed["routing_decision"] == "DELIVER"
        assert parsed["scoring_mode"] == "FULL"
        assert abs(parsed["aggregate_score"] - 0.87) < 1e-6

    def test_update_query_cse_backward_compatible_no_breakdown(self, storage):
        """update_query_cse called without cse_breakdown must still work."""
        mod, db_file = storage
        query_id   = "qid-002"
        rollout_id = "rid-002"
        self._seed_query(db_file, query_id, rollout_id)

        mod.update_query_cse(
            query_id=query_id,
            rollout_id=rollout_id,
            cse_score=0.65,
            routing_decision="RETRY",
        )

        con = sqlite3.connect(db_file)
        row = con.execute(
            "SELECT final_cse_score, routing_decision, cse_breakdown "
            "FROM queries WHERE query_id=? AND rollout_id=?",
            (query_id, rollout_id),
        ).fetchone()
        con.close()

        assert row is not None
        assert abs(row[0] - 0.65) < 1e-6
        assert row[1] == "RETRY"
        assert row[2] is None

    def test_update_query_cse_stores_component_scores(self, storage):
        """Legacy component scores (f_llm, h_llm, relevancy, judge_eval) must be stored."""
        mod, db_file = storage
        query_id   = "qid-003"
        rollout_id = "rid-003"
        self._seed_query(db_file, query_id, rollout_id)

        components = {"f_llm": 0.88, "h_llm": 0.12, "relevancy": 0.75, "judge_eval": 0.90}
        mod.update_query_cse(
            query_id=query_id,
            rollout_id=rollout_id,
            cse_score=0.85,
            routing_decision="DELIVER",
            cse_components=components,
        )

        con = sqlite3.connect(db_file)
        row = con.execute(
            "SELECT cse_f_llm, cse_h_llm, cse_relevancy, cse_judge_eval "
            "FROM queries WHERE query_id=? AND rollout_id=?",
            (query_id, rollout_id),
        ).fetchone()
        con.close()

        assert row is not None
        assert abs(row[0] - 0.88) < 1e-6
        assert abs(row[1] - 0.12) < 1e-6
        assert abs(row[2] - 0.75) < 1e-6
        assert abs(row[3] - 0.90) < 1e-6

    def test_update_query_cse_breakdown_as_none_dict(self, storage):
        """Passing cse_breakdown=None explicitly stores NULL in the DB."""
        mod, db_file = storage
        query_id   = "qid-004"
        rollout_id = "rid-004"
        self._seed_query(db_file, query_id, rollout_id)

        mod.update_query_cse(
            query_id=query_id,
            rollout_id=rollout_id,
            cse_score=0.40,
            routing_decision="HUMAN_REVIEW",
            cse_breakdown=None,
        )

        con = sqlite3.connect(db_file)
        row = con.execute(
            "SELECT cse_breakdown FROM queries WHERE query_id=? AND rollout_id=?",
            (query_id, rollout_id),
        ).fetchone()
        con.close()

        assert row[0] is None
