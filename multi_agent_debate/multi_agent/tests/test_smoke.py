"""
test_smoke.py — Smoke tests: schema checks, model fields, import sanity.

These tests verify that the module structure, Pydantic schemas, and SQLite
schema are correct — without running any LLM inference.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent.parent
for _p in [str(_REPO), str(_REPO / "multi_agent_debate")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_storage(db_path: str):
    """Load storage module with patched DB_PATH."""
    storage_src = _REPO / "multi_agent_debate" / "multi_agent" / "storage.py"

    fake_config = types.ModuleType("multi_agent.config")
    fake_config.DB_PATH = db_path
    sys.modules["multi_agent.config"] = fake_config

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

    spec   = importlib.util.spec_from_file_location("_smoke_storage", storage_src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._conn = _patched_conn
    return module


# ══════════════════════════════════════════════════════════════════════════════
# TestModelFields
# ══════════════════════════════════════════════════════════════════════════════

class TestModelFields:
    def test_mad_output_has_query_id_field(self):
        """MADOutput must have a query_id field with default ''."""
        from multi_agent.models import MADOutput
        fields = MADOutput.model_fields  # Pydantic V2
        assert "query_id" in fields
        assert fields["query_id"].default == ""

    def test_mad_output_has_rollout_id_field(self):
        """MADOutput must have a rollout_id field with default ''."""
        from multi_agent.models import MADOutput
        fields = MADOutput.model_fields
        assert "rollout_id" in fields
        assert fields["rollout_id"].default == ""

    def test_mad_output_has_cse_result_field(self):
        """MADOutput must have a cse_result field that defaults to None."""
        from multi_agent.models import MADOutput
        fields = MADOutput.model_fields
        assert "cse_result" in fields
        assert fields["cse_result"].default is None

    def test_claim_has_severity_field(self):
        """Claim must have a severity field that defaults to None."""
        from multi_agent.models import Claim
        c = Claim(claim_id=1, claim_text="test", is_material=True)
        assert hasattr(c, "severity")
        assert c.severity is None

    def test_mad_output_defaults_backward_compatible(self):
        """Constructing MADOutput without new fields uses correct defaults."""
        from multi_agent.models import MADOutput
        out = MADOutput(
            query="q",
            llm_answer="a",
            claims=[],
            debate_cycles=[],
            evidence_pool=[],
            judge_verdicts=[],
            correction_signal=None,
            routing_decision="DELIVER",
            aggregate_confidence=0.9,
            debate_transcript="t",
        )
        assert out.query_id == ""
        assert out.rollout_id == ""
        assert out.cse_result is None


# ══════════════════════════════════════════════════════════════════════════════
# TestDbSchema
# ══════════════════════════════════════════════════════════════════════════════

class TestDbSchema:
    def test_init_db_creates_cse_breakdown_column(self, tmp_path):
        """The queries table must have a cse_breakdown TEXT column."""
        db_path = str(tmp_path / "schema_check.db")
        storage = _load_storage(db_path)
        storage.init_db()

        con = sqlite3.connect(db_path)
        cols = {r[1] for r in con.execute("PRAGMA table_info(queries)").fetchall()}
        con.close()

        assert "cse_breakdown" in cols
        assert "final_cse_score" in cols
        assert "routing_decision" in cols
        assert "cse_version" in cols

    def test_init_db_idempotent(self, tmp_path):
        """Calling init_db() twice must not raise."""
        db_path = str(tmp_path / "idempotent_smoke.db")
        storage = _load_storage(db_path)
        storage.init_db()
        storage.init_db()  # must not raise

        con = sqlite3.connect(db_path)
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        con.close()

        assert {"queries", "claims", "attacks", "judge_verdicts"}.issubset(tables)


# ══════════════════════════════════════════════════════════════════════════════
# TestCseImport
# ══════════════════════════════════════════════════════════════════════════════

class TestCseImport:
    def test_confidence_scorer_importable(self):
        """ConfidenceScorer must be importable without external services."""
        from confidence.scorer import ConfidenceScorer
        assert ConfidenceScorer is not None

    def test_cse_result_as_dict_has_all_keys(self):
        """CSEResult.as_dict() must contain all required keys."""
        from confidence.scorer import ConfidenceScorer
        from multi_agent.models import Claim, JudgeVerdict

        scorer = ConfidenceScorer()
        scorer._run_deepeval = lambda q, a, c: (None, None, None, "smoke_test")

        claims = [
            Claim(claim_id=1, claim_text="HIPAA requires encryption.", is_material=True)
        ]
        verdicts = [
            JudgeVerdict(
                claim_id=1, claim_text="HIPAA requires encryption.",
                is_material=True, score=1.0, reasoning="Supported."
            )
        ]

        result = scorer.score("query", "answer", ["context chunk"], claims, verdicts)
        d = result.as_dict()

        required_keys = {
            "final_score", "routing_decision", "scoring_mode",
            "score_breakdown", "triggered_flags", "top_failed_claims",
            "explanation", "metadata", "version",
        }
        assert required_keys.issubset(d.keys()), (
            f"Missing keys: {required_keys - d.keys()}"
        )

    def test_cse_routing_decisions_are_valid_strings(self):
        """All RoutingDecision enum values must be non-empty strings."""
        from confidence.cse_types import RoutingDecision
        for rd in RoutingDecision:
            assert isinstance(rd.value, str) and rd.value

    def test_scoring_modes_are_valid_strings(self):
        """All ScoringMode enum values must be non-empty strings."""
        from confidence.cse_types import ScoringMode
        for sm in ScoringMode:
            assert isinstance(sm.value, str) and sm.value
