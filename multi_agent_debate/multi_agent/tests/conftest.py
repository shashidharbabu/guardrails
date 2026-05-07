"""
conftest.py — Make the test files in this directory importable by pytest
without going through the multi_agent package namespace.

pytest collects tests by path; when it finds multi_agent_debate/multi_agent/tests/
it tries to resolve them as multi_agent.tests.* (via pyproject.toml package-dir).
This conftest inserts the multi_agent_debate directory onto sys.path so that
pytest's rootdir-relative import also works.
"""
import sys
from pathlib import Path

# Insert multi_agent_debate so that `from multi_agent import ...` resolves to
# multi_agent_debate/multi_agent/ directly (not via the installed package mapping)
_mad_dir = Path(__file__).resolve().parent.parent.parent
if str(_mad_dir) not in sys.path:
    sys.path.insert(0, str(_mad_dir))
