"""Proves the `llm` pytest marker and its CI job are wired correctly.

Kept alongside the real cassette-based tests in this same directory
(`tests/llm/test_describe_item.py`) as a minimal, always-passing proof that
the `llm` marker, `--strict-markers` pytest config, and the dedicated `llm`
CI job stay wired even if every real cassette test were ever skipped or
removed — per docs/PLAN.md §7's "llm-marked" test layer and the master
brief §6's mandatory MockLLM/cassette-replay pattern.
"""

from __future__ import annotations

import pytest


@pytest.mark.llm
def test_llm_marker_is_registered_and_runnable() -> None:
    """Trivial by design — see module docstring."""
    assert True
