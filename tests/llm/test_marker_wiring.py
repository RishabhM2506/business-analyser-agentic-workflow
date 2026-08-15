"""Proves the `llm` pytest marker and its CI job are wired correctly.

There is no real LLM-dependent code yet (Phase 3) — `app/models.py`'s
`MockLLM` is a stub. This test exists purely so the `llm` marker, the
`--strict-markers` pytest config, and the dedicated `llm` CI job all have
something to run and pass against, per docs/PLAN.md §7's "llm-marked" test
layer and the master brief §6's mandatory MockLLM/cassette-replay pattern —
proven here at the wiring level, filled in with real cassette tests once
`describe_item`/`summarize` exist.
"""

from __future__ import annotations

import pytest


@pytest.mark.llm
def test_llm_marker_is_registered_and_runnable() -> None:
    """Trivial by design: real cassette-based LLM tests land in Phase 3
    alongside `app/nodes/describe_item.py` and `app/nodes/summarize.py`."""
    assert True
