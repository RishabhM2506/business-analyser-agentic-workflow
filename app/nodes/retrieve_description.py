"""`retrieve_description` node: calls `KnowledgeProvider.retrieve()` (v1:
static taxonomy CSV lookup) to fetch the source text that `describe_item`
will write prose about (docs/PLAN.md §2.2). No model call here — retrieval
is deterministic.

# TODO(Phase 3): implement — call
# `app.knowledge.provider.StaticKnowledgeProvider.retrieve`.
"""

from __future__ import annotations

from typing import Any

from app.state import AnalysisState


async def retrieve_description(state: AnalysisState) -> dict[str, Any]:
    """Fetch taxonomy text for `state["query"].hs_code`; writes
    `taxonomy_text`."""
    raise NotImplementedError  # TODO(Phase 3): implement.
