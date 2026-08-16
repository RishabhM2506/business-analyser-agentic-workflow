"""`retrieve_description` node: calls `KnowledgeProvider.retrieve()` (v1:
static taxonomy CSV lookup) to fetch the source text that `describe_item`
will write prose about (docs/PLAN.md §2.2). No model call here — retrieval
is deterministic.
"""

from __future__ import annotations

from typing import Any

from app.knowledge.provider import StaticKnowledgeProvider
from app.state import AnalysisState, has_error

_default_provider = StaticKnowledgeProvider()


async def retrieve_description(state: AnalysisState) -> dict[str, Any]:
    """Fetch taxonomy text for `state["query"].hs_code`; writes
    `taxonomy_text`.

    `query.hs_code` was already confirmed to be a known HS6 code by
    `validate_query`'s allowlist check, so `KeyError` here would indicate a
    genuine inconsistency between the two — it is deliberately NOT swallowed
    (docs/PLAN.md §3.2: a real failure gets a defined error fallback, not a
    silently wrong render; `app/main.py`'s endpoint handler catches any
    unexpected exception from the graph and maps it to a generic
    `ErrorResponse` rather than leaking a stack trace).
    """
    if has_error(state):
        return {}
    query = state.get("query")
    if query is None:
        return {}
    taxonomy_text = await _default_provider.retrieve(query.hs_code)
    return {"taxonomy_text": taxonomy_text}
