"""`describe_item` node: MODEL_UTILITY call, schema-constrained output
(docs/PLAN.md §2.2, §5.1 call #1). Cacheable purely by
`(hs_code, prompt_version)` — never depends on year range.

# TODO(Phase 3): implement — load `prompts/describe_item.md`, call
# `app.models.get_model_for_role("utility", ...)`, validate the output
# against the expected shape.
"""

from __future__ import annotations

from typing import Any

from app.state import AnalysisState


async def describe_item(state: AnalysisState) -> dict[str, Any]:
    """Write `item_description` from `state["taxonomy_text"]` via
    MODEL_UTILITY."""
    raise NotImplementedError  # TODO(Phase 3): implement.
