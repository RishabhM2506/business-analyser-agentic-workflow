"""`summarize` node: MODEL_ANALYSIS call over the pre-aggregated table only
— never raw Comtrade JSON (docs/PLAN.md §2.2, §5.1 call #2, master brief
§7.1). Output must pass `app.guardrails.check_numbers_grounded` before
`assemble_response` runs.

# TODO(Phase 3): implement — load `prompts/summarize.md`, call
# `app.models.get_model_for_role("analysis", ...)`, run the output
# guardrail before returning.
"""

from __future__ import annotations

from typing import Any

from app.state import AnalysisState


async def summarize(state: AnalysisState) -> dict[str, Any]:
    """Write `analytical_summary` from `imports_table`/`exports_table` via
    MODEL_ANALYSIS."""
    raise NotImplementedError  # TODO(Phase 3): implement.
