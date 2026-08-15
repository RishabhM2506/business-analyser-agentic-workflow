"""`fetch_imports` / `fetch_exports` nodes: fan out from `validate_query`
and fan into `aggregate` as a parallel superstep — two independent,
read-only HTTP calls with no data dependency on each other (docs/PLAN.md
§2.2).

# TODO(Phase 3): implement both, calling
# `app.tools.comtrade_client.ComtradeClient.fetch_flow` through the
# tool-result cache (`app.cache.tool_cache.ToolCache`).
"""

from __future__ import annotations

from typing import Any

from app.state import AnalysisState


async def fetch_imports(state: AnalysisState) -> dict[str, Any]:
    """Fetch import-flow records for `state["query"].hs_code`; writes
    `raw_imports`."""
    raise NotImplementedError  # TODO(Phase 3): implement.


async def fetch_exports(state: AnalysisState) -> dict[str, Any]:
    """Fetch export-flow records for `state["query"].hs_code`; writes
    `raw_exports`."""
    raise NotImplementedError  # TODO(Phase 3): implement.
