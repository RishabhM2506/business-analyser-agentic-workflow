"""`AnalysisState` — the LangGraph state schema for the v1 workflow.

Field set mirrors the per-node input/output columns in docs/PLAN.md §2.2
exactly, so each node's signature (`app/nodes/*.py`) can be read directly
against that table.

# TODO(Phase 3): wire this into `app/graph.py`'s `StateGraph(AnalysisState)`
# once the nodes have real bodies.
"""

from __future__ import annotations

from typing import TypedDict

from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.schemas.response import TradeAnalysisResponse, TradeTable
from app.tools.comtrade_client import ComtradeRecord


class AnalysisState(TypedDict, total=False):
    """Shared state threaded through every node of the v1 workflow.

    `total=False`: a given superstep only ever writes a subset of these
    keys (docs/PLAN.md §2.2) — LangGraph merges partial node returns into
    this dict across supersteps.
    """

    query: TradeQuery
    error: ErrorResponse

    raw_imports: list[ComtradeRecord]
    raw_exports: list[ComtradeRecord]

    imports_table: TradeTable
    exports_table: TradeTable

    taxonomy_text: str

    item_description: str
    analytical_summary: str

    response: TradeAnalysisResponse
