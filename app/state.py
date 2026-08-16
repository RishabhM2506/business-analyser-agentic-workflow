"""`AnalysisState` — the LangGraph state schema for the v1 workflow.

Field set mirrors the per-node input/output columns in docs/PLAN.md §2.2
exactly, so each node's signature (`app/nodes/*.py`) can be read directly
against that table.
"""

from __future__ import annotations

import uuid
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

    # Seeded by the caller (`app/main.py`) at `.ainvoke()` time from the same
    # UUID4 already bound as the HTTP request's `X-Request-ID`
    # (`request_id_middleware`) — not written by any node, only read, so any
    # `ErrorResponse` a node constructs correlates with the request's own
    # structured logs rather than minting an independent, uncorrelated id.
    trace_id: str

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


def get_or_mint_trace_id(state: AnalysisState) -> str:
    """`state["trace_id"]` if the caller seeded one (the normal case — see
    the field's docstring above), else a freshly-minted UUID4 — shared by
    every node that constructs an `ErrorResponse`, so there's exactly one
    fallback policy instead of five slightly-different copies."""
    return state.get("trace_id") or str(uuid.uuid4())


def has_error(state: AnalysisState) -> bool:
    """True iff an earlier node already wrote `error` — every node after
    `validate_query` checks this first and no-ops if so (docs/PLAN.md §2.2:
    v1 has no conditional edges, so every node in the fixed pipeline runs
    regardless; a failed validation is a no-op for the rest of the
    pipeline, not a graph-level branch)."""
    return state.get("error") is not None
