"""`validate_query` node: parses the raw request body into `TradeQuery` and
runs the `hs_code`-against-taxonomy allowlist check (docs/PLAN.md §2.2,
§6). Runs before any node that could reach the network or a model —
PLAN.md §1.1's "nothing reaches a model before validation" ordering.

`state["query"]` arrives already shape-valid (`app/main.py`'s endpoint
parses the raw POST body into `TradeQuery` via Pydantic before invoking the
graph at all — docs/PLAN.md §1.1's Gateway -> Guard -> Graph ordering: the
Gateway layer's Pydantic validation is a *different* check than this node's
semantic one). This node's job is the two checks Pydantic's field types
alone cannot express: hs_code taxonomy *membership* (an allowlist, not just
the `^\\d{6}$` shape regex already enforced on the schema) and resolving
the year-range defaults documented on `TradeQuery` itself.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.guardrails import check_hs_code_allowlisted
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.state import AnalysisState, get_or_mint_trace_id

# TradeQuery's own documented contract (app/schemas/query.py):
#   year_start: None = latest available - 4
#   year_end:   None = latest available
# -> a 5-year inclusive window (year_end - 4 ..= year_end).
YEAR_WINDOW = 5


def _latest_available_year(*, now: datetime | None = None) -> int:
    """Heuristic for "latest available" (see `TradeQuery`'s field docstring):
    the last fully-completed calendar year. UN Comtrade annual data lags
    well behind the current calendar year — even the *prior* year is often
    still provisional at the country-partner level
    (`docs/PHASE0-FINDINGS.md` §4) — so the current year essentially never
    has usable annual data yet. No live "what's the newest available year"
    endpoint was identified during Phase 0/3 research, so this is a
    documented, conservative heuristic (current year - 1), not a fetched
    fact — logged here rather than silently assumed.
    """
    reference = now or datetime.now(UTC)
    return reference.year - 1


def resolve_year_range(query: TradeQuery, *, now: datetime | None = None) -> tuple[int, int]:
    """Resolve `TradeQuery.year_start`/`year_end` to concrete years per the
    schema's own documented defaults, so every downstream node (both
    `fetch_imports` and `fetch_exports`) agrees on the identical range
    without each re-deriving "latest available" independently.

    `query.years` (2026-08-26 addition) is a relative alternative to
    year_start/year_end — 'the last N years ending at latest available' —
    resolved against the same `_latest_available_year` heuristic as the
    no-args default, so a frontend "years of history" control never has to
    duplicate that heuristic itself to stay in sync. The schema's own
    validator (`TradeQuery._validate_year_range_ordering_and_span`) already
    guarantees `years` and `year_start`/`year_end` are never both set."""
    year_end = query.year_end if query.year_end is not None else _latest_available_year(now=now)
    window = query.years if query.years is not None else YEAR_WINDOW
    year_start = query.year_start if query.year_start is not None else year_end - (window - 1)
    return year_start, year_end


def _error(state: AnalysisState, *, error_code: str, message: str) -> dict[str, Any]:
    return {
        "error": ErrorResponse(
            error_code=error_code,
            message=message,
            retryable=False,
            trace_id=get_or_mint_trace_id(state),
        )
    }


def validate_query(state: AnalysisState) -> dict[str, Any]:
    """Validate the incoming query; writes `query` on success or `error` on
    failure. No model, no I/O beyond the (cached, in-process) taxonomy CSV
    lookup — pure validation + allowlist lookup + defaulting."""
    query = state.get("query")
    if query is None:
        return _error(state, error_code="INVALID_QUERY", message="No query was provided.")

    if not check_hs_code_allowlisted(query.hs_code):
        return _error(
            state,
            error_code="INVALID_HS_CODE",
            message=f"'{query.hs_code}' is not a recognized HS6 code.",
        )

    year_start, year_end = resolve_year_range(query)
    normalized = query.model_copy(update={"year_start": year_start, "year_end": year_end})
    return {"query": normalized}
