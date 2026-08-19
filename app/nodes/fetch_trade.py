"""`fetch_imports` / `fetch_exports` nodes: fan out from `validate_query`
and fan into `aggregate` as a parallel superstep — two independent,
read-only HTTP calls with no data dependency on each other (docs/PLAN.md
§2.2).

Both fetch through the tool-result cache (`app.cache.tool_cache.ToolCache`),
one year at a time across `query`'s resolved year range — matching the
cache's `(hs_code, flow, year)` key granularity and the client's own
verified one-request-per-year behavior (see `comtrade_client.py`'s module
docstring).

Depends on `TradeDataProvider` (a `Protocol`), not `ComtradeClient`
concretely — `get_comtrade_client()` is the only place a concrete provider
is named. Per the project's 2026-08-20 roadmap decision
(`docs/PLAN.md` "Trade-data-source flexibility"): swapping the source later
means writing one new adapter and changing that one call site, not this
module.
"""

from __future__ import annotations

from typing import Any, Literal

from app.cache.tool_cache import ToolCache, get_tool_cache
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.state import AnalysisState, get_or_mint_trace_id, has_error
from app.tools.comtrade_client import (
    ComtradeCircuitOpenError,
    ComtradeClientError,
    ComtradeRecord,
    ComtradeSchemaValidationError,
    ComtradeTimeoutError,
    ComtradeUpstreamError,
    TradeDataProvider,
    get_comtrade_client,
)

# (error_code, retryable) per exception type — `retryable` here means "worth
# the *user* retrying shortly", distinct from the bounded retries the client
# already performed internally. A schema mismatch means our parsing is out
# of sync with a live upstream shape change: retrying the identical request
# will keep failing until code changes, so it is NOT user-retryable, unlike
# a timeout or a temporarily-open circuit breaker.
#
# `ComtradeUpstreamError` is deliberately NOT a key here (finding M9/QA-03):
# unlike every other exception type, it already carries its own
# correctly-computed `.retryable` (a 429/5xx is transient, a 4xx other than
# 429 means *our own request* was malformed and can never succeed on
# retry — see that class's own docstring in comtrade_client.py). A static
# dict entry could only ever encode one fixed value, which would discard
# that distinction — `_map_error` below special-cases it instead of
# putting it in this dict.
_ERROR_MAPPING: dict[type[ComtradeClientError], tuple[str, bool]] = {
    ComtradeTimeoutError: ("UPSTREAM_TIMEOUT", True),
    ComtradeCircuitOpenError: ("UPSTREAM_UNAVAILABLE", True),
    ComtradeSchemaValidationError: ("UPSTREAM_SCHEMA_INVALID", False),
}
_DEFAULT_ERROR_CODE_AND_RETRYABLE = ("UPSTREAM_ERROR", True)


def _map_error(exc: ComtradeClientError) -> tuple[str, bool]:
    # Checked first, ahead of the dict/default fallback (finding M9/QA-03):
    # before this fix, `ComtradeUpstreamError` was never a key in
    # `_ERROR_MAPPING`, so every instance fell through to
    # `_DEFAULT_ERROR_CODE_AND_RETRYABLE = ("UPSTREAM_ERROR", True)` —
    # discarding the exception's own correctly-computed `.retryable`, so a
    # 400/404 (our own malformed request, can never succeed on retry) or a
    # DNS/transport failure was reported to the caller as `retryable=True`
    # anyway, live-verified against the real node function for exactly
    # those cases.
    if isinstance(exc, ComtradeUpstreamError):
        return "UPSTREAM_ERROR", exc.retryable
    for exc_type, mapping in _ERROR_MAPPING.items():
        if isinstance(exc, exc_type):
            return mapping
    return _DEFAULT_ERROR_CODE_AND_RETRYABLE


async def _fetch_flow_cached(
    query: TradeQuery,
    *,
    flow: Literal["import", "export"],
    client: TradeDataProvider,
    cache: ToolCache,
) -> list[ComtradeRecord]:
    """Fetch one flow direction across `query`'s resolved year range,
    year-by-year through the tool-result cache (docs/PLAN.md §5.4 level 3).
    A cache hit for a given (hs_code, flow, year) skips the network call
    entirely; a miss fetches just that year and populates the cache with
    the finalized/provisional TTL split (`app/cache/tool_cache.py`)."""
    if query.year_start is None or query.year_end is None:
        # validate_query always resolves these before any fetch node runs;
        # defensive, not expected to trigger outside a test calling this
        # helper directly with an un-normalized query.
        raise ValueError("query.year_start/year_end must be resolved before fetching")

    all_records: list[ComtradeRecord] = []
    for year in range(query.year_start, query.year_end + 1):
        cached = await cache.get(hs_code=query.hs_code, flow=flow, year=year)
        if cached is not None:
            all_records.extend(cached)
            continue
        year_records = await client.fetch_flow(
            hs_code=query.hs_code, flow=flow, year_start=year, year_end=year
        )
        # A year is "finalized" only if every retained record for it was
        # genuinely reported (not modeled/estimated) — conservative by
        # design (docs/PLAN.md §11: a provisional year may still revise).
        is_finalized = bool(year_records) and all(not r.is_provisional for r in year_records)
        await cache.set(
            hs_code=query.hs_code,
            flow=flow,
            year=year,
            records=year_records,
            is_finalized=is_finalized,
        )
        all_records.extend(year_records)
    return all_records


async def _fetch_flow_node(
    state: AnalysisState, *, flow: Literal["import", "export"], state_key: str
) -> dict[str, Any]:
    if has_error(state):
        return {}  # validate_query (or a sibling fetch node) already failed
    query = state.get("query")
    if query is None:
        return {}  # defensive: validate_query should always set query or error

    try:
        records = await _fetch_flow_cached(
            query, flow=flow, client=get_comtrade_client(), cache=get_tool_cache()
        )
    except ComtradeClientError as exc:
        error_code, retryable = _map_error(exc)
        return {
            "error": ErrorResponse(
                error_code=error_code,
                message=(
                    f"Could not retrieve {flow} data from UN Comtrade right now. "
                    "Please try again shortly."
                ),
                retryable=retryable,
                trace_id=get_or_mint_trace_id(state),
            )
        }
    return {state_key: records}


async def fetch_imports(state: AnalysisState) -> dict[str, Any]:
    """Fetch import-flow records for `state["query"].hs_code`; writes
    `raw_imports`."""
    return await _fetch_flow_node(state, flow="import", state_key="raw_imports")


async def fetch_exports(state: AnalysisState) -> dict[str, Any]:
    """Fetch export-flow records for `state["query"].hs_code`; writes
    `raw_exports`."""
    return await _fetch_flow_node(state, flow="export", state_key="raw_exports")
