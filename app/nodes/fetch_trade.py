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

**Per-year graceful degradation (2026-08-20, live user-reported finding)**:
a single (year, flow) call exhausting its retries (`ComtradeClientError`)
used to fail the *entire* request with an `ErrorResponse`, discarding
whatever other years already succeeded in the same loop. Real UN Comtrade
rate-limiting can hit one specific call in an otherwise-successful batch —
live-reproduced: 2022 imports got three consecutive 429s while every other
(year, flow) in the same request recovered on retry. A single flaky year
no longer voids the whole analysis: `_fetch_flow_cached` now catches
per-year, keeps fetching the remaining years, and returns the failures
alongside whatever records it did get. `aggregate.py` folds these into
`TradeTable.fetch_issues` — real, honest per-year notes in the final
response itself, never routed through the LLM (the same "structured data,
not model prose" discipline as `years_no_data`/`years_finalized`) — instead
of an opaque request-level failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.cache.tool_cache import ToolCache, get_tool_cache
from app.schemas.query import TradeQuery
from app.state import AnalysisState, FetchIssue, has_error
from app.tools.comtrade_client import (
    ComtradeClientError,
    ComtradeRecord,
    TradeDataProvider,
    get_comtrade_client,
)


@dataclass(frozen=True)
class _FlowFetchResult:
    records: list[ComtradeRecord]
    issues: list[FetchIssue]


async def _fetch_flow_cached(
    query: TradeQuery,
    *,
    flow: Literal["import", "export"],
    client: TradeDataProvider,
    cache: ToolCache,
) -> _FlowFetchResult:
    """Fetch one flow direction across `query`'s resolved year range,
    year-by-year through the tool-result cache (docs/PLAN.md §5.4 level 3).
    A cache hit for a given (hs_code, flow, year) skips the network call
    entirely; a miss fetches just that year and populates the cache with
    the finalized/provisional TTL split (`app/cache/tool_cache.py`).

    A year whose fetch ultimately fails (retries exhausted) is recorded as
    a `FetchIssue` and skipped, not raised — the loop always continues
    through every year regardless (module docstring: no single flaky year
    voids the rest of an otherwise-successful fetch)."""
    if query.year_start is None or query.year_end is None:
        # validate_query always resolves these before any fetch node runs;
        # defensive, not expected to trigger outside a test calling this
        # helper directly with an un-normalized query.
        raise ValueError("query.year_start/year_end must be resolved before fetching")

    all_records: list[ComtradeRecord] = []
    issues: list[FetchIssue] = []
    for year in range(query.year_start, query.year_end + 1):
        cached = await cache.get(hs_code=query.hs_code, flow=flow, year=year)
        if cached is not None:
            all_records.extend(cached)
            continue
        try:
            year_records = await client.fetch_flow(
                hs_code=query.hs_code, flow=flow, year_start=year, year_end=year
            )
        except ComtradeClientError as exc:
            issues.append(FetchIssue(year=year, reason=str(exc)))
            continue
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
    return _FlowFetchResult(records=all_records, issues=issues)


async def _fetch_flow_node(
    state: AnalysisState, *, flow: Literal["import", "export"], records_key: str, issues_key: str
) -> dict[str, Any]:
    if has_error(state):
        return {}  # validate_query (or a sibling fetch node) already failed
    query = state.get("query")
    if query is None:
        return {}  # defensive: validate_query should always set query or error

    result = await _fetch_flow_cached(
        query, flow=flow, client=get_comtrade_client(), cache=get_tool_cache()
    )
    return {records_key: result.records, issues_key: result.issues}


async def fetch_imports(state: AnalysisState) -> dict[str, Any]:
    """Fetch import-flow records for `state["query"].hs_code`; writes
    `raw_imports` and `import_fetch_issues`."""
    return await _fetch_flow_node(
        state, flow="import", records_key="raw_imports", issues_key="import_fetch_issues"
    )


async def fetch_exports(state: AnalysisState) -> dict[str, Any]:
    """Fetch export-flow records for `state["query"].hs_code`; writes
    `raw_exports` and `export_fetch_issues`."""
    return await _fetch_flow_node(
        state, flow="export", records_key="raw_exports", issues_key="export_fetch_issues"
    )
