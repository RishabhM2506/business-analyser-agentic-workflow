"""Tests for `fetch_imports`/`fetch_exports` — the tool-result cache layer
and per-year graceful degradation, against `httpx.MockTransport` (no live
network, docs/PLAN.md §7). `get_comtrade_client`/`get_tool_cache` are
monkeypatched at their point of use in `app.nodes.fetch_trade` so each test
gets an isolated client/cache rather than sharing the process-wide
singletons.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

import app.nodes.fetch_trade as fetch_trade_module
from app.cache.tool_cache import ToolCache
from app.nodes.fetch_trade import fetch_exports, fetch_imports
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.state import AnalysisState
from app.tools.comtrade_client import ComtradeClient


class _CountingHandler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request.url.params["period"])
        return httpx.Response(200, json={"count": 0, "data": [], "error": ""})


def _patch_client_and_cache(
    monkeypatch: pytest.MonkeyPatch, *, handler: _CountingHandler
) -> ToolCache:
    client = ComtradeClient(api_key="test-key", transport=httpx.MockTransport(handler))
    cache = ToolCache()
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: cache)
    return cache


@pytest.mark.integration
async def test_fetch_imports_populates_raw_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _CountingHandler()
    _patch_client_and_cache(monkeypatch, handler=handler)
    query = TradeQuery(hs_code="010121", year_start=2021, year_end=2023)
    state: AnalysisState = {"query": query}

    result = await fetch_imports(state)

    assert result["raw_imports"] == []
    assert result["import_fetch_issues"] == []
    assert sorted(handler.calls) == ["2021", "2022", "2023"]


@pytest.mark.integration
async def test_fetch_exports_populates_raw_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _CountingHandler()
    _patch_client_and_cache(monkeypatch, handler=handler)
    query = TradeQuery(hs_code="010121", year_start=2022, year_end=2022)
    state: AnalysisState = {"query": query}

    result = await fetch_exports(state)

    assert result["raw_exports"] == []
    assert result["export_fetch_issues"] == []
    assert handler.calls == ["2022"]


@pytest.mark.integration
async def test_fetch_imports_reuses_cache_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _CountingHandler()
    _patch_client_and_cache(monkeypatch, handler=handler)
    query = TradeQuery(hs_code="010121", year_start=2022, year_end=2022)
    state: AnalysisState = {"query": query}

    await fetch_imports(state)
    await fetch_imports(state)  # same (hs_code, flow, year) - should be a cache hit

    assert handler.calls == ["2022"]  # only one real network call, ever


@pytest.mark.integration
async def test_fetch_imports_short_circuits_on_existing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _CountingHandler()
    _patch_client_and_cache(monkeypatch, handler=handler)
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121", year_start=2022, year_end=2022),
        "error": ErrorResponse(
            error_code="INVALID_HS_CODE", message="x", retryable=False, trace_id="t-1"
        ),
    }

    result = await fetch_imports(state)

    assert result == {}
    assert handler.calls == []  # never even tried


@pytest.mark.integration
async def test_fetch_imports_missing_query_is_defensive_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _CountingHandler()
    _patch_client_and_cache(monkeypatch, handler=handler)
    result = await fetch_imports({})
    assert result == {}
    assert handler.calls == []


@pytest.mark.integration
async def test_fetch_imports_one_failing_year_degrades_gracefully_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-08-20 roadmap decision (live user-reported finding): a single
    (year, flow) exhausting its retries must not void the whole request —
    the failing year is recorded as a `FetchIssue` and the *other* years
    still succeed, with no `error` key at all. Regression coverage for
    exactly the real shape observed live: 2022 failed while 2021/2023
    succeeded in the same request."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["period"] == "2022":
            raise httpx.ReadTimeout("simulated", request=request)
        return httpx.Response(200, json={"count": 0, "data": [], "error": ""})

    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(handler), max_attempts=1
    )
    cache = ToolCache()
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: cache)

    query = TradeQuery(hs_code="010121", year_start=2021, year_end=2023)
    state: AnalysisState = {"query": query, "trace_id": "t-42"}

    result = await fetch_imports(state)

    assert "error" not in result
    assert result["raw_imports"] == []  # the two successful years had no records either
    issues = result["import_fetch_issues"]
    assert len(issues) == 1
    assert issues[0].year == 2022
    assert "timed out" in issues[0].reason or "timeout" in issues[0].reason.lower()


@pytest.mark.integration
async def test_fetch_imports_every_year_failing_still_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a total per-flow outage (every year fails) is not a hard
    request failure anymore — `raw_imports` is empty and every year is
    listed as an issue, but the request itself still succeeds; the caller
    (`aggregate`/`assemble_response`) is responsible for rendering an
    honestly-empty table plus these notes, not this node."""

    def handler_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(handler_429), max_attempts=1
    )
    cache = ToolCache()
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: cache)

    query = TradeQuery(hs_code="010121", year_start=2021, year_end=2022)
    state: AnalysisState = {"query": query, "trace_id": "t-all-fail"}

    result = await fetch_imports(state)

    assert "error" not in result
    assert result["raw_imports"] == []
    issues = result["import_fetch_issues"]
    assert sorted(issue.year for issue in issues) == [2021, 2022]
    assert all("429" in issue.reason for issue in issues)


def _handler_client_error_400(request: httpx.Request) -> httpx.Response:
    return httpx.Response(400, json={"error": "bad request"})


def _handler_transport_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("simulated connection failure", request=request)


def _handler_server_error_500(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "internal"})


@pytest.mark.integration
@pytest.mark.parametrize(
    "handler",
    [_handler_client_error_400, _handler_transport_error, _handler_server_error_500],
    ids=["client_error_400", "transport_error", "server_error_500"],
)
async def test_fetch_imports_records_the_real_exception_message_as_the_reason(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """`FetchIssue.reason` is the real caught exception's own `str(exc)`,
    never a paraphrase — checked across a few different real failure
    shapes (a non-retryable 4xx, a transport-level error, a retryable
    5xx), not just the timeout/429 cases the other tests already cover."""
    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(handler), max_attempts=1
    )
    cache = ToolCache()
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: cache)

    query = TradeQuery(hs_code="010121", year_start=2022, year_end=2022)
    state: AnalysisState = {"query": query, "trace_id": "t-reason"}

    result = await fetch_imports(state)

    assert "error" not in result
    issues = result["import_fetch_issues"]
    assert len(issues) == 1
    assert issues[0].year == 2022
    assert len(issues[0].reason) > 0
    assert "UN Comtrade" in issues[0].reason  # every real exception message starts this way
