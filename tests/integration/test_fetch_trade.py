"""Tests for `fetch_imports`/`fetch_exports` — the tool-result cache layer
and error mapping, against `httpx.MockTransport` (no live network,
docs/PLAN.md §7). `get_comtrade_client`/`get_tool_cache` are monkeypatched
at their point of use in `app.nodes.fetch_trade` so each test gets an
isolated client/cache rather than sharing the process-wide singletons.
"""

from __future__ import annotations

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
    assert sorted(handler.calls) == ["2021", "2022", "2023"]


@pytest.mark.integration
async def test_fetch_exports_populates_raw_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _CountingHandler()
    _patch_client_and_cache(monkeypatch, handler=handler)
    query = TradeQuery(hs_code="010121", year_start=2022, year_end=2022)
    state: AnalysisState = {"query": query}

    result = await fetch_exports(state)

    assert result["raw_exports"] == []
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
async def test_fetch_imports_maps_timeout_to_upstream_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated", request=request)

    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(timeout_handler), max_attempts=1
    )
    cache = ToolCache()
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: cache)

    query = TradeQuery(hs_code="010121", year_start=2022, year_end=2022)
    state: AnalysisState = {"query": query, "trace_id": "t-42"}

    result = await fetch_imports(state)

    assert "raw_imports" not in result
    error = result["error"]
    assert error.error_code == "UPSTREAM_TIMEOUT"
    assert error.retryable is True
    assert error.trace_id == "t-42"
