"""Integration tests for the assembled, compiled v1 graph
(`app.graph.build_graph` + `app.graph.build_checkpointer`): full invocation
end-to-end under `LLM_PROVIDER=mock` against a mocked Comtrade transport and
a real (in-memory) checkpointer — docs/PLAN.md §7's "Full graph invoke with
`LLM_PROVIDER=mock`, asserting envelope shape end-to-end" row. No live
network, no live model call.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

import httpx
import pytest

import app.nodes.fetch_trade as fetch_trade_module
from app.cache.tool_cache import ToolCache
from app.graph import build_checkpointer, build_graph
from app.schemas.query import TradeQuery
from app.state import AnalysisState
from app.tools.comtrade_client import ComtradeClient

_EXPECTED_NODES = {
    "validate_query",
    "fetch_imports",
    "fetch_exports",
    "aggregate",
    "retrieve_description",
    "describe_item",
    "summarize",
    "assemble_response",
}


def _record(*, partner_code: int, period: str, value: float, reported: bool = True) -> dict:
    return {
        "reporterCode": 699,
        "partnerCode": partner_code,
        "period": period,
        "cmdCode": "010121",
        "flowCode": "M",
        "primaryValue": value,
        "isReported": reported,
        "isAggregate": partner_code == 0,
    }


def _handler_with_data(request: httpx.Request) -> httpx.Response:
    period = request.url.params["period"]
    return httpx.Response(
        200,
        json={
            "count": 2,
            "error": "",
            "data": [
                _record(partner_code=0, period=period, value=5000.0),  # World, stripped
                _record(partner_code=842, period=period, value=1000.0 + int(period)),
            ],
        },
    )


def _handler_empty(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"count": 0, "error": "", "data": []})


def _patch_comtrade(
    monkeypatch: pytest.MonkeyPatch, *, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    client = ComtradeClient(api_key="test-key", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: ToolCache())


async def _run(query: TradeQuery, *, thread_id: str | None = None) -> tuple[AnalysisState, str]:
    thread_id = thread_id or str(uuid.uuid4())
    graph = build_graph()
    async with build_checkpointer("sqlite+aiosqlite:///:memory:") as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
        initial_state: AnalysisState = {
            "trace_id": "trace-1",
            "thread_id": thread_id,
            "message_id": str(uuid.uuid4()),
            "query": query,
        }
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
        final_state = await compiled.ainvoke(initial_state, config=config)
    return final_state, thread_id


@pytest.mark.integration
def test_build_graph_has_exactly_the_planned_nodes() -> None:
    graph = build_graph()
    assert set(graph.nodes.keys()) == _EXPECTED_NODES


@pytest.mark.integration
async def test_full_invocation_happy_path_produces_schema_valid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_comtrade(monkeypatch, handler=_handler_with_data)
    query = TradeQuery(hs_code="010121", year_start=2021, year_end=2022)

    final_state, thread_id = await _run(query)

    assert "error" not in final_state
    response = final_state["response"]
    assert response.thread_id == thread_id
    assert response.hs_code == "010121"
    assert len(response.item_description) > 0
    assert len(response.analytical_summary) > 0
    # The "World" aggregate row (partner_code 0) must never reach the table.
    assert all(row.partner_code != "0" for row in response.imports.rows)
    assert response.imports.excluded_partner_codes == ["0"]
    assert response.imports.years == [2021, 2022]
    assert response.provenance.currency == "USD"


@pytest.mark.integration
async def test_full_invocation_is_resumable_via_aget_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_comtrade(monkeypatch, handler=_handler_with_data)
    query = TradeQuery(hs_code="010121", year_start=2022, year_end=2022)
    thread_id = str(uuid.uuid4())

    graph = build_graph()
    async with build_checkpointer("sqlite+aiosqlite:///:memory:") as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
        initial_state: AnalysisState = {
            "trace_id": "trace-1",
            "thread_id": thread_id,
            "message_id": str(uuid.uuid4()),
            "query": query,
        }
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
        invoked = await compiled.ainvoke(initial_state, config=config)

        snapshot = await compiled.aget_state({"configurable": {"thread_id": thread_id}})

        # A thread that has never run at all returns an empty snapshot
        # (verified live against the installed langgraph — see
        # app/graph.py's module docstring) -- used by GET /threads/{id} to
        # distinguish "unknown thread" from "thread has a completed run".
        unknown_snapshot = await compiled.aget_state(
            {"configurable": {"thread_id": str(uuid.uuid4())}}
        )

    assert snapshot.values["response"] == invoked["response"]
    assert unknown_snapshot.values == {}


@pytest.mark.integration
async def test_full_invocation_invalid_hs_code_short_circuits_before_any_fetch_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _tracking_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["period"])
        return _handler_with_data(request)

    _patch_comtrade(monkeypatch, handler=_tracking_handler)
    query = TradeQuery(hs_code="000000")  # shape-valid, not a real taxonomy entry

    final_state, _ = await _run(query)

    assert "response" not in final_state
    error = final_state["error"]
    assert error.error_code == "INVALID_HS_CODE"
    assert calls == []  # fetch_imports/fetch_exports never made a network call
    assert "item_description" not in final_state
    assert "analytical_summary" not in final_state


@pytest.mark.integration
async def test_full_invocation_upstream_timeout_degrades_gracefully_into_fetch_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-08-20 roadmap decision (live user-reported finding): a Comtrade
    fetch failure no longer fails the whole request — the graph still
    reaches `assemble_response` with a real `response`, and the failed
    year is surfaced as `imports_table.fetch_issues`, excluded from
    `years_no_data` (a fetch failure is not the same claim as "Comtrade
    genuinely had nothing" — see `app.nodes.aggregate.flag_years_no_data`)."""

    def _timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated", request=request)

    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(_timeout_handler), max_attempts=1
    )
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: ToolCache())

    query = TradeQuery(hs_code="010121", year_start=2022, year_end=2022)
    final_state, _ = await _run(query)

    assert "error" not in final_state
    response = final_state["response"]
    assert len(response.imports.fetch_issues) == 1
    assert "2022" in response.imports.fetch_issues[0]
    assert 2022 not in response.imports.years_no_data


@pytest.mark.integration
async def test_full_invocation_both_fetch_nodes_failing_concurrently_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`fetch_imports` and `fetch_exports` run as a genuine parallel
    superstep (docs/PLAN.md §2.2). Historically (before the 2026-08-20
    graceful-degradation change) a systemic upstream outage failed BOTH at
    once and both attempted to write the shared `error` key in the same
    superstep, risking an uncaught `InvalidUpdateError` from two concurrent
    writes to one state key (`app/state.py`'s `_keep_first_error` reducer
    exists because of that exact crash). Fetch failures no longer write
    `error` at all, so that specific collision is structurally impossible
    now — this instead regression-covers that the *new* code path (each
    flow writing to its own distinct `*_fetch_issues` key concurrently)
    is equally crash-free and still produces one well-formed, successful
    response with both tables' issues populated."""

    def _always_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated", request=request)

    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(_always_timeout), max_attempts=1
    )
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: ToolCache())

    query = TradeQuery(hs_code="010121", year_start=2022, year_end=2022)
    final_state, _ = await _run(query)  # must not raise

    assert "error" not in final_state
    response = final_state["response"]
    assert len(response.imports.fetch_issues) == 1
    assert len(response.exports.fetch_issues) == 1


@pytest.mark.integration
async def test_full_invocation_empty_upstream_data_renders_missing_not_fabricated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_comtrade(monkeypatch, handler=_handler_empty)
    query = TradeQuery(hs_code="010121", year_start=2021, year_end=2022)

    final_state, _ = await _run(query)

    assert "error" not in final_state
    response = final_state["response"]
    assert response.imports.rows == []  # no partners at all -- not padded/fabricated
    assert response.exports.rows == []
    assert response.imports.years_finalized == []  # a year with zero records isn't "finalized"
