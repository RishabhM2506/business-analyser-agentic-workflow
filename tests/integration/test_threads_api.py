"""Integration tests for the thread/message API (docs/PLAN.md §3.3):
`POST /threads`, `GET /threads/{id}`, `POST /threads/{id}/messages`. Full
HTTP stack (FastAPI + ASGI transport), `LLM_PROVIDER=mock`, a mocked
Comtrade transport, and a real (in-memory, per-test) checkpointer — no live
network, no live model call.

`httpx.ASGITransport` does not run the ASGI `lifespan` protocol (verified
directly against the installed `httpx==0.28.1` — `ASGITransport.
handle_async_request` only ever builds a `"type": "http"` scope), so every
test here explicitly drives `app.router.lifespan_context(app)` itself
(Starlette's own public mechanism — the same one `TestClient` uses
internally) to actually run `app/main.py`'s `lifespan`, which is what opens
the checkpointer and compiles the graph onto `app.state.compiled_graph`.
Without this, every thread endpoint would 500 with an `AttributeError`.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

import app.nodes.fetch_trade as fetch_trade_module
from app.cache.tool_cache import ToolCache
from app.main import REQUEST_ID_HEADER, create_app
from app.settings import Settings
from app.tools.comtrade_client import ComtradeClient


def _handler_with_data(request: httpx.Request) -> httpx.Response:
    period = request.url.params["period"]
    return httpx.Response(
        200,
        json={
            "count": 1,
            "error": "",
            "data": [
                {
                    "reporterCode": 699,
                    "partnerCode": 842,
                    "period": period,
                    "cmdCode": "010121",
                    "flowCode": request.url.params["flowCode"],
                    "primaryValue": 1000.0 + int(period),
                    "isReported": True,
                    "isAggregate": False,
                }
            ],
        },
    )


def _patch_comtrade(
    monkeypatch: pytest.MonkeyPatch,
    *,
    handler: Callable[[httpx.Request], httpx.Response] = _handler_with_data,
) -> None:
    client = ComtradeClient(api_key="test-key", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: ToolCache())


def _isolated_settings(**overrides: object) -> Settings:
    # A fresh in-memory sqlite checkpointer per test (never shared across
    # tests, unlike a real file path) so thread/response state can never
    # leak between them.
    return Settings.model_validate({"database_url": "sqlite+aiosqlite:///:memory:", **overrides})


def _data(response: httpx.Response) -> dict[str, object]:
    """Unwrap the `{"type": "final", "data": ...}` response envelope
    (docs/PLAN.md §3.3, finding B1/ARCH-01) that every `TradeAnalysisResponse`
    /`ErrorResponse` is now sent inside — asserts the discriminator while
    it's at it, so every call site gets that check for free instead of each
    test re-asserting it individually."""
    body = response.json()
    assert body["type"] == "final"
    return body["data"]  # type: ignore[no-any-return]


@asynccontextmanager
async def _client_for(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):  # actually runs app/main.py's lifespan
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.mark.integration
async def test_post_threads_returns_a_fresh_thread_id() -> None:
    async with _client_for(_isolated_settings()) as client:
        response = await client.post("/threads")

    assert response.status_code == 201
    body = response.json()
    assert len(body["thread_id"]) == 36  # UUID4 string length
    assert REQUEST_ID_HEADER in response.headers


@pytest.mark.integration
async def test_post_threads_returns_distinct_ids_across_calls() -> None:
    async with _client_for(_isolated_settings()) as client:
        first = (await client.post("/threads")).json()["thread_id"]
        second = (await client.post("/threads")).json()["thread_id"]

    assert first != second


@pytest.mark.integration
async def test_post_message_happy_path_returns_schema_valid_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_comtrade(monkeypatch)
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(
            f"/threads/{thread_id}/messages",
            json={"hs_code": "010121", "year_start": 2021, "year_end": 2022},
        )

    assert response.status_code == 200
    body = _data(response)
    assert body["thread_id"] == thread_id
    assert body["hs_code"] == "010121"
    assert len(body["message_id"]) > 0
    assert len(body["item_description"]) > 0
    assert len(body["analytical_summary"]) > 0
    assert body["imports"]["unit"] == "USD"
    assert body["provenance"]["source"] == "UN Comtrade (comtradeapi.un.org)"


@pytest.mark.integration
async def test_post_message_unknown_hs_code_returns_400_before_touching_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _tracking_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["period"])
        return _handler_with_data(request)

    _patch_comtrade(monkeypatch, handler=_tracking_handler)
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(f"/threads/{thread_id}/messages", json={"hs_code": "000000"})

    assert response.status_code == 400
    body = _data(response)
    assert body["error_code"] == "INVALID_HS_CODE"
    assert body["retryable"] is False
    assert len(body["trace_id"]) > 0
    assert calls == []  # rejected before the graph (and any fetch) ever ran


@pytest.mark.integration
async def test_post_message_shape_invalid_hs_code_returns_error_response_not_fastapi_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body that fails `TradeQuery`'s own Pydantic field validation (not
    a 6-digit string at all) must still come back as our `ErrorResponse`
    schema via `handle_validation_error`, never FastAPI's default
    `{"detail": [...]}` shape (docs/PLAN.md §3.2)."""
    _patch_comtrade(monkeypatch)
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(f"/threads/{thread_id}/messages", json={"hs_code": "abc"})

    assert response.status_code == 400
    body = _data(response)
    assert body["error_code"] == "INVALID_QUERY"
    assert "detail" not in body


@pytest.mark.integration
async def test_post_message_extra_field_rejected_as_invalid_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_comtrade(monkeypatch)
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(
            f"/threads/{thread_id}/messages",
            json={"hs_code": "010121", "not_a_real_field": "x"},
        )

    assert response.status_code == 400
    assert _data(response)["error_code"] == "INVALID_QUERY"


@pytest.mark.integration
async def test_post_message_upstream_timeout_maps_to_504(monkeypatch: pytest.MonkeyPatch) -> None:
    def _timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated", request=request)

    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(_timeout_handler), max_attempts=1
    )
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: ToolCache())
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client_app:
        response = await client_app.post(
            f"/threads/{thread_id}/messages", json={"hs_code": "010121"}
        )

    assert response.status_code == 504
    assert _data(response)["error_code"] == "UPSTREAM_TIMEOUT"


@pytest.mark.integration
async def test_get_thread_unknown_id_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_comtrade(monkeypatch)
    async with _client_for(_isolated_settings()) as client:
        response = await client.get(f"/threads/{uuid.uuid4()}")

    assert response.status_code == 404
    assert _data(response)["error_code"] == "THREAD_NOT_FOUND"


@pytest.mark.integration
async def test_get_thread_after_message_resumes_the_same_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_comtrade(monkeypatch)
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        posted = await client.post(f"/threads/{thread_id}/messages", json={"hs_code": "010121"})
        fetched = await client.get(f"/threads/{thread_id}")

    assert posted.status_code == 200
    assert fetched.status_code == 200
    assert fetched.json() == posted.json()


@pytest.mark.integration
async def test_get_thread_after_error_resumes_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uses an *in-graph* failure (upstream timeout), not the
    `INVALID_HS_CODE` pre-check: that check deliberately rejects a request
    before the graph is ever invoked (see `app/main.py`'s module docstring
    on guardrail ordering), so it never writes any checkpoint for that
    `thread_id` at all — `GET /threads/{id}` correctly reports
    `THREAD_NOT_FOUND` for it (covered by
    `test_post_message_unknown_hs_code_returns_400_before_touching_upstream`
    above), not a resumable error. A failure that happens *inside* the
    graph, by contrast, does write real checkpoint state before returning
    — that's what this test proves is resumable.
    """

    def _timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated", request=request)

    client_stub = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(_timeout_handler), max_attempts=1
    )
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client_stub)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: ToolCache())
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        posted = await client.post(f"/threads/{thread_id}/messages", json={"hs_code": "010121"})
        fetched = await client.get(f"/threads/{thread_id}")

    assert posted.status_code == 504
    assert fetched.status_code == 504
    assert _data(fetched)["error_code"] == "UPSTREAM_TIMEOUT"


@pytest.mark.integration
async def test_post_message_two_successful_analyses_on_same_thread_both_succeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARCH-02/B2: a thread models one user's *session* (docs/PLAN.md §2.2 —
    created once on "Start my process," reused for every item picked
    afterward), not one single analysis. Two different, successful
    `hs_code` analyses on the same thread must both succeed — before the
    budget-ceiling fix, the *first* analysis alone exactly exhausted the
    default `max_model_calls_per_thread`, so a completely ordinary second
    item lookup in the same session always failed with `BUDGET_EXCEEDED`.
    This scenario had zero test coverage before this fix (confirmed:
    no existing test in this file posts a second message to an
    already-successful thread)."""
    _patch_comtrade(monkeypatch)
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        first = await client.post(f"/threads/{thread_id}/messages", json={"hs_code": "010121"})
        second = await client.post(f"/threads/{thread_id}/messages", json={"hs_code": "160100"})

    assert first.status_code == 200
    first_body = _data(first)
    assert first_body["hs_code"] == "010121"

    assert second.status_code == 200
    second_body = _data(second)
    assert second_body["hs_code"] == "160100"
    assert second_body["message_id"] != first_body["message_id"]


@pytest.mark.integration
async def test_post_message_concurrent_requests_on_same_thread_do_not_corrupt_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QA-01/B4: nothing previously serialized two `compiled_graph.ainvoke()`
    calls on the same `thread_id` — two genuinely concurrent, successful
    analyses on one thread raced for which run's checkpoint became "the"
    resumable state, and QA-01 reproduced a case with zero errors involved
    where one of the two was simply, arbitrarily lost. This test fires two
    different, valid `hs_code` requests concurrently at the same thread and
    asserts: both individually succeed with their own correct, internally
    consistent result (proving the per-request response is never affected
    by the race), and afterward `GET /threads/{id}` returns one complete,
    uncorrupted result matching exactly one of the two requests (proving no
    torn/interleaved state), not a 500 or a mixed/inconsistent body."""
    _patch_comtrade(monkeypatch)
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        first, second = await asyncio.gather(
            client.post(f"/threads/{thread_id}/messages", json={"hs_code": "010121"}),
            client.post(f"/threads/{thread_id}/messages", json={"hs_code": "160100"}),
        )
        fetched = await client.get(f"/threads/{thread_id}")

    # Each request's own response is unaffected by the other running
    # concurrently on the same thread — both fully succeed, each internally
    # consistent (its own hs_code matches its own message_id's result).
    assert first.status_code == 200
    assert second.status_code == 200
    first_body = _data(first)
    second_body = _data(second)
    assert first_body["hs_code"] == "010121"
    assert second_body["hs_code"] == "160100"
    assert first_body["message_id"] != second_body["message_id"]

    # GET afterward is a well-defined outcome: one complete, valid response
    # that matches exactly one of the two requests in full (never a body
    # that mixes fields from both, and never a corrupted/incomplete state).
    assert fetched.status_code == 200
    fetched_body = _data(fetched)
    assert fetched_body in (first_body, second_body)


@pytest.mark.integration
async def test_get_thread_after_sequential_messages_reflects_the_latest_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QA-01/B4, the sequential half: with writes now serialized by
    `_ThreadLockRegistry`, a thread holding multiple analyses over its life
    (finding B2) has well-defined, deterministic `GET` semantics — the most
    recently *completed* message, not an arbitrary pick. Purely sequential
    (no `asyncio.gather`), so this also holds even without genuine
    scheduler interleaving."""
    _patch_comtrade(monkeypatch)
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        first = await client.post(f"/threads/{thread_id}/messages", json={"hs_code": "010121"})
        second = await client.post(f"/threads/{thread_id}/messages", json={"hs_code": "160100"})
        fetched = await client.get(f"/threads/{thread_id}")

    assert first.status_code == 200
    assert second.status_code == 200
    assert fetched.status_code == 200
    assert _data(fetched) == _data(second)


@pytest.mark.integration
async def test_post_message_after_earlier_failure_on_same_thread_is_not_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PBO-01/QA-01, finding B3: a thread that failed once must not stay
    permanently poisoned. Live-reproduced by PBO-01: after one message on a
    thread failed, posting a second, *different* `hs_code` to that same
    thread came back with the identical error, identical `trace_id`, in
    27ms — proof no real work happened, just a replay of the stale
    `AnalysisState.error` left behind by the first call. This test posts a
    failing message, then a different, succeeding one, on the same thread,
    and asserts the second genuinely ran (a real, matching
    `TradeAnalysisResponse`) rather than reflecting the first call's error.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["cmdCode"] == "010121":
            raise httpx.ReadTimeout("simulated upstream failure", request=request)
        return _handler_with_data(request)

    client_stub = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(_handler), max_attempts=1
    )
    monkeypatch.setattr(fetch_trade_module, "get_comtrade_client", lambda: client_stub)
    monkeypatch.setattr(fetch_trade_module, "get_tool_cache", lambda: ToolCache())
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        failed = await client.post(f"/threads/{thread_id}/messages", json={"hs_code": "010121"})
        succeeded = await client.post(f"/threads/{thread_id}/messages", json={"hs_code": "160100"})
        fetched = await client.get(f"/threads/{thread_id}")

    assert failed.status_code == 504
    assert _data(failed)["error_code"] == "UPSTREAM_TIMEOUT"

    # The critical assertion: the second, different hs_code actually ran (a
    # real TradeAnalysisResponse for 160100) instead of replaying the first
    # request's stale, unrelated error.
    assert succeeded.status_code == 200
    succeeded_body = _data(succeeded)
    assert succeeded_body["hs_code"] == "160100"
    assert "error_code" not in succeeded_body

    # The thread's resumable state reflects the latest completed message,
    # not the earlier failure.
    assert fetched.status_code == 200
    assert _data(fetched)["hs_code"] == "160100"


@pytest.mark.integration
async def test_post_message_budget_exceeded_maps_to_429(monkeypatch: pytest.MonkeyPatch) -> None:
    # `app.budget.get_budget_tracker` is a module-level singleton keyed off
    # the *global* `get_settings()`, like every other singleton factory in
    # this codebase (`get_comtrade_client`, `get_tool_cache`) — it does not
    # read `create_app(settings=...)`'s per-test override. Same pattern
    # `tests/unit/test_models.py` already uses for the same reason: force a
    # cache-clear + env override, then restore both.
    import app.budget as budget_module
    from app.settings import get_settings

    monkeypatch.setenv("MAX_MODEL_CALLS_PER_THREAD", "0")
    get_settings.cache_clear()
    monkeypatch.setattr(budget_module, "_budget_tracker_singleton", None)
    try:
        _patch_comtrade(monkeypatch)
        thread_id = str(uuid.uuid4())

        async with _client_for(_isolated_settings()) as client:
            response = await client.post(
                f"/threads/{thread_id}/messages", json={"hs_code": "010121"}
            )

        assert response.status_code == 429
        assert _data(response)["error_code"] == "BUDGET_EXCEEDED"
    finally:
        get_settings.cache_clear()
        monkeypatch.setattr(budget_module, "_budget_tracker_singleton", None)
