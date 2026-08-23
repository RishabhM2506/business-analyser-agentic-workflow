"""Unit tests for `app.pipeline.data_gov_in` — the shared data.gov.in
resource-API mechanics (retry on both real rate-limit shapes, pagination,
filter-param construction) extracted from `app.pipeline.agmarknet` once a
second dataset needed the identical behavior. Uses `httpx.MockTransport`
(this repo's established pattern), never a live call.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from app.pipeline.data_gov_in import (
    DataGovInError,
    DataGovInRateLimitedError,
    fetch_all_pages,
    fetch_page,
)

pytestmark = pytest.mark.unit


async def test_fetch_page_returns_records_on_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"total": 1, "count": 1, "records": [{"a": 1}]})

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    page = await fetch_page(client, resource_path="/resource/x", api_key="key", offset=0, limit=10)

    assert page == [{"a": 1}]


async def test_fetch_page_sends_the_real_required_user_agent() -> None:
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(200, json={"records": []})

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    await fetch_page(client, resource_path="/resource/x", api_key="key", offset=0, limit=10)

    assert seen_headers["user-agent"] == "business-analyser-agentic-workflow/1.0"


async def test_fetch_page_builds_filters_params() -> None:
    seen_params: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.update(request.url.params)
        return httpx.Response(200, json={"records": []})

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    await fetch_page(
        client,
        resource_path="/resource/x",
        api_key="key",
        offset=0,
        limit=10,
        filters={"Commodity": "Wheat", "State": "Punjab"},
    )

    assert seen_params["filters[Commodity]"] == "Wheat"
    assert seen_params["filters[State]"] == "Punjab"


async def test_fetch_page_retries_on_the_http_200_error_body_shape() -> None:
    """Real shape found building app.pipeline.agmarknet."""
    attempt_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            return httpx.Response(200, json={"error": "Rate limit exceeded"})
        return httpx.Response(200, json={"records": [{"a": 1}]})

    delays: list[float] = []

    async def recording_sleep(delay: float) -> None:
        delays.append(delay)

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    page = await fetch_page(
        client,
        resource_path="/resource/x",
        api_key="key",
        offset=0,
        limit=10,
        sleep_fn=recording_sleep,
        random_fn=lambda lo, hi: 0.0,
    )

    assert page == [{"a": 1}]
    assert len(delays) == 1


async def test_fetch_page_retries_on_a_real_http_429() -> None:
    """Real shape found discovering the MSP dataset - a different real
    rate-limit signal than Agmarknet's HTTP-200-with-error-body one."""
    attempt_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            return httpx.Response(429, text="")
        return httpx.Response(200, json={"records": [{"a": 1}]})

    async def no_sleep(_delay: float) -> None:
        return None

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    page = await fetch_page(
        client,
        resource_path="/resource/x",
        api_key="key",
        offset=0,
        limit=10,
        sleep_fn=no_sleep,
        random_fn=lambda lo, hi: 0.0,
    )

    assert page == [{"a": 1}]


async def test_fetch_page_raises_after_exhausting_the_retry_schedule() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="")

    async def no_sleep(_delay: float) -> None:
        return None

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(DataGovInRateLimitedError):
        await fetch_page(
            client,
            resource_path="/resource/x",
            api_key="key",
            offset=0,
            limit=10,
            sleep_fn=no_sleep,
            random_fn=lambda lo, hi: 0.0,
        )


async def test_fetch_page_raises_on_a_non_retryable_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(DataGovInError):
        await fetch_page(client, resource_path="/resource/x", api_key="key", offset=0, limit=10)


def _paginating_handler(
    all_records: list[dict[str, object]], *, server_limit: int | None = None
) -> tuple[Callable[[httpx.Request], httpx.Response], list[str | None]]:
    """A realistic paginated-backend fake: slices `all_records` by the
    request's real `offset`, honoring `server_limit` as a hard cap on the
    server's own effective page size regardless of what the client
    requested (reproducing the real MSP-dataset behavior)."""
    offsets_seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset") or 0)
        requested_limit = int(request.url.params.get("limit") or 0)
        offsets_seen.append(request.url.params.get("offset"))
        effective_limit = min(requested_limit, server_limit) if server_limit else requested_limit
        page = all_records[offset : offset + effective_limit]
        return httpx.Response(200, json={"records": page})

    return handler, offsets_seen


async def test_fetch_all_pages_pages_until_an_empty_page() -> None:
    all_records: list[dict[str, object]] = [{"n": 1}, {"n": 2}, {"n": 3}]
    handler, offsets_seen = _paginating_handler(all_records)

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    records = await fetch_all_pages(client, resource_path="/resource/x", api_key="key", page_size=2)

    assert [r["n"] for r in records] == [1, 2, 3]
    assert offsets_seen == ["0", "2", "3"]  # last call confirms a real empty page, not inferred


async def test_fetch_all_pages_handles_a_server_that_silently_caps_the_page_size() -> None:
    """Regression test for a real bug found live, 2026-08-24, building
    app.pipeline.msp: the real MSP-and-cost-of-production data.gov.in
    resource silently capped its own effective page size to 10 regardless
    of the `limit` requested (a real `limit=25` request returned exactly
    10 records). The old "stop at the first page shorter than the
    requested page_size" heuristic would have silently truncated every
    dataset on this resource to its first 10 rows, since every page looks
    "short" relative to what was asked for. This test uses the real shape
    (22 total records, server caps every page to 10) with a client
    `page_size` of 25, matching the exact real numbers from that bug."""
    all_records: list[dict[str, object]] = [{"n": i} for i in range(22)]
    handler, offsets_seen = _paginating_handler(all_records, server_limit=10)

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    records = await fetch_all_pages(
        client, resource_path="/resource/x", api_key="key", page_size=25
    )

    assert len(records) == 22  # not silently truncated to the server's first 10-row page
    assert offsets_seen == ["0", "10", "20", "22"]  # last call confirms a real empty page


async def test_fetch_all_pages_with_no_records_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"records": []})

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    records = await fetch_all_pages(
        client, resource_path="/resource/x", api_key="key", page_size=10
    )

    assert records == []
