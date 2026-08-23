"""Unit tests for `app.pipeline.dgcis.DgcisClient` — the CSRF/session
mechanics verified live against the real site (`docs/PLAN.md` §1),
exercised here via `httpx.MockTransport` (this repo's established
pattern, `tests/integration/test_comtrade_client.py`), never a real
network call.
"""

from __future__ import annotations

import httpx
import pytest

from app.pipeline.dgcis import ANNUAL_IMPORT_PATH, DgcisClient, DgcisRequestError

pytestmark = pytest.mark.unit

_TOKEN_PAGE = '<html><body><input type="hidden" name="_token" value="real-token-123"></body></html>'
_RESULT_PAGE = '<html><body><table id="example"><tr><td>real data</td></tr></table></body></html>'


@pytest.mark.unit
async def test_fetch_annual_completes_a_real_get_then_post_round_trip() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, text=_TOKEN_PAGE)
        return httpx.Response(200, text=_RESULT_PAGE)

    client = DgcisClient(transport=httpx.MockTransport(handler))
    html = await client.fetch_annual(
        path=ANNUAL_IMPORT_PATH, hs8="12079100", country_code="409", year="2024"
    )

    assert html == _RESULT_PAGE
    assert [c.method for c in calls] == ["GET", "POST"]
    post_body = calls[1].content.decode()
    assert "_token=real-token-123" in post_body
    assert "searchTerm=12079100" in post_body
    assert "ContEidbi=409" in post_body  # import field name, not the export one
    assert "ContEidbyi=2024" in post_body


@pytest.mark.unit
async def test_fetch_annual_retries_once_on_session_expired() -> None:
    """docs/PLAN.md §1: a 15-minute session TTL means a long-running job
    will hit a 419 ("Page Expired") occasionally - must retry with a
    fresh token, not fail the whole request."""
    post_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_attempts
        if request.method == "GET":
            return httpx.Response(200, text=_TOKEN_PAGE)
        post_attempts += 1
        if post_attempts == 1:
            return httpx.Response(419, text="Page Expired")
        return httpx.Response(200, text=_RESULT_PAGE)

    client = DgcisClient(transport=httpx.MockTransport(handler))
    html = await client.fetch_annual(
        path=ANNUAL_IMPORT_PATH, hs8="12079100", country_code="409", year="2024"
    )

    assert html == _RESULT_PAGE
    assert post_attempts == 2


@pytest.mark.unit
async def test_fetch_annual_raises_after_two_consecutive_session_expirations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_TOKEN_PAGE)
        return httpx.Response(419, text="Page Expired")

    client = DgcisClient(transport=httpx.MockTransport(handler))
    with pytest.raises(DgcisRequestError):
        await client.fetch_annual(
            path=ANNUAL_IMPORT_PATH, hs8="12079100", country_code="409", year="2024"
        )


@pytest.mark.unit
async def test_fetch_annual_raises_on_missing_csrf_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>no token here</body></html>")

    client = DgcisClient(transport=httpx.MockTransport(handler))
    with pytest.raises(DgcisRequestError):
        await client.fetch_annual(
            path=ANNUAL_IMPORT_PATH, hs8="12079100", country_code="409", year="2024"
        )


@pytest.mark.unit
async def test_fetch_annual_raises_on_unexpected_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_TOKEN_PAGE)
        return httpx.Response(500, text="server error")

    client = DgcisClient(transport=httpx.MockTransport(handler))
    with pytest.raises(DgcisRequestError):
        await client.fetch_annual(
            path=ANNUAL_IMPORT_PATH, hs8="12079100", country_code="409", year="2024"
        )
