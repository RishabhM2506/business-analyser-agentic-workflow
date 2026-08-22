"""Unit tests for `app.fx.client.FrankfurterClient`. Uses `httpx.MockTransport`
(this repo's established pattern, `tests/integration/test_comtrade_client.py`)
— never a real network call."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import httpx
import pytest

from app.fx.client import FrankfurterClient, FxRateFetchError

_TEST_DATE = date(2021, 6, 15)


@pytest.mark.unit
async def test_get_rate_returns_the_rate_as_a_decimal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/rate/USD/INR"
        assert request.url.params["date"] == "2021-06-15"
        return httpx.Response(
            200, json={"date": "2021-06-15", "base": "USD", "quote": "INR", "rate": 73.349}
        )

    client = FrankfurterClient(transport=httpx.MockTransport(handler))
    rate = await client.get_rate(_TEST_DATE)

    assert rate == Decimal("73.349")


@pytest.mark.unit
async def test_get_rate_raises_on_non_200_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    client = FrankfurterClient(transport=httpx.MockTransport(handler))
    with pytest.raises(FxRateFetchError):
        await client.get_rate(_TEST_DATE)


@pytest.mark.unit
async def test_get_rate_raises_on_malformed_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client = FrankfurterClient(transport=httpx.MockTransport(handler))
    with pytest.raises(FxRateFetchError):
        await client.get_rate(_TEST_DATE)


@pytest.mark.unit
async def test_get_rate_raises_on_missing_rate_field() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"date": "2021-06-15", "base": "USD", "quote": "INR"})

    client = FrankfurterClient(transport=httpx.MockTransport(handler))
    with pytest.raises(FxRateFetchError):
        await client.get_rate(_TEST_DATE)


@pytest.mark.unit
async def test_get_rate_raises_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = FrankfurterClient(transport=httpx.MockTransport(handler))
    with pytest.raises(FxRateFetchError):
        await client.get_rate(_TEST_DATE)
