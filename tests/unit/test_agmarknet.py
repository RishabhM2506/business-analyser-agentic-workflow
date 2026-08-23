"""Unit tests for `app.pipeline.agmarknet` — real response-shape parsing
(`Arrival_Date` as `dd/mm/yyyy`, price fields as strings, never a float),
the real `{"error": "Rate limit exceeded"}` (HTTP 200) retry path, and
`fetch_all_records`'s short-page pagination stop condition. Uses
`httpx.MockTransport` (this repo's established pattern), never a live
call.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from app.pipeline.agmarknet import (
    AgmarknetRateLimitedError,
    _parse_arrival_date,
    _parse_price_paise,
    _record_from_raw,
    fetch_all_records,
    fetch_page,
)

pytestmark = pytest.mark.unit


def _real_record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "Arrival_Date": "21/02/2023",
        "Commodity": "Green Peas",
        "Commodity_Code": "50",
        "District": "Sangrur",
        "Grade": "FAQ",
        "Market": "Sangrur",
        "Max_Price": "2700",
        "Min_Price": "2400",
        "Modal_Price": "2500",
        "State": "Punjab",
        "Variety": "Green Peas",
    }
    base.update(overrides)
    return base


def test_parse_arrival_date_real_format() -> None:
    assert _parse_arrival_date("21/02/2023") == date(2023, 2, 21)


def test_parse_price_paise_converts_rupees_string_to_paise() -> None:
    assert _parse_price_paise("2500") == 250_000


def test_parse_price_paise_handles_decimal_rupees() -> None:
    assert _parse_price_paise("2500.50") == 250_050


def test_parse_price_paise_is_none_for_blank_string() -> None:
    """Never coerced to 0 - D2's "missing != zero" discipline."""
    assert _parse_price_paise("") is None


def test_parse_price_paise_is_none_for_non_numeric_text() -> None:
    assert _parse_price_paise("NR") is None


def test_parse_price_paise_is_none_for_missing_field() -> None:
    assert _parse_price_paise(None) is None


def test_record_from_raw_builds_a_real_record() -> None:
    record = _record_from_raw(_real_record())

    assert record is not None
    assert record.price_date == date(2023, 2, 21)
    assert record.commodity == "Green Peas"
    assert record.market == "Sangrur"
    assert record.state == "Punjab"
    assert record.modal_price_inr_paise_per_qtl == 250_000
    assert record.raw_payload["Grade"] == "FAQ"


def test_record_from_raw_preserves_a_missing_modal_price_as_none() -> None:
    record = _record_from_raw(_real_record(Modal_Price=""))

    assert record is not None
    assert record.modal_price_inr_paise_per_qtl is None


def test_record_from_raw_returns_none_for_a_malformed_row() -> None:
    assert _record_from_raw({"Commodity": "Green Peas"}) is None


async def test_fetch_page_returns_records_on_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"total": 1, "count": 1, "records": [_real_record()]})

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    page = await fetch_page(client, api_key="key", offset=0, limit=10)

    assert page == [_real_record()]


async def test_fetch_page_retries_on_the_real_rate_limit_shape() -> None:
    """The real, live-confirmed behavior: HTTP 200 with
    `{"error": "Rate limit exceeded"}`, not a 4xx/5xx status."""
    attempt_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            return httpx.Response(200, json={"error": "Rate limit exceeded"})
        return httpx.Response(200, json={"total": 1, "count": 1, "records": [_real_record()]})

    delays: list[float] = []

    async def recording_sleep(delay: float) -> None:
        delays.append(delay)

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    page = await fetch_page(
        client,
        api_key="key",
        offset=0,
        limit=10,
        sleep_fn=recording_sleep,
        random_fn=lambda lo, hi: 0.0,
    )

    assert page == [_real_record()]
    assert len(delays) == 1


async def test_fetch_page_raises_after_exhausting_the_retry_schedule() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "Rate limit exceeded"})

    async def no_sleep(_delay: float) -> None:
        return None

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(AgmarknetRateLimitedError):
        await fetch_page(
            client,
            api_key="key",
            offset=0,
            limit=10,
            sleep_fn=no_sleep,
            random_fn=lambda lo, hi: 0.0,
        )


async def test_fetch_all_records_pages_until_a_short_page() -> None:
    """page_size=2: first page full (2 records) triggers another fetch;
    second page short (1 record) is the real end-of-results signal."""
    pages = [
        [_real_record(Market="A"), _real_record(Market="B")],
        [_real_record(Market="C")],
    ]
    offsets_seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = request.url.params.get("offset")
        offsets_seen.append(offset)
        page = pages[int(offset or 0) // 2]
        return httpx.Response(200, json={"total": 3, "count": len(page), "records": page})

    client = httpx.AsyncClient(
        base_url="https://api.data.gov.in", transport=httpx.MockTransport(handler)
    )
    records = [
        r
        async for r in fetch_all_records(client, api_key="key", commodity="Green Peas", page_size=2)
    ]

    assert [r.market for r in records] == ["A", "B", "C"]
    assert offsets_seen == ["0", "2"]
