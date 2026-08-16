"""Integration tests for `app.tools.comtrade_client.ComtradeClient`.

Every test runs against `httpx.MockTransport` — the exact request-building,
retry, circuit-breaker, and response-parsing code path used in production
runs here too, but zero real network calls are made (docs/PLAN.md §7: "no
live external calls"). Fixtures under `tests/fixtures/comtrade/` are real
(or real-values-reconstructed — see that directory's README) captures from
`app/tools/comtrade_client.py`'s live verification, not invented shapes.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.tools.comtrade_client import (
    ComtradeCircuitOpenError,
    ComtradeClient,
    ComtradeSchemaValidationError,
    ComtradeTimeoutError,
    ComtradeUpstreamError,
    is_aggregate_partner_code,
    resolve_partner_country,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "comtrade"


def _load_fixture(name: str) -> dict[str, object]:
    result: dict[str, object] = json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))
    return result


def _json_handler(payload: dict[str, object], *, status_code: int = 200) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return httpx.MockTransport(handler)


class _CountingHandler:
    """Records every request it sees; returns a fixed response every time."""

    def __init__(self, payload: dict[str, object], *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status_code, json=self.payload)


class _FlakyHandler:
    """Fails with a retryable status for the first `fail_times` calls, then
    succeeds — used to prove bounded retry actually recovers."""

    def __init__(
        self, payload: dict[str, object], *, fail_times: int, fail_status: int = 500
    ) -> None:
        self.payload = payload
        self.fail_times = fail_times
        self.fail_status = fail_status
        self.calls = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.fail_times:
            return httpx.Response(
                self.fail_status, json={"count": 0, "data": [], "error": "temporary"}
            )
        return httpx.Response(200, json=self.payload)


class _AlwaysTimeoutHandler:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        raise httpx.ReadTimeout("simulated timeout", request=request)


class _AlwaysFailingHandler:
    def __init__(self, status_code: int = 500) -> None:
        self.calls = 0
        self.status_code = status_code

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(self.status_code, json={"count": 0, "data": [], "error": "boom"})


@pytest.mark.integration
async def test_fetch_flow_parses_real_fixture_into_normalized_records() -> None:
    payload = _load_fixture("horses_import_2022.json")
    client = ComtradeClient(api_key="test-key", transport=_json_handler(payload))
    try:
        records = await client.fetch_flow(
            hs_code="010121", flow="import", year_start=2022, year_end=2022
        )
    finally:
        await client.aclose()

    assert len(records) == 3
    by_partner = {r.partner_code: r for r in records}

    world = by_partner["0"]
    assert world.partner_country == "World"
    assert world.is_provisional is True  # isReported: false
    assert world.value == pytest.approx(1992455.942)
    assert world.year == 2022
    assert world.hs_code == "010121"
    assert world.flow == "import"

    uk = by_partner["826"]
    assert uk.partner_country == "United Kingdom"
    assert uk.is_provisional is False  # isReported: true
    assert uk.value == pytest.approx(1861448.346)

    australia = by_partner["36"]
    assert australia.partner_country == "Australia"
    assert australia.is_provisional is False


@pytest.mark.integration
async def test_fetch_flow_handles_many_partner_rows_in_one_response() -> None:
    payload = _load_fixture("tea_export_2023_partial.json")
    client = ComtradeClient(api_key="test-key", transport=_json_handler(payload))
    try:
        records = await client.fetch_flow(
            hs_code="090240", flow="export", year_start=2023, year_end=2023
        )
    finally:
        await client.aclose()

    assert len(records) == 15
    assert all(r.flow == "export" for r in records)
    assert all(r.is_provisional for r in records)  # every real row here was isReported: false


@pytest.mark.integration
async def test_fetch_flow_handles_empty_result() -> None:
    payload = _load_fixture("empty.json")
    client = ComtradeClient(api_key="test-key", transport=_json_handler(payload))
    try:
        records = await client.fetch_flow(
            hs_code="999999", flow="import", year_start=2023, year_end=2023
        )
    finally:
        await client.aclose()
    assert records == []


@pytest.mark.integration
async def test_fetch_flow_issues_one_request_per_year() -> None:
    payload = _load_fixture("empty.json")
    handler = _CountingHandler(payload)
    client = ComtradeClient(api_key="test-key", transport=httpx.MockTransport(handler))
    try:
        await client.fetch_flow(hs_code="010121", flow="import", year_start=2019, year_end=2023)
    finally:
        await client.aclose()

    assert len(handler.requests) == 5
    periods = sorted(req.url.params["period"] for req in handler.requests)
    assert periods == ["2019", "2020", "2021", "2022", "2023"]
    # Every request carries the reporter (India) and the subscription-key header.
    for req in handler.requests:
        assert req.url.params["reporterCode"] == "699"
        assert req.headers["Ocp-Apim-Subscription-Key"] == "test-key"


@pytest.mark.integration
async def test_fetch_flow_rejects_inverted_year_range() -> None:
    client = ComtradeClient(
        api_key="test-key", transport=_json_handler(_load_fixture("empty.json"))
    )
    try:
        with pytest.raises(ValueError, match="year_start"):
            await client.fetch_flow(hs_code="010121", flow="import", year_start=2023, year_end=2019)
    finally:
        await client.aclose()


@pytest.mark.integration
async def test_retry_recovers_from_transient_5xx() -> None:
    handler = _FlakyHandler(_load_fixture("empty.json"), fail_times=2, fail_status=503)
    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(handler), max_attempts=3
    )
    try:
        records = await client.fetch_flow(
            hs_code="010121", flow="import", year_start=2022, year_end=2022
        )
    finally:
        await client.aclose()
    assert records == []
    assert handler.calls == 3  # 2 failures + 1 success, within max_attempts


@pytest.mark.integration
async def test_retry_exhausted_raises_upstream_error() -> None:
    handler = _AlwaysFailingHandler(status_code=500)
    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(handler), max_attempts=3
    )
    try:
        with pytest.raises(ComtradeUpstreamError):
            await client.fetch_flow(hs_code="010121", flow="import", year_start=2022, year_end=2022)
    finally:
        await client.aclose()
    assert handler.calls == 3  # bounded — exactly max_attempts, never unbounded


@pytest.mark.integration
async def test_rate_limit_429_is_retried() -> None:
    handler = _FlakyHandler(_load_fixture("empty.json"), fail_times=1, fail_status=429)
    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(handler), max_attempts=3
    )
    try:
        records = await client.fetch_flow(
            hs_code="010121", flow="import", year_start=2022, year_end=2022
        )
    finally:
        await client.aclose()
    assert records == []
    assert handler.calls == 2


@pytest.mark.integration
async def test_client_error_4xx_is_not_retried() -> None:
    handler = _AlwaysFailingHandler(status_code=400)
    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(handler), max_attempts=3
    )
    try:
        with pytest.raises(ComtradeUpstreamError):
            await client.fetch_flow(hs_code="010121", flow="import", year_start=2022, year_end=2022)
    finally:
        await client.aclose()
    assert handler.calls == 1  # not retried — a bad request retried verbatim can't succeed


@pytest.mark.integration
async def test_timeout_maps_to_comtrade_timeout_error() -> None:
    handler = _AlwaysTimeoutHandler()
    client = ComtradeClient(
        api_key="test-key", transport=httpx.MockTransport(handler), max_attempts=2
    )
    try:
        with pytest.raises(ComtradeTimeoutError):
            await client.fetch_flow(hs_code="010121", flow="import", year_start=2022, year_end=2022)
    finally:
        await client.aclose()
    assert handler.calls == 2


@pytest.mark.integration
async def test_malformed_response_raises_schema_validation_error() -> None:
    payload = _load_fixture("malformed.json")
    client = ComtradeClient(api_key="test-key", transport=_json_handler(payload), max_attempts=1)
    try:
        with pytest.raises(ComtradeSchemaValidationError):
            await client.fetch_flow(hs_code="010121", flow="import", year_start=2022, year_end=2022)
    finally:
        await client.aclose()


@pytest.mark.integration
async def test_populated_error_field_raises_upstream_error() -> None:
    payload = _load_fixture("error_envelope.json")
    client = ComtradeClient(api_key="test-key", transport=_json_handler(payload), max_attempts=1)
    try:
        with pytest.raises(ComtradeUpstreamError):
            await client.fetch_flow(hs_code="010121", flow="import", year_start=2022, year_end=2022)
    finally:
        await client.aclose()


@pytest.mark.integration
async def test_circuit_breaker_opens_after_threshold_and_fails_fast() -> None:
    from app.tools.comtrade_client import (
        _CircuitBreaker,
    )

    handler = _AlwaysFailingHandler(status_code=500)
    client = ComtradeClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        max_attempts=1,  # isolate circuit-breaker behavior from retry behavior
        circuit_breaker=_CircuitBreaker(failure_threshold=2, reset_timeout_seconds=999.0),
    )
    try:
        # Two failing years trip the breaker (threshold=2).
        with pytest.raises(ComtradeUpstreamError):
            await client.fetch_flow(hs_code="010121", flow="import", year_start=2020, year_end=2020)
        with pytest.raises(ComtradeUpstreamError):
            await client.fetch_flow(hs_code="010121", flow="import", year_start=2021, year_end=2021)
        calls_before_open = handler.calls

        # Circuit is now open — the next call must fail fast, no network call.
        with pytest.raises(ComtradeCircuitOpenError):
            await client.fetch_flow(hs_code="010121", flow="import", year_start=2022, year_end=2022)
        assert handler.calls == calls_before_open  # handler was never invoked again
    finally:
        await client.aclose()


@pytest.mark.integration
async def test_circuit_breaker_half_open_allows_trial_after_cooldown() -> None:
    from app.tools.comtrade_client import _CircuitBreaker

    handler = _FlakyHandler(_load_fixture("empty.json"), fail_times=2, fail_status=500)
    client = ComtradeClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        max_attempts=1,
        circuit_breaker=_CircuitBreaker(failure_threshold=2, reset_timeout_seconds=0.0),
    )
    try:
        with pytest.raises(ComtradeUpstreamError):
            await client.fetch_flow(hs_code="010121", flow="import", year_start=2020, year_end=2020)
        with pytest.raises(ComtradeUpstreamError):
            await client.fetch_flow(hs_code="010121", flow="import", year_start=2021, year_end=2021)

        # reset_timeout_seconds=0.0 -> immediately half-open; the next (3rd,
        # succeeding per _FlakyHandler) call is let through and closes the circuit.
        records = await client.fetch_flow(
            hs_code="010121", flow="import", year_start=2022, year_end=2022
        )
        assert records == []
        assert handler.calls == 3
    finally:
        await client.aclose()


@pytest.mark.integration
def test_resolve_partner_country_known_codes() -> None:
    assert resolve_partner_country("699") == "India"
    assert resolve_partner_country("842") == "USA"
    assert resolve_partner_country("0") == "World"


@pytest.mark.integration
def test_resolve_partner_country_unknown_code_falls_back_without_raising() -> None:
    assert resolve_partner_country("999999") == "Unknown partner (999999)"


@pytest.mark.integration
def test_is_aggregate_partner_code_matches_verified_catch_alls() -> None:
    assert is_aggregate_partner_code("0") is True  # World
    assert is_aggregate_partner_code("837") is True  # Bunkers
    assert is_aggregate_partner_code("838") is True  # Free Zones
    assert is_aggregate_partner_code("899") is True  # Areas, nes
    assert is_aggregate_partner_code("490") is True  # Other Asia, nes
    assert is_aggregate_partner_code("699") is False  # India — a real country
    assert is_aggregate_partner_code("842") is False  # USA — a real country


@pytest.mark.integration
def test_is_aggregate_partner_code_unknown_code_defaults_to_not_aggregate() -> None:
    assert is_aggregate_partner_code("999999") is False
