"""Unit tests for `app.pipeline.comtrade_mirror` — D5's two verified query
shapes and D6's fixed retry schedule / jitter / `Retry-After` override /
429-specific circuit breaker. Uses `httpx.MockTransport` (this repo's
established pattern), never a live call.
"""

from __future__ import annotations

import httpx
import pytest

from app.pipeline.comtrade_mirror import (
    RETRY_SCHEDULE_SECONDS,
    ComtradeMirrorCircuitOpenError,
    ComtradeMirrorError,
    _MirrorCircuitBreaker,
    build_query_params,
    fetch_with_retry,
)

pytestmark = pytest.mark.unit


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def _no_sleep(_delay: float) -> None:
    return None


def _zero_jitter(_low: float, _high: float) -> float:
    return 0.0


def test_build_query_params_reporter_role_sets_reporter_code_only() -> None:
    params = build_query_params(role="reporter", cmd_codes=["120791"], periods=["2023"])

    assert params["reporterCode"] == "699"
    assert "partnerCode" not in params  # omitted, verified live to mean "all partners"


def test_build_query_params_partner_role_sets_partner_code_only() -> None:
    params = build_query_params(role="partner", cmd_codes=["120791"], periods=["2023"])

    assert params["partnerCode"] == "699"
    assert "reporterCode" not in params


def test_build_query_params_joins_multiple_periods_and_codes() -> None:
    params = build_query_params(
        role="reporter", cmd_codes=["120791", "090111"], periods=["2022", "2023"]
    )

    assert params["cmdCode"] == "120791,090111"
    assert params["period"] == "2022,2023"
    assert params["flowCode"] == "M,X"


def test_build_query_params_always_pins_the_extra_breakdown_dimensions() -> None:
    """Regression test for a real bug found live, resolved iteratively:
    Comtrade rows carry three extra breakdown dimensions (partner2Code,
    motCode, customsCode) not tracked by raw_comtrade_records' unique
    key. Leaving any one unconstrained returned genuine duplicate
    (period, reporter, partner, flow, cmd) rows that Postgres's
    ON CONFLICT DO UPDATE correctly refused to upsert twice in one
    statement. Pinning all three to their aggregate value eliminated
    every duplicate in a real response."""
    params = build_query_params(role="reporter", cmd_codes=["120791"], periods=["2023"])

    assert params["partner2Code"] == "0"
    assert params["motCode"] == "0"
    assert params["customsCode"] == "C00"


async def test_fetch_with_retry_succeeds_on_first_try_with_no_delay() -> None:
    sleeps: list[float] = []

    async def recording_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    client = httpx.AsyncClient(
        base_url="https://comtradeapi.un.org", transport=httpx.MockTransport(handler)
    )
    result = await fetch_with_retry(
        client,
        params={},
        api_key="key",
        breaker=_MirrorCircuitBreaker(),
        sleep_fn=recording_sleep,
    )

    assert result == {"data": []}
    assert sleeps == []


async def test_fetch_with_retry_follows_the_fixed_schedule_with_jitter_bounds() -> None:
    """D6: fixed [30, 60, 120, 300]s schedule, ±20% jitter."""
    attempt_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count <= 2:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"data": ["ok"]})

    delays: list[float] = []

    async def recording_sleep(delay: float) -> None:
        delays.append(delay)

    client = httpx.AsyncClient(
        base_url="https://comtradeapi.un.org", transport=httpx.MockTransport(handler)
    )
    result = await fetch_with_retry(
        client,
        params={},
        api_key="key",
        breaker=_MirrorCircuitBreaker(),
        sleep_fn=recording_sleep,
    )

    assert result == {"data": ["ok"]}
    assert len(delays) == 2
    for observed, scheduled in zip(delays, RETRY_SCHEDULE_SECONDS, strict=False):
        jitter = scheduled * 0.2
        assert scheduled - jitter <= observed <= scheduled + jitter


async def test_fetch_with_retry_retry_after_header_overrides_the_schedule() -> None:
    attempt_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"data": []})

    delays: list[float] = []

    async def recording_sleep(delay: float) -> None:
        delays.append(delay)

    client = httpx.AsyncClient(
        base_url="https://comtradeapi.un.org", transport=httpx.MockTransport(handler)
    )
    await fetch_with_retry(
        client, params={}, api_key="key", breaker=_MirrorCircuitBreaker(), sleep_fn=recording_sleep
    )

    assert delays == [7.0]  # not the schedule's 30s - Retry-After wins


async def test_fetch_with_retry_raises_after_exhausting_the_schedule() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="always limited")

    client = httpx.AsyncClient(
        base_url="https://comtradeapi.un.org", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ComtradeMirrorError):
        await fetch_with_retry(
            client,
            params={},
            api_key="key",
            breaker=_MirrorCircuitBreaker(),
            sleep_fn=_no_sleep,
            random_fn=_zero_jitter,
        )


async def test_fetch_with_retry_raises_immediately_on_a_non_retryable_4xx() -> None:
    """A 4xx other than 429 means our own request is malformed - never
    retried, matching app.tools.comtrade_client's identical reasoning."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(400, text="bad request")

    client = httpx.AsyncClient(
        base_url="https://comtradeapi.un.org", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ComtradeMirrorError):
        await fetch_with_retry(
            client, params={}, api_key="key", breaker=_MirrorCircuitBreaker(), sleep_fn=_no_sleep
        )

    assert call_count == 1  # never retried


def test_circuit_breaker_opens_after_three_consecutive_429s() -> None:
    clock = _FakeClock()
    breaker = _MirrorCircuitBreaker(threshold=3, reset_seconds=900.0, now_fn=clock)

    breaker.record_429()
    breaker.record_429()
    assert breaker.is_open is False
    breaker.record_429()
    assert breaker.is_open is True

    with pytest.raises(ComtradeMirrorCircuitOpenError):
        breaker.before_call()


def test_circuit_breaker_a_non_429_resets_the_consecutive_count() -> None:
    clock = _FakeClock()
    breaker = _MirrorCircuitBreaker(threshold=3, reset_seconds=900.0, now_fn=clock)

    breaker.record_429()
    breaker.record_429()
    breaker.record_non_429()
    breaker.record_429()
    breaker.record_429()

    assert breaker.is_open is False  # only 2 consecutive since the reset


def test_circuit_breaker_half_opens_after_the_reset_window() -> None:
    clock = _FakeClock()
    breaker = _MirrorCircuitBreaker(threshold=3, reset_seconds=900.0, now_fn=clock)

    breaker.record_429()
    breaker.record_429()
    breaker.record_429()
    assert breaker.is_open is True

    clock.now += 899.0
    with pytest.raises(ComtradeMirrorCircuitOpenError):
        breaker.before_call()

    clock.now += 1.0  # now exactly at the reset window
    breaker.before_call()  # half-open: does not raise, trial call allowed through
