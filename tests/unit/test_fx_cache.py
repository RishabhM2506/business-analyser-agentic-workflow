"""Unit tests for `app.fx.cache.FxCache` — the `docs/PLAN.md` §6/§8 D8
contract: one Frankfurter call per date maximum, historical dates cached
with no expiry, today's date expires at IST midnight, and a genuine
Frankfurter failure falls back to the most recent cached rate rather than
crashing the report. A fake in-memory `RedisLike` + a fake `FxClient`
double — never a real Redis connection or network call.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from app.fx.cache import FxCache
from app.fx.client import FxRateFetchError

_HISTORICAL_DATE = date(2021, 6, 15)


class _FakeRedis:
    """In-memory stand-in for `app.fx.cache.RedisLike`."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, name: str) -> str | None:
        return self._store.get(name)

    async def set(self, name: str, value: str, *, ex: int | None = None) -> None:
        # `ex` (TTL seconds) isn't simulated — these tests assert cache
        # *presence*/*value*, not real expiry timing, which would require
        # a real clock/sleep and isn't what "one call per date" is testing.
        self._store[name] = value


class _CountingFxClient:
    def __init__(self, rate: Decimal) -> None:
        self._rate = rate
        self.calls: list[date] = []

    async def get_rate(self, as_of: date) -> Decimal:
        self.calls.append(as_of)
        return self._rate


class _AlwaysFailingFxClient:
    async def get_rate(self, as_of: date) -> Decimal:
        raise FxRateFetchError("simulated Frankfurter outage")


@pytest.mark.unit
async def test_get_or_fetch_calls_the_client_exactly_once_for_a_repeated_date() -> None:
    """docs/PLAN.md §8: 'a second call for a date already cached is a MAJOR.'"""
    client = _CountingFxClient(Decimal("73.349"))
    cache = FxCache(redis=_FakeRedis(), client=client)

    first = await cache.get_or_fetch(_HISTORICAL_DATE)
    second = await cache.get_or_fetch(_HISTORICAL_DATE)

    assert first.rate == Decimal("73.349")
    assert second.rate == Decimal("73.349")
    assert client.calls == [_HISTORICAL_DATE]  # not called twice


@pytest.mark.unit
async def test_get_or_fetch_returns_the_real_rate_for_a_historical_date() -> None:
    client = _CountingFxClient(Decimal("83.27"))
    cache = FxCache(redis=_FakeRedis(), client=client)

    result = await cache.get_or_fetch(date(2024, 1, 6))

    assert result.rate == Decimal("83.27")
    assert result.is_stale is False
    assert result.actual_date == date(2024, 1, 6)


@pytest.mark.unit
async def test_get_or_fetch_falls_back_to_the_most_recent_cached_rate_on_failure() -> None:
    """D8: 'never fail a report because FX failed... mark every affected
    figure FX_STALE, and surface it... with the rate's actual date.'"""
    redis = _FakeRedis()
    working_client = _CountingFxClient(Decimal("95.50"))
    cache = FxCache(redis=redis, client=working_client)
    # Populate a real cached rate for the day before the one we'll ask for.
    await cache.get_or_fetch(date(2026, 8, 20))

    failing_cache = FxCache(redis=redis, client=_AlwaysFailingFxClient())
    result = await failing_cache.get_or_fetch(date(2026, 8, 21))

    assert result.is_stale is True
    assert result.rate == Decimal("95.50")
    assert result.requested_date == date(2026, 8, 21)
    assert result.actual_date == date(2026, 8, 20)


@pytest.mark.unit
async def test_get_or_fetch_raises_when_no_fallback_exists_within_the_lookback_window() -> None:
    cache = FxCache(redis=_FakeRedis(), client=_AlwaysFailingFxClient())

    with pytest.raises(FxRateFetchError):
        await cache.get_or_fetch(date(2026, 8, 21))


@pytest.mark.unit
async def test_todays_rate_is_cached_with_a_ttl_and_historical_rate_with_none() -> None:
    """Verifies the `ex` argument passed to `redis.set` directly — TTL for
    today, no expiry (`ex=None`) for a historical date — rather than only
    checking the cached value, since that's the actual D8 contract."""

    class _RecordingRedis(_FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self.set_calls: list[tuple[str, int | None]] = []

        async def set(self, name: str, value: str, *, ex: int | None = None) -> None:
            self.set_calls.append((name, ex))
            await super().set(name, value, ex=ex)

    fixed_now = datetime(2026, 8, 21, 10, 0, tzinfo=UTC)
    redis = _RecordingRedis()
    cache = FxCache(
        redis=redis, client=_CountingFxClient(Decimal("95.67")), now_fn=lambda: fixed_now
    )

    await cache.get_or_fetch(date(2026, 8, 21))  # "today" relative to fixed_now (IST)
    await cache.get_or_fetch(_HISTORICAL_DATE)

    today_key, today_ttl = redis.set_calls[0]
    historical_key, historical_ttl = redis.set_calls[1]
    assert today_key.endswith("2026-08-21")
    assert today_ttl is not None and today_ttl > 0
    assert historical_key.endswith("2021-06-15")
    assert historical_ttl is None
