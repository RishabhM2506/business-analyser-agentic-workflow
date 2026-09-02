"""Redis-backed FX rate cache (`docs/PLAN.md` §6, D8's exact contract).

Cache key `fx:USD:INR:<YYYY-MM-DD>`. Historical dates cache with no expiry
(immutable — Frankfurter's rate for a past date never changes). Today's
date caches until the next IST midnight, then a fresh fetch is required.

One Frankfurter call per (date) maximum: the cache is always checked
*before* any client call, so two `get_or_fetch` calls for the same date
issue exactly one outbound request between them (`docs/PLAN.md` §8: "a
second call for a date already cached is a MAJOR" — tested directly in
`tests/unit/fx/test_cache.py` by asserting the fake client's call count).

On a genuine Frankfurter failure, falls back to the most recent cached
rate found by scanning backward up to `_MAX_FALLBACK_LOOKBACK_DAYS` days —
`docs/PLAN.md` §1's live verification found Frankfurter's v2 API returns a
rate for every calendar date with no weekend/holiday gap, so this fallback
path is expected to fire only from a genuine outage, not routine gaps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Protocol
from zoneinfo import ZoneInfo

import structlog

from app.fx.client import FxClient, FxRateFetchError

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

_IST = ZoneInfo("Asia/Kolkata")
_KEY_PREFIX = "fx:USD:INR:"
_MAX_FALLBACK_LOOKBACK_DAYS = 7


class RedisLike(Protocol):
    """The exact subset of `redis.asyncio.Redis`'s interface this module
    needs — narrow on purpose so tests use a plain in-memory fake instead
    of a real Redis connection (this repo's "unit tests never touch the
    network" convention, `tests/unit/fx/test_cache.py`)."""

    async def get(self, name: str) -> bytes | str | None: ...
    async def set(self, name: str, value: str, *, ex: int | None = None) -> object: ...


@dataclass(frozen=True)
class FxRateResult:
    """`is_stale=True` means `rate` is a fallback value, not the rate for
    `requested_date` — `actual_date` is the real date it came from,
    surfaced to the user per D8's "FX_STALE... showing which date's rate
    was applied"."""

    rate: Decimal
    is_stale: bool
    requested_date: date
    actual_date: date


def _cache_key(as_of: date) -> str:
    return f"{_KEY_PREFIX}{as_of.isoformat()}"


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _seconds_until_ist_midnight(now: datetime) -> int:
    now_ist = now.astimezone(_IST)
    next_midnight = (now_ist + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((next_midnight - now_ist).total_seconds()))


class FxCache:
    """Orchestrates `RedisLike` + `FxClient` into the D8 cache contract."""

    def __init__(
        self,
        *,
        redis: RedisLike,
        client: FxClient,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._redis = redis
        self._client = client
        self._now_fn = now_fn

    async def get_or_fetch(self, as_of: date) -> FxRateResult:
        cached = await self._redis.get(_cache_key(as_of))
        if cached is not None:
            return FxRateResult(
                rate=Decimal(_decode(cached)),
                is_stale=False,
                requested_date=as_of,
                actual_date=as_of,
            )

        try:
            rate = await self._client.get_rate(as_of)
        except FxRateFetchError:
            logger.warning("fx.fetch_failed", requested_date=as_of.isoformat())
            return await self._fallback(as_of)

        today_ist = self._now_fn().astimezone(_IST).date()
        ttl = _seconds_until_ist_midnight(self._now_fn()) if as_of == today_ist else None
        await self._redis.set(_cache_key(as_of), str(rate), ex=ttl)
        return FxRateResult(rate=rate, is_stale=False, requested_date=as_of, actual_date=as_of)

    async def _fallback(self, as_of: date) -> FxRateResult:
        for offset in range(1, _MAX_FALLBACK_LOOKBACK_DAYS + 1):
            candidate = as_of - timedelta(days=offset)
            cached = await self._redis.get(_cache_key(candidate))
            if cached is not None:
                logger.warning(
                    "fx.stale_fallback_used",
                    requested_date=as_of.isoformat(),
                    fallback_date=candidate.isoformat(),
                )
                return FxRateResult(
                    rate=Decimal(_decode(cached)),
                    is_stale=True,
                    requested_date=as_of,
                    actual_date=candidate,
                )
        raise FxRateFetchError(
            f"Frankfurter unreachable and no cached fallback rate found within "
            f"{_MAX_FALLBACK_LOOKBACK_DAYS} days of {as_of.isoformat()} — never a report crash "
            f"(docs/PLAN.md D8: 'never fail a report because FX failed'), the caller decides "
            f"how to degrade (e.g. omit the FX-dependent figure, mark it unavailable)."
        )
