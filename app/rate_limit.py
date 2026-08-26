"""Per-IP request rate limiting (docs/PLAN.md §6, master brief §8: "rate
limiting per IP/tenant before any model spend occurs"). Hand-rolled
token-bucket, mirroring `app/budget.py`'s `BudgetTracker` shape (an
`asyncio.Lock`-guarded in-memory dict, one process-wide instance per
running server — docs/PLAN.md §1.2: v1 runs one instance).

Finding M5/AWR-07/ARCH-04: this was fully unimplemented despite
`rate_limit_per_minute` existing in `app/settings.py` and being presented
in `.env.example` as if it were already live — nothing throttled
`POST /threads/{id}/messages` by IP or otherwise, so a burst of requests
(a buggy frontend retry loop, or a deliberate one) could exhaust the
entire shared per-day model budget (`app/budget.py`) in seconds, with no
control between an unauthenticated client and that shared counter.

Applied as ASGI middleware (`app/main.py`'s `rate_limit_middleware`), ahead
of `check_hs_code_allowlisted` and the graph entirely — a client that's
currently rate-limited never reaches route dispatch, body parsing, the
response cache, or a single model call.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

# How long a key (client IP) can sit untouched before its bucket is safe to
# evict entirely (architect-review finding, 2026-08-26: `_tokens`/
# `_last_refill` never shrank — every distinct IP ever seen added a
# permanent entry for the life of the process, unlike `app.main.
# _ThreadLockRegistry`, which already solves the identical "unbounded dict
# keyed by a client-controlled/high-cardinality value" shape for thread
# locks). Generously above the time to refill an empty bucket to full
# (60s, `requests_per_minute / 60` tokens/sec) — a genuinely idle key's
# bucket is indistinguishable from a fresh one by the time this fires, so
# eviction changes no observable behavior: `check_and_consume` already
# treats a never-seen key as a full bucket (its own docstring), exactly
# what re-creating an evicted key's entry produces.
_IDLE_EVICTION_SECONDS = 300.0
# Sweep periodically, not on every call — an O(n) scan every single request
# would be wasteful; every 500 calls keeps overhead negligible while still
# bounding worst-case dict size to roughly one sweep interval's worth of
# distinct keys.
_SWEEP_INTERVAL_CALLS = 500


class RateLimiter:
    """Token-bucket limiter, one bucket per key (client IP in practice).

    Refills continuously at `requests_per_minute / 60` tokens per second,
    capped at `requests_per_minute` (a full minute's burst allowance) —
    not a hard fixed window, so a client isn't unfairly denied right at a
    window boundary the way a naive fixed-window counter would.
    """

    def __init__(
        self,
        *,
        requests_per_minute: int,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._requests_per_minute = requests_per_minute
        self._refill_per_second = requests_per_minute / 60.0
        self._now_fn = now_fn
        self._tokens: dict[str, float] = {}
        self._last_refill: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._calls_since_sweep = 0

    async def check_and_consume(self, key: str) -> bool:
        """Return True (and consume one token for `key`) iff capacity is
        available right now; return False (consuming nothing) if `key` is
        currently rate-limited. Fails closed in the sense that matters here
        — a key with no prior history starts at a full bucket, never an
        empty one, so a first-ever request from a new client is never
        itself rejected."""
        async with self._lock:
            now = self._now_fn()
            tokens = self._tokens.get(key, float(self._requests_per_minute))
            last_refill = self._last_refill.get(key, now)
            elapsed = max(0.0, now - last_refill)
            tokens = min(
                float(self._requests_per_minute), tokens + elapsed * self._refill_per_second
            )
            self._last_refill[key] = now

            self._calls_since_sweep += 1
            if self._calls_since_sweep >= _SWEEP_INTERVAL_CALLS:
                self._calls_since_sweep = 0
                self._sweep_idle_keys(now)

            if tokens < 1.0:
                self._tokens[key] = tokens
                return False

            self._tokens[key] = tokens - 1.0
            return True

    def _sweep_idle_keys(self, now: float) -> None:
        """Drop every key untouched for `_IDLE_EVICTION_SECONDS` or longer.
        Called with `self._lock` already held."""
        cutoff = now - _IDLE_EVICTION_SECONDS
        idle_keys = [key for key, last in self._last_refill.items() if last < cutoff]
        for key in idle_keys:
            self._tokens.pop(key, None)
            self._last_refill.pop(key, None)
