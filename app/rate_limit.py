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

            if tokens < 1.0:
                self._tokens[key] = tokens
                return False

            self._tokens[key] = tokens - 1.0
            return True
