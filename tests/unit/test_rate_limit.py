"""Unit tests for `app.rate_limit.RateLimiter` — per-key token-bucket rate
limiting (finding M5/AWR-07/ARCH-04). Pure in-memory state, no I/O, same
classification as `tests/unit/test_budget.py` for the structurally
identical `BudgetTracker`.
"""

from __future__ import annotations

import pytest

from app.rate_limit import _IDLE_EVICTION_SECONDS, _SWEEP_INTERVAL_CALLS, RateLimiter


@pytest.mark.unit
async def test_check_and_consume_allows_calls_up_to_the_per_minute_ceiling() -> None:
    limiter = RateLimiter(requests_per_minute=3, now_fn=lambda: 0.0)
    assert await limiter.check_and_consume("1.2.3.4") is True
    assert await limiter.check_and_consume("1.2.3.4") is True
    assert await limiter.check_and_consume("1.2.3.4") is True


@pytest.mark.unit
async def test_check_and_consume_rejects_once_the_ceiling_is_exceeded() -> None:
    limiter = RateLimiter(requests_per_minute=2, now_fn=lambda: 0.0)
    assert await limiter.check_and_consume("1.2.3.4") is True
    assert await limiter.check_and_consume("1.2.3.4") is True
    assert await limiter.check_and_consume("1.2.3.4") is False  # no time elapsed, no refill


@pytest.mark.unit
async def test_rejection_does_not_consume_a_token() -> None:
    limiter = RateLimiter(requests_per_minute=1, now_fn=lambda: 0.0)
    assert await limiter.check_and_consume("1.2.3.4") is True
    assert await limiter.check_and_consume("1.2.3.4") is False
    assert await limiter.check_and_consume("1.2.3.4") is False  # still rejected, not "recovered"


@pytest.mark.unit
async def test_bucket_is_scoped_per_key_not_global() -> None:
    limiter = RateLimiter(requests_per_minute=1, now_fn=lambda: 0.0)
    assert await limiter.check_and_consume("1.2.3.4") is True
    # A different key (client IP) has its own independent bucket.
    assert await limiter.check_and_consume("5.6.7.8") is True


@pytest.mark.unit
async def test_a_brand_new_key_starts_with_a_full_bucket() -> None:
    # A first-ever request from a client with no prior history must never
    # itself be rejected - the bucket starts full, not empty.
    limiter = RateLimiter(requests_per_minute=60, now_fn=lambda: 0.0)
    assert await limiter.check_and_consume("never-seen-before") is True


@pytest.mark.unit
async def test_tokens_refill_continuously_over_elapsed_time() -> None:
    clock = iter([0.0, 0.0, 30.0])  # two calls at t=0 exhaust a 1/min bucket, refill by t=30
    limiter = RateLimiter(requests_per_minute=1, now_fn=lambda: next(clock))
    assert await limiter.check_and_consume("1.2.3.4") is True
    assert await limiter.check_and_consume("1.2.3.4") is False
    # 30s elapsed at 1 token/60s = 0.5 tokens refilled - still under 1, still rejected.
    assert await limiter.check_and_consume("1.2.3.4") is False


@pytest.mark.unit
async def test_tokens_refill_fully_after_enough_elapsed_time() -> None:
    clock = iter([0.0, 60.0])  # one call at t=0, next check a full minute later
    limiter = RateLimiter(requests_per_minute=1, now_fn=lambda: next(clock))
    assert await limiter.check_and_consume("1.2.3.4") is True
    assert await limiter.check_and_consume("1.2.3.4") is True  # fully refilled after 60s


@pytest.mark.unit
async def test_refill_never_exceeds_the_per_minute_cap() -> None:
    # A very long idle period must not let tokens accumulate beyond one
    # minute's worth of burst allowance.
    calls = 0

    def _clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 10_000.0  # t=0 for the seed call, far later after that

    limiter = RateLimiter(requests_per_minute=5, now_fn=_clock)
    await limiter.check_and_consume("1.2.3.4")  # seed the key at t=0, consumes 1 of 5
    # After a huge idle gap, only 5 calls total should succeed in a row
    # (the bucket refills to its cap, not beyond it), not more.
    results = [await limiter.check_and_consume("1.2.3.4") for _ in range(6)]
    assert results == [True, True, True, True, True, False]


# --- idle-key eviction (architect-review finding, 2026-08-26) -----------------
#
# `_tokens`/`_last_refill` used to grow forever — every distinct key
# (client IP) ever seen added a permanent entry for the life of the
# process. These tests drive the sweep directly via `now_fn`/enough calls,
# never real wall-clock time.


@pytest.mark.unit
async def test_sweep_evicts_a_key_idle_past_the_eviction_threshold() -> None:
    limiter = RateLimiter(requests_per_minute=5, now_fn=lambda: 0.0)
    await limiter.check_and_consume("1.2.3.4")
    assert "1.2.3.4" in limiter._tokens

    # Advance past the idle threshold and burn enough other calls (from a
    # different key, so "1.2.3.4" itself is never touched again) to trigger
    # the periodic sweep.
    later = _IDLE_EVICTION_SECONDS + 1.0
    limiter._now_fn = lambda: later
    for _ in range(_SWEEP_INTERVAL_CALLS):
        await limiter.check_and_consume("other-key-not-under-test")

    assert "1.2.3.4" not in limiter._tokens
    assert "1.2.3.4" not in limiter._last_refill


@pytest.mark.unit
async def test_sweep_does_not_evict_a_recently_touched_key() -> None:
    limiter = RateLimiter(requests_per_minute=5, now_fn=lambda: 0.0)
    await limiter.check_and_consume("1.2.3.4")

    # Advance only slightly (well under the idle threshold) and trigger the
    # sweep interval.
    limiter._now_fn = lambda: 1.0
    for _ in range(_SWEEP_INTERVAL_CALLS):
        await limiter.check_and_consume("other-key")

    assert "1.2.3.4" in limiter._tokens  # still tracked - was not idle long enough


@pytest.mark.unit
async def test_an_evicted_key_behaves_identically_to_a_never_seen_one() -> None:
    """Eviction must be a pure memory-hygiene optimization, not an
    observable behavior change: a key whose entry was swept starts with a
    full bucket again, exactly like a genuinely new key — never an empty or
    partially-refilled one, and never itself rejected."""
    limiter = RateLimiter(requests_per_minute=1, now_fn=lambda: 0.0)
    await limiter.check_and_consume("1.2.3.4")  # exhausts the 1-token bucket
    assert await limiter.check_and_consume("1.2.3.4") is False

    later = _IDLE_EVICTION_SECONDS + 1.0
    limiter._now_fn = lambda: later
    for _ in range(_SWEEP_INTERVAL_CALLS):
        await limiter.check_and_consume("other-key")
    assert "1.2.3.4" not in limiter._tokens  # confirmed evicted

    assert await limiter.check_and_consume("1.2.3.4") is True  # fresh full bucket, not rejected
