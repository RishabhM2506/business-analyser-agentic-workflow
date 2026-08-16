"""Unit tests for `app.rate_limit.RateLimiter` — per-key token-bucket rate
limiting (finding M5/AWR-07/ARCH-04). Pure in-memory state, no I/O, same
classification as `tests/unit/test_budget.py` for the structurally
identical `BudgetTracker`.
"""

from __future__ import annotations

import pytest

from app.rate_limit import RateLimiter


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
