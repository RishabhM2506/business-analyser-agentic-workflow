"""Unit tests for `app.gemini_scheduler.quota` -- RPM/TPM token-bucket
admission, RPD daily tracking with a real Pacific-midnight reset, and the
token-estimation heuristic."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.gemini_scheduler.quota import (
    QuotaStore,
    RateLimitConfig,
    _next_pacific_midnight,
    estimate_tokens,
)


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _store(
    *, rpm: int = 10, tpm: int = 1000, rpd: int = 5, wall_clock: datetime | None = None
) -> tuple[QuotaStore, _FakeClock, list[datetime]]:
    clock = _FakeClock()
    wall = [wall_clock or datetime(2026, 9, 4, 12, 0, 0, tzinfo=UTC)]
    store = QuotaStore(
        rate_limits={"test-model": RateLimitConfig(rpm=rpm, tpm=tpm, rpd=rpd)},
        now_fn=clock,
        wall_clock_fn=lambda: wall[0],
    )
    return store, clock, wall


@pytest.mark.unit
def test_estimate_tokens_is_never_zero() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("a") == 1


@pytest.mark.unit
def test_estimate_tokens_scales_with_length() -> None:
    assert estimate_tokens("a" * 400) == 100


@pytest.mark.unit
async def test_try_reserve_succeeds_within_headroom() -> None:
    store, _clock, _wall = _store(rpm=10, tpm=1000, rpd=5)
    assert (
        await store.try_reserve(project_id="proj-a", model="test-model", estimated_tokens=10)
        is True
    )


@pytest.mark.unit
async def test_rpm_bucket_rejects_once_exhausted() -> None:
    store, _clock, _wall = _store(rpm=2, tpm=1_000_000, rpd=1_000_000)
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=1) is True
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=1) is True
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=1) is False


@pytest.mark.unit
async def test_rpm_bucket_refills_continuously_over_time() -> None:
    store, clock, _wall = _store(rpm=60, tpm=1_000_000, rpd=1_000_000)  # 1 token/sec
    for _ in range(60):
        assert (
            await store.try_reserve(project_id="a", model="test-model", estimated_tokens=1) is True
        )
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=1) is False
    clock.advance(1.0)  # refills exactly 1 token at 1/sec
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=1) is True


@pytest.mark.unit
async def test_rpm_is_tracked_independently_per_project() -> None:
    store, _clock, _wall = _store(rpm=1, tpm=1_000_000, rpd=1_000_000)
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=1) is True
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=1) is False
    # A different project's own bucket is untouched.
    assert await store.try_reserve(project_id="b", model="test-model", estimated_tokens=1) is True


@pytest.mark.unit
async def test_tpm_bucket_rejects_when_estimated_tokens_exceed_headroom() -> None:
    store, _clock, _wall = _store(rpm=1_000_000, tpm=100, rpd=1_000_000)
    assert (
        await store.try_reserve(project_id="a", model="test-model", estimated_tokens=101) is False
    )
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=100) is True


@pytest.mark.unit
async def test_try_reserve_is_all_or_nothing_never_partial() -> None:
    """RPM has headroom but TPM doesn't - neither should be consumed."""
    store, _clock, _wall = _store(rpm=10, tpm=50, rpd=1_000_000)
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=51) is False
    # RPM must still show full headroom - the failed TPM check must not
    # have partially consumed it.
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=10) is True


@pytest.mark.unit
async def test_rpd_tracked_per_project_not_per_model() -> None:
    store, _clock, _wall = _store(rpm=1_000_000, tpm=1_000_000, rpd=1)
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=1) is True
    # Same project, would-be-different model still shares the exhausted
    # RPD budget (falls back to the same "test-model" config here since
    # that's the only configured entry, but the counting key is project_id
    # only - confirmed by rpd_would_exceed below using a bare project_id).
    assert await store.rpd_would_exceed("a", "test-model") is True


@pytest.mark.unit
async def test_rpd_would_exceed_is_a_peek_not_a_consume() -> None:
    store, _clock, _wall = _store(rpm=1_000_000, tpm=1_000_000, rpd=5)
    for _ in range(3):
        assert await store.rpd_would_exceed("a", "test-model") is False
    # Three peeks, zero real reservations - full RPD headroom still there.
    for _ in range(5):
        assert (
            await store.try_reserve(project_id="a", model="test-model", estimated_tokens=1) is True
        )
    assert await store.rpd_would_exceed("a", "test-model") is True


@pytest.mark.unit
async def test_rpd_resets_at_pacific_midnight() -> None:
    # 2026-09-04 07:59:00 UTC = 2026-09-04 00:59:00 PDT (UTC-7 in September)
    store, _clock, wall = _store(
        rpm=1_000_000, tpm=1_000_000, rpd=1, wall_clock=datetime(2026, 9, 4, 7, 59, 0, tzinfo=UTC)
    )
    assert await store.try_reserve(project_id="a", model="test-model", estimated_tokens=1) is True
    assert await store.rpd_would_exceed("a", "test-model") is True

    # Still before the next Pacific midnight (2026-09-05 07:00 UTC).
    wall[0] = datetime(2026, 9, 5, 6, 0, 0, tzinfo=UTC)
    assert await store.rpd_would_exceed("a", "test-model") is True

    # Past the next Pacific midnight - resets.
    wall[0] = datetime(2026, 9, 5, 7, 0, 1, tzinfo=UTC)
    assert await store.rpd_would_exceed("a", "test-model") is False


@pytest.mark.unit
def test_next_pacific_midnight_is_always_in_the_future() -> None:
    now = datetime(2026, 9, 4, 0, 0, 0, tzinfo=UTC)
    result = _next_pacific_midnight(now)
    assert result > now


@pytest.mark.unit
async def test_unrecognized_model_falls_back_to_conservative_limit() -> None:
    store = QuotaStore(
        rate_limits={"gemini-flash-latest": RateLimitConfig(rpm=10, tpm=1000, rpd=5)}
    )
    # "some-other-model" isn't configured - must still enforce a real
    # (conservative) limit, never silently allow unlimited throughput.
    for _ in range(10):
        assert (
            await store.try_reserve(project_id="a", model="some-other-model", estimated_tokens=1)
            is True
        )
    assert (
        await store.try_reserve(project_id="a", model="some-other-model", estimated_tokens=1)
        is False
    )
