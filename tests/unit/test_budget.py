"""Unit tests for `app.budget.BudgetTracker` — per-thread and per-(tenant,
day) model-call ceilings, fail-closed (docs/PLAN.md §5.5). Pure in-memory
state, no I/O — `unit`, not `integration`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.budget import (
    _SWEEP_INTERVAL_CALLS,
    _THREAD_IDLE_TTL,
    BudgetExceededError,
    BudgetTracker,
    get_budget_tracker,
)


@pytest.mark.unit
async def test_check_and_increment_allows_calls_under_both_ceilings() -> None:
    tracker = BudgetTracker(max_calls_per_thread=2, max_calls_per_day=10)
    await tracker.check_and_increment(thread_id="t-1", tenant_id="default")
    await tracker.check_and_increment(thread_id="t-1", tenant_id="default")  # exactly at ceiling


@pytest.mark.unit
async def test_check_and_increment_raises_when_thread_ceiling_would_be_exceeded() -> None:
    tracker = BudgetTracker(max_calls_per_thread=2, max_calls_per_day=10)
    await tracker.check_and_increment(thread_id="t-1", tenant_id="default")
    await tracker.check_and_increment(thread_id="t-1", tenant_id="default")

    with pytest.raises(BudgetExceededError):
        await tracker.check_and_increment(thread_id="t-1", tenant_id="default")


@pytest.mark.unit
async def test_thread_ceiling_is_scoped_per_thread_not_global() -> None:
    tracker = BudgetTracker(max_calls_per_thread=1, max_calls_per_day=10)
    await tracker.check_and_increment(thread_id="t-1", tenant_id="default")
    # A different thread has its own, independent counter.
    await tracker.check_and_increment(thread_id="t-2", tenant_id="default")


@pytest.mark.unit
async def test_check_and_increment_raises_when_day_ceiling_would_be_exceeded() -> None:
    tracker = BudgetTracker(max_calls_per_thread=100, max_calls_per_day=2)
    await tracker.check_and_increment(thread_id="t-1", tenant_id="default")
    await tracker.check_and_increment(thread_id="t-2", tenant_id="default")

    with pytest.raises(BudgetExceededError):
        await tracker.check_and_increment(thread_id="t-3", tenant_id="default")


@pytest.mark.unit
async def test_day_ceiling_is_scoped_per_tenant() -> None:
    # max_calls_per_day=2 here (not 1): the tenant-scoped counter and the
    # C1 global backstop (below) share this one ceiling value, so this
    # needs headroom for tenant-a (1) + tenant-b (1) = 2 real calls to both
    # succeed without the *global* check tripping first and masking what
    # this test is actually verifying (that tenant-a and tenant-b each
    # have their own independent per-tenant counter).
    tracker = BudgetTracker(max_calls_per_thread=100, max_calls_per_day=2)
    await tracker.check_and_increment(thread_id="t-1", tenant_id="tenant-a")
    # A different tenant has its own, independent daily counter.
    await tracker.check_and_increment(thread_id="t-2", tenant_id="tenant-b")

    with pytest.raises(BudgetExceededError):
        await tracker.check_and_increment(thread_id="t-3", tenant_id="tenant-a")


@pytest.mark.unit
async def test_global_day_ceiling_cannot_be_bypassed_by_rotating_tenant_id() -> None:
    """Finding C1 (architect review, 2026-08-20): tenant_id is a free-form,
    client-supplied string on an unauthenticated endpoint. Before this
    fix, an adversarial client defeated the day ceiling for free by
    minting a fresh tenant_id per request -- each fresh key started its
    own counter at zero. Reproduces the review's own repro sketch: many
    distinct tenants, each making exactly one call, must still trip the
    ceiling once the deployment-wide total is reached."""
    tracker = BudgetTracker(max_calls_per_thread=100, max_calls_per_day=3)
    await tracker.check_and_increment(thread_id="t-1", tenant_id="tenant-0")
    await tracker.check_and_increment(thread_id="t-2", tenant_id="tenant-1")
    await tracker.check_and_increment(thread_id="t-3", tenant_id="tenant-2")

    # A 4th distinct, never-before-seen tenant_id still gets blocked --
    # the per-tenant counter alone would show 0/3 for this brand-new
    # tenant and allow it; the global backstop is what actually stops it.
    with pytest.raises(BudgetExceededError):
        await tracker.check_and_increment(thread_id="t-4", tenant_id="tenant-3")


@pytest.mark.unit
async def test_global_day_ceiling_resets_on_a_new_day() -> None:
    days = iter(
        [datetime(2026, 8, 16, 23, 59, tzinfo=UTC), datetime(2026, 8, 17, 0, 1, tzinfo=UTC)]
    )
    tracker = BudgetTracker(
        max_calls_per_thread=100, max_calls_per_day=1, now_fn=lambda: next(days)
    )
    await tracker.check_and_increment(thread_id="t-1", tenant_id="tenant-a")  # 2026-08-16: 1/1
    # A brand-new tenant, but a new day too -- the global counter must
    # reset with the date, exactly like the per-tenant one already does.
    await tracker.check_and_increment(thread_id="t-2", tenant_id="tenant-b")  # 2026-08-17: fresh


@pytest.mark.unit
async def test_day_ceiling_resets_on_a_new_day() -> None:
    days = iter(
        [datetime(2026, 8, 16, 23, 59, tzinfo=UTC), datetime(2026, 8, 17, 0, 1, tzinfo=UTC)]
    )
    tracker = BudgetTracker(
        max_calls_per_thread=100, max_calls_per_day=1, now_fn=lambda: next(days)
    )
    await tracker.check_and_increment(thread_id="t-1", tenant_id="default")  # 2026-08-16: 1/1
    await tracker.check_and_increment(thread_id="t-2", tenant_id="default")  # 2026-08-17: fresh


@pytest.mark.unit
async def test_failed_check_does_not_partially_increment_either_counter() -> None:
    """A call that would breach the *day* ceiling must not still bump the
    *thread* counter (and vice versa) — fails closed with no partial
    spend recorded (docs/PLAN.md §5.5)."""
    tracker = BudgetTracker(max_calls_per_thread=100, max_calls_per_day=0)
    with pytest.raises(BudgetExceededError):
        await tracker.check_and_increment(thread_id="t-1", tenant_id="default")

    # If the thread counter had been bumped anyway, raising the ceiling
    # would still show 1 call already spent; assert it's genuinely 0 by
    # allowing exactly `max_calls_per_thread` after raising the day ceiling.
    tracker2 = BudgetTracker(max_calls_per_thread=1, max_calls_per_day=0)
    with pytest.raises(BudgetExceededError):
        await tracker2.check_and_increment(thread_id="t-2", tenant_id="default")
    with pytest.raises(BudgetExceededError):
        # Still the day ceiling that trips, not "thread ceiling now at 1/1
        # already used" -- proves no partial increment happened above.
        await tracker2.check_and_increment(thread_id="t-2", tenant_id="default")


# --- idle-entry eviction (architect-review finding, 2026-08-26) ---------------
#
# `_thread_calls` used to grow forever — every distinct thread_id ever seen
# (a free-form, client-supplied string on an unauthenticated endpoint, per
# this class's own C1 finding) added a permanent entry for the life of the
# process. These tests drive the sweep directly via `now_fn`, never real
# wall-clock time.


@pytest.mark.unit
async def test_sweep_evicts_a_thread_idle_past_the_ttl() -> None:
    start = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
    tracker = BudgetTracker(max_calls_per_thread=5, max_calls_per_day=10_000, now_fn=lambda: start)
    await tracker.check_and_increment(thread_id="t-1", tenant_id="default")
    assert "t-1" in tracker._thread_calls

    later = start + _THREAD_IDLE_TTL + timedelta(seconds=1)
    tracker._now_fn = lambda: later
    for i in range(_SWEEP_INTERVAL_CALLS):
        await tracker.check_and_increment(thread_id=f"other-thread-{i}", tenant_id="default")

    assert "t-1" not in tracker._thread_calls
    assert "t-1" not in tracker._thread_last_seen


@pytest.mark.unit
async def test_sweep_does_not_evict_a_recently_touched_thread() -> None:
    start = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
    tracker = BudgetTracker(max_calls_per_thread=5, max_calls_per_day=10_000, now_fn=lambda: start)
    await tracker.check_and_increment(thread_id="t-1", tenant_id="default")

    soon = start + timedelta(minutes=1)  # well under the idle TTL
    tracker._now_fn = lambda: soon
    for i in range(_SWEEP_INTERVAL_CALLS):
        await tracker.check_and_increment(thread_id=f"other-thread-{i}", tenant_id="default")

    assert "t-1" in tracker._thread_calls  # still tracked - was not idle long enough


@pytest.mark.unit
async def test_a_rejected_call_still_counts_as_activity_for_eviction_purposes() -> None:
    """A thread actively hammering its own ceiling (every call rejected)
    must not have its counter evicted mid-stream - that would let a client
    reset its own per-thread ceiling early by waiting out the idle TTL
    while still nominally "using" the same thread_id."""
    start = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
    tracker = BudgetTracker(max_calls_per_thread=1, max_calls_per_day=10_000, now_fn=lambda: start)
    await tracker.check_and_increment(thread_id="t-1", tenant_id="default")  # exhausts the 1 slot

    later = start + _THREAD_IDLE_TTL + timedelta(seconds=1)
    tracker._now_fn = lambda: later
    with pytest.raises(BudgetExceededError):
        await tracker.check_and_increment(thread_id="t-1", tenant_id="default")  # still rejected

    assert tracker._thread_calls["t-1"] == 1  # counter intact, never silently reset


@pytest.mark.unit
async def test_sweep_evicts_stale_day_call_entries_once_the_day_has_passed() -> None:
    yesterday = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
    tracker = BudgetTracker(
        max_calls_per_thread=100, max_calls_per_day=10_000, now_fn=lambda: yesterday
    )
    await tracker.check_and_increment(thread_id="t-1", tenant_id="tenant-a")
    assert ("tenant-a", yesterday.date()) in tracker._day_calls

    today = yesterday + timedelta(days=1)
    tracker._now_fn = lambda: today
    for i in range(_SWEEP_INTERVAL_CALLS):
        await tracker.check_and_increment(thread_id=f"other-thread-{i}", tenant_id="tenant-b")

    assert ("tenant-a", yesterday.date()) not in tracker._day_calls


@pytest.mark.unit
def test_get_budget_tracker_returns_a_process_wide_singleton() -> None:
    assert get_budget_tracker() is get_budget_tracker()


@pytest.mark.unit
def test_get_budget_tracker_uses_configured_ceilings(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.budget as budget_module
    from app.settings import get_settings

    monkeypatch.setattr(budget_module, "_budget_tracker_singleton", None)
    get_settings.cache_clear()
    try:
        tracker = get_budget_tracker()
        assert tracker._max_calls_per_thread == get_settings().max_model_calls_per_thread
        assert tracker._max_calls_per_day == get_settings().max_model_calls_per_day
    finally:
        monkeypatch.setattr(budget_module, "_budget_tracker_singleton", None)
        get_settings.cache_clear()
