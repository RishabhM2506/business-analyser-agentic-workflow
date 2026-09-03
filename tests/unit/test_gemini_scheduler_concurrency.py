"""Unit tests for `app.gemini_scheduler.concurrency` -- AIMD adjustment,
hierarchical global+project/model enforcement, and a real concurrent-task
race test proving no over-allocation within one process."""

from __future__ import annotations

import asyncio

import pytest

from app.gemini_scheduler.concurrency import AdaptiveLimiter, HierarchicalConcurrency


@pytest.mark.unit
async def test_adaptive_limiter_rejects_once_at_limit() -> None:
    limiter = AdaptiveLimiter(initial_limit=2, min_limit=1, max_limit=10)
    assert await limiter.try_acquire() is True
    assert await limiter.try_acquire() is True
    assert await limiter.try_acquire() is False


@pytest.mark.unit
async def test_adaptive_limiter_release_frees_a_slot() -> None:
    limiter = AdaptiveLimiter(initial_limit=1, min_limit=1, max_limit=10)
    assert await limiter.try_acquire() is True
    assert await limiter.try_acquire() is False
    await limiter.release()
    assert await limiter.try_acquire() is True


@pytest.mark.unit
async def test_additive_increase_on_success() -> None:
    limiter = AdaptiveLimiter(initial_limit=4, min_limit=1, max_limit=10)
    await limiter.on_success()
    assert await limiter.current_limit() == 5


@pytest.mark.unit
async def test_additive_increase_is_capped_at_max() -> None:
    limiter = AdaptiveLimiter(initial_limit=9, min_limit=1, max_limit=10)
    await limiter.on_success()
    await limiter.on_success()
    assert await limiter.current_limit() == 10


@pytest.mark.unit
async def test_multiplicative_decrease_on_congestion() -> None:
    limiter = AdaptiveLimiter(initial_limit=8, min_limit=1, max_limit=20)
    await limiter.on_congestion()
    assert await limiter.current_limit() == 4


@pytest.mark.unit
async def test_multiplicative_decrease_is_floored_at_min() -> None:
    limiter = AdaptiveLimiter(initial_limit=1, min_limit=1, max_limit=20)
    await limiter.on_congestion()
    assert await limiter.current_limit() == 1


@pytest.mark.unit
async def test_headroom_reflects_inflight_vs_limit() -> None:
    limiter = AdaptiveLimiter(initial_limit=5, min_limit=1, max_limit=10)
    await limiter.try_acquire()
    await limiter.try_acquire()
    assert await limiter.headroom() == 3


@pytest.mark.unit
async def test_concurrent_try_acquire_never_over_allocates() -> None:
    """The real correctness property spec §17 asks for, right-sized to a
    single process: many concurrent asyncio tasks racing to acquire the
    same limiter must never push inflight above the configured limit."""
    limiter = AdaptiveLimiter(initial_limit=5, min_limit=1, max_limit=5)
    results: list[bool] = []

    async def attempt() -> None:
        results.append(await limiter.try_acquire())

    await asyncio.gather(*(attempt() for _ in range(50)))

    assert sum(results) == 5


@pytest.mark.unit
async def test_hierarchical_acquire_checks_both_global_and_project_model() -> None:
    concurrency = HierarchicalConcurrency(
        global_limit=1, project_model_initial_limit=5, project_model_max_limit=5
    )
    assert await concurrency.try_acquire("proj-a", "model-x") is True
    # Global limit of 1 is exhausted, even though the project/model limiter
    # itself has headroom.
    assert await concurrency.try_acquire("proj-a", "model-x") is False


@pytest.mark.unit
async def test_hierarchical_release_frees_both_layers() -> None:
    concurrency = HierarchicalConcurrency(
        global_limit=1, project_model_initial_limit=5, project_model_max_limit=5
    )
    await concurrency.try_acquire("proj-a", "model-x")
    await concurrency.release("proj-a", "model-x")
    assert await concurrency.try_acquire("proj-a", "model-x") is True


@pytest.mark.unit
async def test_project_model_saturation_releases_the_already_acquired_global_slot() -> None:
    concurrency = HierarchicalConcurrency(
        global_limit=5, project_model_initial_limit=1, project_model_max_limit=1
    )
    assert await concurrency.try_acquire("proj-a", "model-x") is True
    assert await concurrency.try_acquire("proj-a", "model-x") is False
    # A DIFFERENT project/model pair must still be able to use the global
    # slot that was released back when proj-a/model-x's own limiter was
    # saturated -- proves the global acquisition wasn't leaked.
    assert await concurrency.try_acquire("proj-b", "model-x") is True


@pytest.mark.unit
async def test_acquire_context_manager_always_releases_on_exception() -> None:
    concurrency = HierarchicalConcurrency(
        global_limit=1, project_model_initial_limit=1, project_model_max_limit=1
    )
    with pytest.raises(RuntimeError):
        async with concurrency.acquire("proj-a", "model-x") as acquired:
            assert acquired is True
            raise RuntimeError("simulated dispatch failure")

    # The slot must be free again despite the exception.
    assert await concurrency.try_acquire("proj-a", "model-x") is True


@pytest.mark.unit
async def test_acquire_context_manager_reports_false_when_saturated_without_double_release() -> (
    None
):
    concurrency = HierarchicalConcurrency(
        global_limit=1, project_model_initial_limit=1, project_model_max_limit=1
    )
    async with concurrency.acquire("proj-a", "model-x") as first:
        assert first is True
        async with concurrency.acquire("proj-a", "model-x") as second:
            assert second is False
    # After both context managers exit, exactly one slot is free again (not
    # over-released from the `second` no-op exit).
    assert await concurrency.try_acquire("proj-a", "model-x") is True
    assert await concurrency.try_acquire("proj-a", "model-x") is False


@pytest.mark.unit
async def test_single_project_congestion_does_not_shrink_global() -> None:
    concurrency = HierarchicalConcurrency(global_limit=10, degraded_project_threshold=3)
    await concurrency.on_congestion("proj-a", "model-x")
    assert await concurrency.global_limit() == 10


@pytest.mark.unit
async def test_congestion_across_enough_distinct_projects_shrinks_global() -> None:
    concurrency = HierarchicalConcurrency(global_limit=10, degraded_project_threshold=3)
    await concurrency.on_congestion("proj-a", "model-x")
    await concurrency.on_congestion("proj-b", "model-x")
    await concurrency.on_congestion("proj-c", "model-x")
    assert await concurrency.global_limit() == 5


@pytest.mark.unit
async def test_success_clears_a_projects_overload_flag() -> None:
    concurrency = HierarchicalConcurrency(global_limit=10, degraded_project_threshold=2)
    await concurrency.on_congestion("proj-a", "model-x")
    await concurrency.on_success("proj-a", "model-x")
    # proj-a no longer counts toward the degraded-project threshold.
    await concurrency.on_congestion("proj-b", "model-x")
    assert await concurrency.global_limit() == 10
