"""Hierarchical adaptive concurrency for the Gemini Provider Scheduler
(Phase 4). Hand-rolled counter + `asyncio.Lock`, not `asyncio.Semaphore` --
a semaphore's limit can't shrink once created, and AIMD (the spec's own
suggested strategy; chosen here as the standard, well-understood approach
for adapting to *observed* capacity signals when the real ceiling is
unknown -- see `app.gemini_scheduler.health`'s own "no real Google quota
numbers to weight against" limitation) needs the limit itself to move up
and down.

Every acquired slot is released via `async with` (see `HierarchicalConcurrency.
acquire` below), so an exception or cancellation always frees it -- Python's
own context-manager guarantee is the single-process equivalent of the
spec's Redis-lease/TTL machinery (built to survive a *different worker
process* crashing after reserving capacity; there is no such gap here,
since if this process crashes, all in-memory state -- including any
"reserved but never released" slot -- resets together).

Hierarchical: a dispatch acquires the GLOBAL limiter, then the PROJECT+MODEL
limiter, both-or-neither (if the second is saturated, the first is
released immediately, not held while nothing can proceed).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

DEFAULT_INITIAL_LIMIT = 4
DEFAULT_MIN_LIMIT = 1
DEFAULT_MAX_LIMIT = 20
DEFAULT_GLOBAL_LIMIT = 20
# How many *distinct* projects must show congestion before the GLOBAL
# ceiling also shrinks (spec §10: "if the majority of projects experience
# 503 ... reduce global/model concurrency" -- a single project's own
# congestion only shrinks that project's own limiter, never global).
DEFAULT_DEGRADED_PROJECT_THRESHOLD = 3


class AdaptiveLimiter:
    """One AIMD-adjusted concurrency ceiling."""

    def __init__(
        self,
        *,
        initial_limit: int = DEFAULT_INITIAL_LIMIT,
        min_limit: int = DEFAULT_MIN_LIMIT,
        max_limit: int = DEFAULT_MAX_LIMIT,
    ) -> None:
        self._limit = initial_limit
        self._min_limit = min_limit
        self._max_limit = max_limit
        self._inflight = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._inflight >= self._limit:
                return False
            self._inflight += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._inflight = max(0, self._inflight - 1)

    async def on_success(self) -> None:
        """Additive increase."""
        async with self._lock:
            self._limit = min(self._max_limit, self._limit + 1)

    async def on_congestion(self) -> None:
        """Multiplicative decrease -- called on `RESOURCE_EXHAUSTED`/
        `SERVER_OVERLOADED` (the spec's own example AIMD-decrease
        triggers)."""
        async with self._lock:
            self._limit = max(self._min_limit, self._limit // 2)

    async def headroom(self) -> int:
        async with self._lock:
            return max(0, self._limit - self._inflight)

    async def current_limit(self) -> int:
        async with self._lock:
            return self._limit


class HierarchicalConcurrency:
    """Global + per-(project, model) concurrency, enforced together on
    every dispatch."""

    def __init__(
        self,
        *,
        global_limit: int = DEFAULT_GLOBAL_LIMIT,
        project_model_initial_limit: int = DEFAULT_INITIAL_LIMIT,
        project_model_min_limit: int = DEFAULT_MIN_LIMIT,
        project_model_max_limit: int = DEFAULT_MAX_LIMIT,
        degraded_project_threshold: int = DEFAULT_DEGRADED_PROJECT_THRESHOLD,
    ) -> None:
        # Global limiter never grows past its configured ceiling via AIMD
        # increase (min==max) -- only `on_congestion` (multiplicative
        # decrease, driven by widespread project degradation) ever moves
        # it, and only downward; there's no real signal that would justify
        # growing *global* capacity beyond what was configured, unlike a
        # per-project/model limiter which legitimately discovers more
        # headroom than its conservative starting point.
        self._global = AdaptiveLimiter(
            initial_limit=global_limit, min_limit=1, max_limit=global_limit
        )
        self._project_model: dict[tuple[str, str], AdaptiveLimiter] = {}
        self._dict_lock = asyncio.Lock()
        self._project_model_initial_limit = project_model_initial_limit
        self._project_model_min_limit = project_model_min_limit
        self._project_model_max_limit = project_model_max_limit
        self._degraded_project_threshold = degraded_project_threshold
        self._recently_overloaded_projects: set[str] = set()

    async def _limiter_for(self, project_id: str, model: str) -> AdaptiveLimiter:
        key = (project_id, model)
        async with self._dict_lock:
            limiter = self._project_model.get(key)
            if limiter is None:
                limiter = AdaptiveLimiter(
                    initial_limit=self._project_model_initial_limit,
                    min_limit=self._project_model_min_limit,
                    max_limit=self._project_model_max_limit,
                )
                self._project_model[key] = limiter
            return limiter

    async def try_acquire(self, project_id: str, model: str) -> bool:
        if not await self._global.try_acquire():
            return False
        limiter = await self._limiter_for(project_id, model)
        if not await limiter.try_acquire():
            await self._global.release()
            return False
        return True

    async def release(self, project_id: str, model: str) -> None:
        limiter = await self._limiter_for(project_id, model)
        await limiter.release()
        await self._global.release()

    async def on_success(self, project_id: str, model: str) -> None:
        limiter = await self._limiter_for(project_id, model)
        await limiter.on_success()
        await self._global.on_success()
        async with self._dict_lock:
            self._recently_overloaded_projects.discard(project_id)

    async def on_congestion(self, project_id: str, model: str) -> None:
        limiter = await self._limiter_for(project_id, model)
        await limiter.on_congestion()
        async with self._dict_lock:
            self._recently_overloaded_projects.add(project_id)
            if len(self._recently_overloaded_projects) >= self._degraded_project_threshold:
                self._recently_overloaded_projects.clear()
                shrink_global = True
            else:
                shrink_global = False
        if shrink_global:
            await self._global.on_congestion()

    async def headroom(self, project_id: str, model: str) -> int:
        return await (await self._limiter_for(project_id, model)).headroom()

    async def project_model_limit(self, project_id: str, model: str) -> int:
        return await (await self._limiter_for(project_id, model)).current_limit()

    async def global_limit(self) -> int:
        return await self._global.current_limit()

    @asynccontextmanager
    async def acquire(self, project_id: str, model: str) -> AsyncIterator[bool]:
        """`async with concurrency.acquire(project_id, model) as acquired:`
        -- `acquired` is `False` if no slot was available (caller must not
        dispatch); the slot (if any) is always released on exit, including
        on an exception or cancellation inside the `with` block."""
        acquired = await self.try_acquire(project_id, model)
        try:
            yield acquired
        finally:
            if acquired:
                await self.release(project_id, model)


_concurrency: HierarchicalConcurrency | None = None


def get_concurrency() -> HierarchicalConcurrency:
    """Process-wide singleton, matching `app.gemini_scheduler.health.
    get_health_store`'s own construct-on-first-use pattern."""
    global _concurrency
    if _concurrency is None:
        _concurrency = HierarchicalConcurrency()
    return _concurrency
