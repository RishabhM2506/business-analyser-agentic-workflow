"""Per-session and per-day model-call budget ceilings (docs/PLAN.md §5.5,
master brief §7.6). Fails closed: exceeding any ceiling must produce
`ErrorResponse(error_code="BUDGET_EXCEEDED")`, never a silent continuation.

Called from the two model-call nodes (`app/nodes/describe_item.py`,
`app/nodes/summarize.py`) immediately before each real model call — not
pre-checked once at the API edge, and not checked after the call completes.
Checking inside each node (rather than once in `app/main.py` before the
graph even runs) is deliberate: it's the only place a check can guarantee
"no model spend past the ceiling" while still letting the free, deterministic
nodes ahead of it (`validate_query`, `fetch_imports`/`fetch_exports`,
`aggregate`, `retrieve_description`) run normally, and it avoids a second,
redundant increment for a call that hasn't happened yet if this were instead
front-loaded into the endpoint handler.

In-memory counters (module-level singleton, see `get_budget_tracker` below)
— real cross-process/cross-restart persistence is a later concern (the same
datastore as the checkpointer, `settings.database_url`, is the natural
home), explicitly out of v1 scope per this module's original scaffold
docstring. In-memory is still a real, load-bearing ceiling for a single
running instance (docs/PLAN.md §1.2: v1 runs one instance), not a placeholder
that silently no-ops.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

# How long a thread_id can sit untouched before its counter is safe to
# evict entirely (architect-review finding, 2026-08-26: `_thread_calls`
# never shrank — every distinct thread_id ever seen added a permanent
# entry for the life of the process, and `thread_id` is a free-form,
# client-controlled string on an unauthenticated endpoint per this class's
# own C1 finding, so this grows as a side effect of ordinary use, not even
# needing to defeat the day ceiling on purpose). Generously above any
# realistic single browsing session (docs/PLAN.md's per-session framing for
# max_calls_per_thread) — an evicted thread_id's counter simply restarts at
# zero if ever reused, identical to how a never-seen thread_id already
# behaves (`check_and_increment`'s own `.get(thread_id, 0)`), so eviction
# changes no observable behavior for a thread that's genuinely done.
_THREAD_IDLE_TTL = timedelta(hours=2)
# Sweep periodically, not on every call — see `app.rate_limit`'s identical
# reasoning for the same trade-off.
_SWEEP_INTERVAL_CALLS = 500


class BudgetExceededError(Exception):
    """Raised when a model-call ceiling (per-thread or per-day) would be
    breached by the call about to be made."""


class BudgetTracker:
    """Tracks model-call counts per thread, per (tenant, day), and per day
    globally, against configured ceilings (`settings.max_model_calls_per_thread`,
    `settings.max_model_calls_per_day`).

    Per-day counters are scoped by `tenant_id` (not global) so the
    mechanism is already multi-tenant-shaped even though v1 defaults every
    request to `tenant_id="default"` (master brief §3: tenant_id is
    "threaded into... budget accounting" from day one, not bolted on when
    real multi-tenancy ships).

    Finding C1 (architect review, 2026-08-20, filed once real Gemini/Comtrade
    keys went live): `thread_id`/`tenant_id` are both free-form, client-
    supplied strings on an unauthenticated endpoint (`docs/PLAN.md`'s
    already-accepted "no auth in v1" decision) — an adversarial client
    defeats the tenant-scoped day ceiling for free by minting a fresh
    `tenant_id` per request, since each fresh key starts its own counter at
    zero. The tenant-scoped ceiling stays (real value once real
    multi-tenancy ships), but it can no longer be the *only* day-level
    control: `_global_day_calls` enforces the same `max_calls_per_day`
    figure again, keyed on the date alone, so rotating identities no longer
    resets the effective ceiling. Zero behavior change for an honest client
    (v1 has exactly one real tenant, "default" — both checks trip at the
    same count); closes the bypass for an adversarial one.
    """

    def __init__(
        self,
        *,
        max_calls_per_thread: int,
        max_calls_per_day: int,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._max_calls_per_thread = max_calls_per_thread
        self._max_calls_per_day = max_calls_per_day
        self._now_fn = now_fn
        self._thread_calls: dict[str, int] = {}
        self._thread_last_seen: dict[str, datetime] = {}
        self._day_calls: dict[tuple[str, date], int] = {}
        self._global_day_calls: dict[date, int] = {}
        self._lock = asyncio.Lock()
        self._calls_since_sweep = 0

    async def check_and_increment(self, *, thread_id: str, tenant_id: str) -> None:
        """Raise `BudgetExceededError` if incrementing would breach any
        ceiling; otherwise atomically increment all three counters. Checked
        before any counter is mutated, so a call that would breach one
        ceiling never partially increments another — fails closed, no
        partial spend recorded for a call that was refused.
        """
        async with self._lock:
            now = self._now_fn()
            today = now.date()
            day_key = (tenant_id, today)
            thread_count = self._thread_calls.get(thread_id, 0)
            day_count = self._day_calls.get(day_key, 0)
            global_day_count = self._global_day_calls.get(today, 0)

            # Updated unconditionally, before the ceiling checks below — a
            # thread that's actively being rejected for exceeding its own
            # ceiling is still "active" for eviction purposes; touching this
            # only on a successful increment would let a client bypass its
            # own per-thread ceiling by waiting out `_THREAD_IDLE_TTL` while
            # still hammering the same thread_id.
            self._thread_last_seen[thread_id] = now

            self._calls_since_sweep += 1
            if self._calls_since_sweep >= _SWEEP_INTERVAL_CALLS:
                self._calls_since_sweep = 0
                self._sweep_idle_entries(now)

            if thread_count + 1 > self._max_calls_per_thread:
                raise BudgetExceededError(
                    f"thread {thread_id!r} would exceed max_model_calls_per_thread="
                    f"{self._max_calls_per_thread}"
                )
            if day_count + 1 > self._max_calls_per_day:
                raise BudgetExceededError(
                    f"tenant {tenant_id!r} would exceed max_model_calls_per_day="
                    f"{self._max_calls_per_day} for {today.isoformat()}"
                )
            # C1: a global, non-keyed backstop — cannot be reset by minting a
            # fresh tenant_id, unlike the check immediately above.
            if global_day_count + 1 > self._max_calls_per_day:
                raise BudgetExceededError(
                    f"deployment would exceed max_model_calls_per_day="
                    f"{self._max_calls_per_day} for {today.isoformat()} (global ceiling)"
                )

            self._thread_calls[thread_id] = thread_count + 1
            self._day_calls[day_key] = day_count + 1
            self._global_day_calls[today] = global_day_count + 1

    def _sweep_idle_entries(self, now: datetime) -> None:
        """Drop every thread_id untouched for `_THREAD_IDLE_TTL` or longer,
        and every `_day_calls` entry whose day has already passed (a dead
        counter the instant the day rolls over — no TTL needed, just a
        direct comparison). `_global_day_calls` needs no sweep: it has at
        most one entry per calendar day ever seen, already bounded.
        Called with `self._lock` already held."""
        cutoff = now - _THREAD_IDLE_TTL
        idle_threads = [tid for tid, seen in self._thread_last_seen.items() if seen < cutoff]
        for tid in idle_threads:
            self._thread_calls.pop(tid, None)
            self._thread_last_seen.pop(tid, None)

        today = now.date()
        self._day_calls = {key: count for key, count in self._day_calls.items() if key[1] == today}


_budget_tracker_singleton: BudgetTracker | None = None


def get_budget_tracker() -> BudgetTracker:
    """Process-wide singleton, matching `app.cache.tool_cache.get_tool_cache`'s
    pattern — one shared set of counters across every request within this
    running instance (docs/PLAN.md §1.2: v1 runs one instance)."""
    global _budget_tracker_singleton
    if _budget_tracker_singleton is None:
        from app.settings import get_settings

        settings = get_settings()
        _budget_tracker_singleton = BudgetTracker(
            max_calls_per_thread=settings.max_model_calls_per_thread,
            max_calls_per_day=settings.max_model_calls_per_day,
        )
    return _budget_tracker_singleton
