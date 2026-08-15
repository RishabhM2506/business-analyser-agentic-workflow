"""Per-session and per-day model-call budget ceilings (docs/PLAN.md §5.5,
master brief §7.6). Fails closed: exceeding any ceiling must produce
`ErrorResponse(error_code="BUDGET_EXCEEDED")`, never a silent continuation.

# TODO(Phase 3): back these counters with real per-thread/per-day
# persistence (likely the same datastore as the checkpointer,
# `settings.database_url`) instead of an in-memory placeholder.
"""

from __future__ import annotations


class BudgetExceededError(Exception):
    """Raised when a model-call ceiling (per-thread or per-day) would be
    breached by the call about to be made."""


class BudgetTracker:
    """Tracks model-call counts per thread and per day against configured
    ceilings (`settings.max_model_calls_per_thread`,
    `settings.max_model_calls_per_day`)."""

    def __init__(self, *, max_calls_per_thread: int, max_calls_per_day: int) -> None:
        self._max_calls_per_thread = max_calls_per_thread
        self._max_calls_per_day = max_calls_per_day

    async def check_and_increment(self, *, thread_id: str, tenant_id: str) -> None:
        """Raise `BudgetExceededError` if incrementing would breach either ceiling."""
        raise NotImplementedError  # TODO(Phase 3): implement counters + persistence.
