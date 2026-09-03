"""Per-credential/per-(project, model) health and circuit-breaker state for
the Gemini Provider Scheduler (Phase 3) -- in-memory, `asyncio.Lock`-guarded,
matching `app/budget.py`'s `BudgetTracker` shape exactly (this repo's own
precedent for single-process, per-session hot state; see that module's
docstring for why in-memory is a real, load-bearing choice at v1's
"runs one instance" scale, not a placeholder).

Three separate, independently-tracked scopes (never one global boolean --
spec §6):

- **Credential scope** (`is_credential_healthy`/`_credential_cooldown_until`):
  authentication failures. A 401 on one credential must not penalize a
  *different* credential sharing the same `project_id` -- exactly the
  correctness gap a per-project-only breaker would have.
- **Project+model scope** (`HealthState`, the circuit breaker proper):
  429/500/503/504/409-aborted and a strong 403 penalty. CLOSED -> OPEN
  (after enough consecutive failures, or immediately for a 403) -> HALF_OPEN
  (after a cooldown, exactly one trial request, atomically claimed via
  `try_acquire`) -> CLOSED on success, or OPEN again (extended, capped
  cooldown) on failure. Never permanently open -- every OPEN circuit
  eventually gets a HALF_OPEN trial.
- **Daily quota scope** (`is_daily_exhausted`/`_project_daily_exhausted_until`):
  tracked per-project (Gemini's daily quota is a project-level ceiling, not
  per-model), independent of the circuit breaker -- the spec's own explicit
  distinction (§9): a project that's merely rate-limited should keep
  getting HALF_OPEN trials; one that's hit its daily ceiling should not be
  retried at all until the ceiling resets.

**Honest limitation**: Google's exact per-project daily-reset time isn't
independently confirmed for these accounts (no real daily-exhaustion
response was captured live this session -- see `app.gemini_scheduler.
errors`'s own docstring for the same caveat on *detecting* daily exhaustion
in the first place). Rather than guess a specific reset timezone, this
defaults to a conservative fixed cooldown from the moment exhaustion was
detected, configurable via `Settings` if a real reset time is ever
confirmed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import structlog

from app.gemini_scheduler.errors import GeminiErrorClass

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

DEFAULT_CONSECUTIVE_FAILURE_THRESHOLD = 3
DEFAULT_OPEN_DURATION_SECONDS = 15.0
DEFAULT_MAX_OPEN_DURATION_SECONDS = 300.0
DEFAULT_DAILY_QUOTA_COOLDOWN_SECONDS = 24 * 60 * 60.0
DEFAULT_CREDENTIAL_COOLDOWN_SECONDS = 60 * 60.0
_EWMA_ALPHA = 0.2

# No project/model/credential state change at all -- matches the spec's own
# table (400/404: "No" penalty everywhere; a schema-validation failure,
# cancellation, or safety block is about the *request*, not the provider's
# health).
_NO_PENALTY_CLASSES = frozenset(
    {
        GeminiErrorClass.BAD_REQUEST,
        GeminiErrorClass.NOT_FOUND,
        GeminiErrorClass.SCHEMA_VALIDATION_FAILED,
        GeminiErrorClass.CANCELLED,
        GeminiErrorClass.SAFETY_BLOCKED,
    }
)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class HealthState:
    project_id: str
    model: str
    circuit_state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    open_count: int = 0
    cooldown_until: float | None = None  # monotonic seconds
    # Optimistic prior (1.0): a never-seen candidate isn't penalized just
    # for being new -- it competes on equal footing with a proven-healthy
    # one until real signal accumulates.
    success_ewma: float = 1.0
    half_open_trial_in_flight: bool = False


@dataclass
class HealthSnapshot:
    """Read-only view for scoring/eligibility -- callers must not mutate
    this; state changes only happen through `HealthStore`'s own methods."""

    circuit_state: CircuitState
    health_score: float
    daily_exhausted: bool


class HealthStore:
    def __init__(
        self,
        *,
        consecutive_failure_threshold: int = DEFAULT_CONSECUTIVE_FAILURE_THRESHOLD,
        open_duration_seconds: float = DEFAULT_OPEN_DURATION_SECONDS,
        max_open_duration_seconds: float = DEFAULT_MAX_OPEN_DURATION_SECONDS,
        daily_quota_cooldown_seconds: float = DEFAULT_DAILY_QUOTA_COOLDOWN_SECONDS,
        credential_cooldown_seconds: float = DEFAULT_CREDENTIAL_COOLDOWN_SECONDS,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._states: dict[tuple[str, str], HealthState] = {}
        self._project_daily_exhausted_until: dict[str, float] = {}
        self._credential_cooldown_until: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._consecutive_failure_threshold = consecutive_failure_threshold
        self._open_duration_seconds = open_duration_seconds
        self._max_open_duration_seconds = max_open_duration_seconds
        self._daily_quota_cooldown_seconds = daily_quota_cooldown_seconds
        self._credential_cooldown_seconds = credential_cooldown_seconds
        self._now = now_fn

    def _get_or_create(self, project_id: str, model: str) -> HealthState:
        key = (project_id, model)
        state = self._states.get(key)
        if state is None:
            state = HealthState(project_id=project_id, model=model)
            self._states[key] = state
        return state

    async def is_credential_healthy(self, credential_id: str) -> bool:
        async with self._lock:
            until = self._credential_cooldown_until.get(credential_id)
            return until is None or self._now() >= until

    async def is_daily_exhausted(self, project_id: str) -> bool:
        async with self._lock:
            until = self._project_daily_exhausted_until.get(project_id)
            return until is not None and self._now() < until

    async def snapshot(self, project_id: str, model: str) -> HealthSnapshot:
        async with self._lock:
            state = self._get_or_create(project_id, model)
            self._maybe_expire_cooldown(state)
            until = self._project_daily_exhausted_until.get(project_id)
            daily_exhausted = until is not None and self._now() < until
            score = state.success_ewma
            if state.circuit_state == CircuitState.HALF_OPEN:
                score *= 0.5  # a real option, but a riskier bet than CLOSED
            return HealthSnapshot(
                circuit_state=state.circuit_state,
                health_score=score,
                daily_exhausted=daily_exhausted,
            )

    async def try_acquire(self, project_id: str, model: str) -> bool:
        """Atomically checks the circuit breaker and, for an OPEN circuit
        whose cooldown has elapsed, claims the single HALF_OPEN trial slot.
        Returns `True` iff the caller may dispatch a real request against
        this `(project_id, model)` right now. On `True`, the caller MUST
        call `record_success` or `record_failure` exactly once when the
        dispatch resolves (matches this app's existing `async with
        semaphore:`-style "release always happens" discipline -- see
        `app.gemini_scheduler.concurrency`)."""
        async with self._lock:
            state = self._get_or_create(project_id, model)
            self._maybe_expire_cooldown(state)
            if state.circuit_state == CircuitState.OPEN:
                return False
            if state.circuit_state == CircuitState.HALF_OPEN:
                if state.half_open_trial_in_flight:
                    return False
                state.half_open_trial_in_flight = True
                return True
            return True  # CLOSED

    async def cancel_acquire(self, project_id: str, model: str) -> None:
        """Releases a HALF_OPEN trial slot claimed by `try_acquire` without
        ever actually dispatching a request against it (e.g. the scheduler
        picked this candidate but lost the concurrency-layer race right
        after) -- clears `half_open_trial_in_flight` only, with **no**
        effect on `consecutive_failures`/`success_ewma`/`circuit_state`,
        since nothing about the provider's real health was observed. Safe
        to call even when nothing was reserved (a CLOSED circuit's
        `try_acquire` never sets `half_open_trial_in_flight` in the first
        place)."""
        async with self._lock:
            self._get_or_create(project_id, model).half_open_trial_in_flight = False

    def _maybe_expire_cooldown(self, state: HealthState) -> None:
        if (
            state.circuit_state == CircuitState.OPEN
            and state.cooldown_until is not None
            and self._now() >= state.cooldown_until
        ):
            state.circuit_state = CircuitState.HALF_OPEN
            state.cooldown_until = None
            logger.info(
                "gemini_scheduler.circuit_half_open",
                project_id=state.project_id,
                model=state.model,
            )

    async def record_success(self, project_id: str, model: str) -> None:
        async with self._lock:
            state = self._get_or_create(project_id, model)
            state.consecutive_failures = 0
            state.consecutive_successes += 1
            state.success_ewma += _EWMA_ALPHA * (1.0 - state.success_ewma)
            state.half_open_trial_in_flight = False
            if state.circuit_state in (CircuitState.OPEN, CircuitState.HALF_OPEN):
                state.circuit_state = CircuitState.CLOSED
                state.open_count = 0
                state.cooldown_until = None
                logger.info("gemini_scheduler.circuit_closed", project_id=project_id, model=model)

    async def record_failure(
        self, *, project_id: str, model: str, credential_id: str, error_class: GeminiErrorClass
    ) -> None:
        async with self._lock:
            if error_class in _NO_PENALTY_CLASSES:
                return
            if error_class == GeminiErrorClass.DAILY_QUOTA_EXHAUSTED:
                self._project_daily_exhausted_until[project_id] = (
                    self._now() + self._daily_quota_cooldown_seconds
                )
                logger.warning(
                    "gemini_scheduler.daily_quota_exhausted",
                    project_id=project_id,
                    cooldown_seconds=self._daily_quota_cooldown_seconds,
                )
                return
            if error_class == GeminiErrorClass.AUTHENTICATION_FAILED:
                # Credential-scoped only (spec: "Optional" project penalty
                # for 401 -- an invalid/revoked key says nothing about the
                # project's own health, so a sibling credential in the same
                # project stays fully eligible).
                self._credential_cooldown_until[credential_id] = (
                    self._now() + self._credential_cooldown_seconds
                )
                logger.warning(
                    "gemini_scheduler.credential_disabled",
                    credential_id=credential_id,
                    error_class=error_class,
                )
                return

            state = self._get_or_create(project_id, model)
            state.half_open_trial_in_flight = False
            state.consecutive_successes = 0
            state.success_ewma += _EWMA_ALPHA * (0.0 - state.success_ewma)

            if error_class == GeminiErrorClass.PERMISSION_DENIED:
                # Strong on both scopes (spec's table) -- trips the circuit
                # immediately rather than waiting for the usual consecutive
                # threshold.
                self._credential_cooldown_until[credential_id] = (
                    self._now() + self._credential_cooldown_seconds
                )
                logger.warning(
                    "gemini_scheduler.credential_disabled",
                    credential_id=credential_id,
                    error_class=error_class,
                )
                self._trip_circuit(state)
                return

            state.consecutive_failures += 1
            if (
                state.circuit_state == CircuitState.HALF_OPEN
                or state.consecutive_failures >= self._consecutive_failure_threshold
            ):
                self._trip_circuit(state)

    def _trip_circuit(self, state: HealthState) -> None:
        state.open_count += 1
        state.circuit_state = CircuitState.OPEN
        duration = min(
            self._max_open_duration_seconds,
            self._open_duration_seconds * (2 ** (state.open_count - 1)),
        )
        state.cooldown_until = self._now() + duration
        logger.warning(
            "gemini_scheduler.circuit_opened",
            project_id=state.project_id,
            model=state.model,
            open_count=state.open_count,
            cooldown_seconds=duration,
        )


_health_store: HealthStore | None = None


def get_health_store() -> HealthStore:
    """Process-wide singleton, matching `app.budget.get_budget_tracker`'s
    own construct-on-first-use pattern."""
    global _health_store
    if _health_store is None:
        _health_store = HealthStore()
    return _health_store
