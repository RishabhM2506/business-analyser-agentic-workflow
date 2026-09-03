"""Unit tests for `app.gemini_scheduler.health` -- circuit-breaker state
machine (CLOSED -> OPEN -> HALF_OPEN -> CLOSED / OPEN again), daily-quota
exhaustion tracking, and credential-scoped auth-failure isolation."""

from __future__ import annotations

import pytest

from app.gemini_scheduler.errors import GeminiErrorClass
from app.gemini_scheduler.health import CircuitState, HealthStore


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _store(**overrides: object) -> tuple[HealthStore, _FakeClock]:
    clock = _FakeClock()
    defaults: dict[str, object] = {
        "consecutive_failure_threshold": 3,
        "open_duration_seconds": 10.0,
        "max_open_duration_seconds": 100.0,
        "daily_quota_cooldown_seconds": 3600.0,
        "credential_cooldown_seconds": 300.0,
        "now_fn": clock,
    }
    defaults.update(overrides)
    return HealthStore(**defaults), clock  # type: ignore[arg-type]


@pytest.mark.unit
async def test_starts_closed_and_eligible() -> None:
    store, _ = _store()
    assert await store.try_acquire("proj-a", "model-x") is True
    snapshot = await store.snapshot("proj-a", "model-x")
    assert snapshot.circuit_state == CircuitState.CLOSED


@pytest.mark.unit
async def test_trips_open_after_consecutive_failure_threshold() -> None:
    store, _ = _store(consecutive_failure_threshold=3)
    for _ in range(3):
        await store.record_failure(
            project_id="proj-a",
            model="model-x",
            credential_id="cred-1",
            error_class=GeminiErrorClass.SERVER_OVERLOADED,
        )
    snapshot = await store.snapshot("proj-a", "model-x")
    assert snapshot.circuit_state == CircuitState.OPEN
    assert await store.try_acquire("proj-a", "model-x") is False


@pytest.mark.unit
async def test_a_success_before_threshold_resets_consecutive_count() -> None:
    store, _ = _store(consecutive_failure_threshold=3)
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.SERVER_OVERLOADED,
    )
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.SERVER_OVERLOADED,
    )
    await store.record_success("proj-a", "model-x")
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.SERVER_OVERLOADED,
    )
    # Only 1 consecutive failure since the reset - still CLOSED.
    snapshot = await store.snapshot("proj-a", "model-x")
    assert snapshot.circuit_state == CircuitState.CLOSED


@pytest.mark.unit
async def test_open_transitions_to_half_open_after_cooldown_and_grants_one_trial() -> None:
    store, clock = _store(consecutive_failure_threshold=1, open_duration_seconds=10.0)
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.SERVER_OVERLOADED,
    )
    assert await store.try_acquire("proj-a", "model-x") is False  # still OPEN, cooldown not elapsed

    clock.advance(10.0)

    assert await store.try_acquire("proj-a", "model-x") is True  # HALF_OPEN trial granted
    # A second concurrent attempt must not also get the trial slot.
    assert await store.try_acquire("proj-a", "model-x") is False


@pytest.mark.unit
async def test_half_open_trial_success_closes_the_circuit() -> None:
    store, clock = _store(consecutive_failure_threshold=1, open_duration_seconds=10.0)
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.SERVER_OVERLOADED,
    )
    clock.advance(10.0)
    assert await store.try_acquire("proj-a", "model-x") is True

    await store.record_success("proj-a", "model-x")

    snapshot = await store.snapshot("proj-a", "model-x")
    assert snapshot.circuit_state == CircuitState.CLOSED
    assert await store.try_acquire("proj-a", "model-x") is True


@pytest.mark.unit
async def test_half_open_trial_failure_reopens_with_extended_cooldown() -> None:
    store, clock = _store(consecutive_failure_threshold=1, open_duration_seconds=10.0)
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.SERVER_OVERLOADED,
    )
    clock.advance(10.0)
    assert await store.try_acquire("proj-a", "model-x") is True

    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.SERVER_OVERLOADED,
    )
    snapshot = await store.snapshot("proj-a", "model-x")
    assert snapshot.circuit_state == CircuitState.OPEN

    # Extended (2nd open) cooldown is 20s (10 * 2^1), not the original 10s.
    clock.advance(10.0)
    assert await store.try_acquire("proj-a", "model-x") is False
    clock.advance(10.0)
    assert await store.try_acquire("proj-a", "model-x") is True


@pytest.mark.unit
async def test_open_duration_is_capped_at_max() -> None:
    store, clock = _store(
        consecutive_failure_threshold=1, open_duration_seconds=10.0, max_open_duration_seconds=15.0
    )
    for _ in range(4):
        await store.record_failure(
            project_id="proj-a",
            model="model-x",
            credential_id="cred-1",
            error_class=GeminiErrorClass.SERVER_OVERLOADED,
        )
        clock.advance(15.0)  # always enough to clear the capped duration
        await store.try_acquire("proj-a", "model-x")
    snapshot_before = await store.snapshot("proj-a", "model-x")
    assert snapshot_before.circuit_state == CircuitState.HALF_OPEN
    # Fail one more time from HALF_OPEN and confirm the cooldown never
    # exceeds the configured cap even as open_count keeps growing.
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.SERVER_OVERLOADED,
    )
    clock.advance(15.0)
    assert await store.try_acquire("proj-a", "model-x") is True


@pytest.mark.unit
async def test_daily_quota_exhaustion_is_tracked_per_project_not_per_model() -> None:
    store, _clock = _store(daily_quota_cooldown_seconds=3600.0)
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.DAILY_QUOTA_EXHAUSTED,
    )
    assert await store.is_daily_exhausted("proj-a") is True
    # A different model under the same project is also exhausted (RPD is
    # project-wide, not per-model).
    snapshot = await store.snapshot("proj-a", "model-y")
    assert snapshot.daily_exhausted is True


@pytest.mark.unit
async def test_daily_quota_exhaustion_does_not_trip_the_circuit_breaker() -> None:
    """Distinct mechanisms (spec §9) - a daily-exhausted project's circuit
    stays CLOSED; it's excluded from routing via is_daily_exhausted/
    snapshot().daily_exhausted, not via the breaker."""
    store, _ = _store()
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.DAILY_QUOTA_EXHAUSTED,
    )
    snapshot = await store.snapshot("proj-a", "model-x")
    assert snapshot.circuit_state == CircuitState.CLOSED


@pytest.mark.unit
async def test_daily_quota_exhaustion_auto_restores_after_cooldown() -> None:
    store, clock = _store(daily_quota_cooldown_seconds=3600.0)
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.DAILY_QUOTA_EXHAUSTED,
    )
    clock.advance(3600.0)
    assert await store.is_daily_exhausted("proj-a") is False


@pytest.mark.unit
async def test_authentication_failure_disables_only_that_credential() -> None:
    store, _ = _store()
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-bad",
        error_class=GeminiErrorClass.AUTHENTICATION_FAILED,
    )
    assert await store.is_credential_healthy("cred-bad") is False
    # A sibling credential in the same project is unaffected.
    assert await store.is_credential_healthy("cred-good") is True
    snapshot = await store.snapshot("proj-a", "model-x")
    assert snapshot.circuit_state == CircuitState.CLOSED  # no project-level penalty for a bare 401


@pytest.mark.unit
async def test_authentication_failure_auto_recovers_after_cooldown() -> None:
    store, clock = _store(credential_cooldown_seconds=300.0)
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-bad",
        error_class=GeminiErrorClass.AUTHENTICATION_FAILED,
    )
    clock.advance(300.0)
    assert await store.is_credential_healthy("cred-bad") is True


@pytest.mark.unit
async def test_permission_denied_trips_circuit_immediately_and_disables_credential() -> None:
    store, _ = _store(consecutive_failure_threshold=10)  # would NOT trip via the gradual path
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.PERMISSION_DENIED,
    )
    snapshot = await store.snapshot("proj-a", "model-x")
    assert snapshot.circuit_state == CircuitState.OPEN
    assert await store.is_credential_healthy("cred-1") is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "error_class",
    [
        GeminiErrorClass.BAD_REQUEST,
        GeminiErrorClass.NOT_FOUND,
        GeminiErrorClass.SCHEMA_VALIDATION_FAILED,
        GeminiErrorClass.CANCELLED,
        GeminiErrorClass.SAFETY_BLOCKED,
    ],
)
async def test_no_penalty_classes_leave_state_untouched(error_class: GeminiErrorClass) -> None:
    store, _ = _store(consecutive_failure_threshold=1)
    await store.record_failure(
        project_id="proj-a", model="model-x", credential_id="cred-1", error_class=error_class
    )
    snapshot = await store.snapshot("proj-a", "model-x")
    assert snapshot.circuit_state == CircuitState.CLOSED
    assert await store.is_credential_healthy("cred-1") is True


@pytest.mark.unit
async def test_health_score_degrades_and_recovers_with_failure_then_success() -> None:
    store, _ = _store(consecutive_failure_threshold=100)  # never trips, isolates the score itself
    before = await store.snapshot("proj-a", "model-x")
    for _ in range(10):
        await store.record_failure(
            project_id="proj-a",
            model="model-x",
            credential_id="cred-1",
            error_class=GeminiErrorClass.SERVER_OVERLOADED,
        )
    degraded = await store.snapshot("proj-a", "model-x")
    assert degraded.health_score < before.health_score

    for _ in range(10):
        await store.record_success("proj-a", "model-x")
    recovered = await store.snapshot("proj-a", "model-x")
    assert recovered.health_score > degraded.health_score


@pytest.mark.unit
async def test_cancel_acquire_frees_a_half_open_trial_without_affecting_health() -> None:
    store, clock = _store(consecutive_failure_threshold=1, open_duration_seconds=10.0)
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.SERVER_OVERLOADED,
    )
    clock.advance(10.0)
    assert await store.try_acquire("proj-a", "model-x") is True
    before = await store.snapshot("proj-a", "model-x")

    await store.cancel_acquire("proj-a", "model-x")

    after = await store.snapshot("proj-a", "model-x")
    assert after.circuit_state == before.circuit_state == CircuitState.HALF_OPEN
    assert after.health_score == before.health_score
    # The trial slot is free again for a different attempt to claim.
    assert await store.try_acquire("proj-a", "model-x") is True


@pytest.mark.unit
async def test_cancel_acquire_on_a_closed_circuit_is_a_safe_no_op() -> None:
    store, _ = _store()
    await store.cancel_acquire("proj-a", "model-x")
    assert await store.try_acquire("proj-a", "model-x") is True


@pytest.mark.unit
async def test_half_open_health_score_is_penalized_relative_to_closed() -> None:
    store, clock = _store(consecutive_failure_threshold=1, open_duration_seconds=10.0)
    await store.record_failure(
        project_id="proj-a",
        model="model-x",
        credential_id="cred-1",
        error_class=GeminiErrorClass.SERVER_OVERLOADED,
    )
    clock.advance(10.0)
    half_open = await store.snapshot("proj-a", "model-x")
    assert half_open.circuit_state == CircuitState.HALF_OPEN

    await store.record_success("proj-a", "model-x")
    closed = await store.snapshot("proj-a", "model-x")
    assert closed.health_score > half_open.health_score
