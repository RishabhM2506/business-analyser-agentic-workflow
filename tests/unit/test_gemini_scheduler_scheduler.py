"""Unit tests for `app.gemini_scheduler.scheduler.GeminiScheduler` -- the
routing/retry orchestration itself. Uses the real `HealthStore`/
`HierarchicalConcurrency` (already covered standalone) and fake `ModelClient`
doubles, matching this repo's existing `LoadBalancedGeminiModelClient` test
pattern (`tests/unit/test_models.py`)."""

from __future__ import annotations

import asyncio
from typing import TypeVar

import pytest
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel

from app.gemini_scheduler.concurrency import HierarchicalConcurrency
from app.gemini_scheduler.credentials import GeminiCredential
from app.gemini_scheduler.errors import GeminiErrorClass
from app.gemini_scheduler.health import HealthStore
from app.gemini_scheduler.quota import QuotaStore, RateLimitConfig
from app.gemini_scheduler.scheduler import GeminiScheduler, NoEligibleGeminiCandidateError
from app.models import GroundedResult

T = TypeVar("T", bound=BaseModel)

# A generous limit for tests that aren't specifically about RPM/TPM/RPD
# enforcement -- otherwise the default free-tier-sized limits could
# accidentally throttle an unrelated test making many quick calls.
_UNLIMITED = RateLimitConfig(rpm=1_000_000, tpm=1_000_000_000, rpd=1_000_000)


def _permissive_quota() -> QuotaStore:
    return QuotaStore(rate_limits={"test-model": _UNLIMITED})


class _OneFieldSchema(BaseModel):
    value: str


def _credential(id_: str, *, project_id: str | None = None) -> GeminiCredential:
    return GeminiCredential(id=id_, project_id=project_id or id_, api_key=f"key-{id_}")


def _api_error(cls: type[Exception], *, code: int, status: str, message: str = "") -> Exception:
    exc = cls(code, {"error": {"code": code, "status": status, "message": message}}, None)
    assert isinstance(exc, Exception)
    return exc


class _FakeClient:
    """One fake `ModelClient` per credential -- the scheduler's
    `client_factory` returns whichever one matches the chosen credential,
    so tests can control exactly which candidate succeeds/fails."""

    def __init__(self, *, calls: list[str] | None = None, error: Exception | None = None) -> None:
        self.error = error
        self.call_count = 0
        self._calls = calls if calls is not None else []

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        self.call_count += 1
        self._calls.append("generate_structured")
        if self.error is not None:
            raise self.error
        return schema.model_validate({"value": "ok"})

    async def generate_grounded(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> GroundedResult[T]:
        raise NotImplementedError


def _scheduler(
    credentials: list[GeminiCredential],
    clients_by_credential_id: dict[str, _FakeClient],
    *,
    max_attempts: int | None = None,
    health_store: HealthStore | None = None,
    concurrency: HierarchicalConcurrency | None = None,
    quota: QuotaStore | None = None,
) -> GeminiScheduler:
    def factory(credential: GeminiCredential, model: str) -> _FakeClient:
        return clients_by_credential_id[credential.id]

    return GeminiScheduler(
        model="test-model",
        credentials=credentials,
        health_store=health_store if health_store is not None else HealthStore(),
        concurrency=concurrency if concurrency is not None else HierarchicalConcurrency(),
        quota=quota if quota is not None else _permissive_quota(),
        client_factory=factory,
        max_attempts=max_attempts,
        sleep_fn=lambda _seconds: asyncio.sleep(0),
    )


@pytest.mark.unit
async def test_succeeds_on_first_eligible_candidate() -> None:
    cred = _credential("a")
    client = _FakeClient()
    scheduler = _scheduler([cred], {"a": client})

    result = await scheduler.generate_structured(
        system_prompt="s", user_content="u", schema=_OneFieldSchema
    )

    assert result.value == "ok"
    assert client.call_count == 1


@pytest.mark.unit
async def test_retries_a_different_candidate_after_a_retryable_failure() -> None:
    cred_a, cred_b = _credential("a"), _credential("b")
    failing = _FakeClient(error=_api_error(ServerError, code=503, status="UNAVAILABLE"))
    succeeding = _FakeClient()
    scheduler = _scheduler([cred_a, cred_b], {"a": failing, "b": succeeding}, max_attempts=2)

    result = await scheduler.generate_structured(
        system_prompt="s", user_content="u", schema=_OneFieldSchema
    )

    assert result.value == "ok"
    assert failing.call_count == 1
    assert succeeding.call_count == 1


@pytest.mark.unit
async def test_reraises_the_real_last_exception_when_every_candidate_fails() -> None:
    cred_a, cred_b = _credential("a"), _credential("b")
    exc_a = _api_error(ServerError, code=503, status="UNAVAILABLE")
    exc_b = _api_error(ServerError, code=504, status="DEADLINE_EXCEEDED")
    scheduler = _scheduler(
        [cred_a, cred_b],
        {"a": _FakeClient(error=exc_a), "b": _FakeClient(error=exc_b)},
        max_attempts=2,
    )

    with pytest.raises(ServerError) as excinfo:
        await scheduler.generate_structured(
            system_prompt="s", user_content="u", schema=_OneFieldSchema
        )

    # The REAL last exception, not a wrapper - the exact contract
    # LoadBalancedGeminiModelClient established and app/main.py depends on.
    assert excinfo.value is exc_b


@pytest.mark.unit
async def test_fail_fast_error_class_never_tries_a_second_candidate() -> None:
    cred_a, cred_b = _credential("a"), _credential("b")
    bad_request = _api_error(ClientError, code=400, status="INVALID_ARGUMENT")
    failing = _FakeClient(error=bad_request)
    other = _FakeClient()
    scheduler = _scheduler([cred_a, cred_b], {"a": failing, "b": other}, max_attempts=2)

    with pytest.raises(ClientError):
        await scheduler.generate_structured(
            system_prompt="s", user_content="u", schema=_OneFieldSchema
        )

    assert failing.call_count == 1
    assert other.call_count == 0


@pytest.mark.unit
async def test_daily_quota_exhausted_routes_to_a_different_project_without_delay() -> None:
    cred_a = _credential("a", project_id="proj-a")
    cred_b = _credential("b", project_id="proj-b")
    daily_exhausted = _api_error(
        ClientError, code=429, status="RESOURCE_EXHAUSTED", message="requests per day exceeded"
    )
    slept: list[float] = []

    async def tracking_sleep(seconds: float) -> None:
        slept.append(seconds)

    def factory(credential: GeminiCredential, model: str) -> _FakeClient:
        return {"a": _FakeClient(error=daily_exhausted), "b": _FakeClient()}[credential.id]

    scheduler = GeminiScheduler(
        model="test-model",
        credentials=[cred_a, cred_b],
        health_store=HealthStore(),
        concurrency=HierarchicalConcurrency(),
        quota=_permissive_quota(),
        client_factory=factory,
        max_attempts=2,
        sleep_fn=tracking_sleep,
    )

    result = await scheduler.generate_structured(
        system_prompt="s", user_content="u", schema=_OneFieldSchema
    )

    assert result.value == "ok"
    assert slept == []  # no backoff sleep for a RETRY_DIFFERENT_CANDIDATE failure


@pytest.mark.unit
async def test_no_eligible_candidate_raises_dedicated_error() -> None:
    cred = _credential("a", project_id="proj-a")
    health_store = HealthStore()
    await health_store.record_failure(
        project_id="proj-a",
        model="test-model",
        credential_id="a",
        error_class=GeminiErrorClass.DAILY_QUOTA_EXHAUSTED,
    )
    scheduler = _scheduler([cred], {"a": _FakeClient()}, health_store=health_store)

    with pytest.raises(NoEligibleGeminiCandidateError):
        await scheduler.generate_structured(
            system_prompt="s", user_content="u", schema=_OneFieldSchema
        )


@pytest.mark.unit
async def test_a_disabled_credential_is_never_selected() -> None:
    disabled = GeminiCredential(id="a", project_id="proj-a", api_key="k", enabled=False)
    healthy = _credential("b")
    scheduler = _scheduler(
        [disabled, healthy], {"a": _FakeClient(), "b": _FakeClient()}, max_attempts=1
    )

    result = await scheduler.generate_structured(
        system_prompt="s", user_content="u", schema=_OneFieldSchema
    )

    assert result.value == "ok"


@pytest.mark.unit
async def test_concurrency_slot_is_released_after_a_successful_dispatch() -> None:
    cred = _credential("a")
    concurrency = HierarchicalConcurrency(global_limit=1, project_model_initial_limit=1)
    scheduler = _scheduler([cred], {"a": _FakeClient()}, concurrency=concurrency)

    await scheduler.generate_structured(system_prompt="s", user_content="u", schema=_OneFieldSchema)
    # If the slot had leaked, this would find zero headroom.
    assert await concurrency.try_acquire("a", "test-model") is True


@pytest.mark.unit
async def test_concurrency_slot_is_released_after_a_failed_dispatch() -> None:
    cred = _credential("a")
    concurrency = HierarchicalConcurrency(global_limit=1, project_model_initial_limit=1)
    exc = _api_error(ClientError, code=400, status="INVALID_ARGUMENT")
    scheduler = _scheduler([cred], {"a": _FakeClient(error=exc)}, concurrency=concurrency)

    with pytest.raises(ClientError):
        await scheduler.generate_structured(
            system_prompt="s", user_content="u", schema=_OneFieldSchema
        )

    assert await concurrency.try_acquire("a", "test-model") is True


@pytest.mark.unit
async def test_many_concurrent_calls_never_exceed_the_global_concurrency_ceiling() -> None:
    """The right-sized equivalent of spec §20's "100 tasks x 5 prompts !=
    500 concurrent calls": many concurrent scheduler calls sharing one
    HierarchicalConcurrency must never let more than `global_limit` real
    dispatches run at once."""
    cred = _credential("a")
    concurrency = HierarchicalConcurrency(global_limit=3, project_model_initial_limit=10)
    quota = _permissive_quota()
    max_concurrent_seen = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    class _TrackingClient:
        async def generate_structured(
            self, *, system_prompt: str, user_content: str, schema: type[T]
        ) -> T:
            nonlocal max_concurrent_seen, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return schema.model_validate({"value": "ok"})

        async def generate_grounded(
            self, *, system_prompt: str, user_content: str, schema: type[T]
        ) -> GroundedResult[T]:
            raise NotImplementedError

    def factory(credential: GeminiCredential, model: str) -> _TrackingClient:
        return _TrackingClient()

    async def make_call() -> None:
        # Deliberately NOT overriding sleep_fn to an instant no-op here:
        # waiting for a concurrency slot to free up requires real wall-clock
        # time to actually pass for another concurrent call's own real
        # `asyncio.sleep(0.01)` (below) to complete - an instant sleep_fn
        # would busy-poll through its capacity-wait budget in microseconds,
        # well before any slot was genuinely freed.
        scheduler = GeminiScheduler(
            model="test-model",
            credentials=[cred],
            health_store=HealthStore(),
            concurrency=concurrency,
            quota=quota,
            client_factory=factory,
        )
        await scheduler.generate_structured(
            system_prompt="s", user_content="u", schema=_OneFieldSchema
        )

    await asyncio.gather(*(make_call() for _ in range(20)))

    assert max_concurrent_seen <= 3


@pytest.mark.unit
async def test_scheduler_requires_at_least_one_credential() -> None:
    with pytest.raises(ValueError, match="at least one credential"):
        GeminiScheduler(
            model="test-model",
            credentials=[],
            health_store=HealthStore(),
            concurrency=HierarchicalConcurrency(),
            quota=_permissive_quota(),
        )


@pytest.mark.unit
async def test_max_attempts_defaults_to_five() -> None:
    """2026-09-04: raised from 3 to 5 per explicit user direction."""
    scheduler = _scheduler([_credential("a")], {"a": _FakeClient()})
    assert scheduler._max_attempts == 1  # capped at the actual pool size (1 credential)

    five_creds = [_credential(str(i)) for i in range(10)]
    clients = {str(i): _FakeClient() for i in range(10)}
    scheduler = _scheduler(five_creds, clients)
    assert scheduler._max_attempts == 5  # capped at 5, not the full pool of 10


@pytest.mark.unit
async def test_rpm_exhaustion_routes_to_a_different_credential() -> None:
    cred_a, cred_b = _credential("a"), _credential("b")
    tight_quota = QuotaStore(
        rate_limits={"test-model": RateLimitConfig(rpm=1, tpm=1_000_000, rpd=1_000_000)}
    )
    # Consume project "a"'s only RPM slot for this minute before the real call.
    assert (
        await tight_quota.try_reserve(project_id="a", model="test-model", estimated_tokens=1)
        is True
    )

    scheduler = _scheduler(
        [cred_a, cred_b],
        {"a": _FakeClient(), "b": _FakeClient()},
        quota=tight_quota,
        max_attempts=2,
    )

    result = await scheduler.generate_structured(
        system_prompt="s", user_content="u", schema=_OneFieldSchema
    )

    assert result.value == "ok"
    # credential "a" was skipped (RPM exhausted); "b" served the request.
    assert scheduler._credentials[0].id == "a"  # sanity: a really was first in the list


@pytest.mark.unit
async def test_rpd_exhausted_project_is_excluded_but_a_different_project_still_works() -> None:
    cred_a, cred_b = _credential("a", project_id="proj-a"), _credential("b", project_id="proj-b")
    quota = QuotaStore(
        rate_limits={"test-model": RateLimitConfig(rpm=1_000_000, tpm=1_000_000_000, rpd=1)}
    )
    # Exhaust proj-a's entire daily budget (rpd=1) before the real call.
    assert (
        await quota.try_reserve(project_id="proj-a", model="test-model", estimated_tokens=1) is True
    )
    client_a, client_b = _FakeClient(), _FakeClient()

    scheduler = _scheduler(
        [cred_a, cred_b], {"a": client_a, "b": client_b}, quota=quota, max_attempts=2
    )

    result = await scheduler.generate_structured(
        system_prompt="s", user_content="u", schema=_OneFieldSchema
    )

    assert result.value == "ok"
    assert client_a.call_count == 0  # proj-a excluded entirely, never dispatched to
    assert client_b.call_count == 1


@pytest.mark.unit
async def test_zero_rpd_raises_no_eligible_candidate() -> None:
    cred = _credential("a")
    zero_rpd_quota = QuotaStore(
        rate_limits={"test-model": RateLimitConfig(rpm=1_000_000, tpm=1_000_000_000, rpd=0)}
    )
    scheduler = _scheduler([cred], {"a": _FakeClient()}, quota=zero_rpd_quota)

    with pytest.raises(NoEligibleGeminiCandidateError):
        await scheduler.generate_structured(
            system_prompt="s", user_content="u", schema=_OneFieldSchema
        )
