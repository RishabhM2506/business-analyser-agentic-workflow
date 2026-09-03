"""`GeminiScheduler` (Phase 5): the actual `ModelClient` implementation the
rest of the app talks to -- ties together `app.gemini_scheduler.errors`
(classification), `.credentials` (the project-grouped pool), `.health`
(circuit breaker/daily-quota/credential health), and `.concurrency`
(hierarchical AIMD) into one routing/retry loop. Replaces `app.models.
LoadBalancedGeminiModelClient`'s blind per-key round-robin.

**Answers "what capacity is safely available right now?", not "which key
comes next?"** (the spec's own framing): every attempt (1) filters
candidates to ones whose credential is healthy, whose project isn't
daily-exhausted, and whose circuit isn't OPEN; (2) scores the survivors by
health + concurrency headroom, with a small fairness tiebreak so healthy
candidates don't starve each other; (3) atomically claims both the circuit
breaker's slot (a HALF_OPEN circuit permits exactly one concurrent trial)
and a concurrency slot before ever making a real network call.

**Preserves the exact "re-raise the real last exception, never wrap"
contract** `LoadBalancedGeminiModelClient` established (`app/models.py`'s
own docstring explains why: `app/main.py`'s `isinstance(exc, ValidationError
| OutputParserException)` classification, and the two 2026-09-03 fixes in
`app/nodes/summarize.py`/`app/report/narrative.py`, both depend on the real
exception type surviving through every layer of retry)."""

from __future__ import annotations

import asyncio
import itertools
import random
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

import structlog
from pydantic import BaseModel

from app.gemini_scheduler.concurrency import HierarchicalConcurrency
from app.gemini_scheduler.credentials import GeminiCredential
from app.gemini_scheduler.errors import (
    GeminiErrorClass,
    RetryAction,
    classify_error,
    retry_action_for,
)
from app.gemini_scheduler.health import CircuitState, HealthStore

if TYPE_CHECKING:
    from app.models import GroundedResult, ModelClient

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BASE_DELAY_SECONDS = 0.5
_DEFAULT_MAX_DELAY_SECONDS = 20.0
# Concurrency-slot contention resolves as soon as *any* in-flight call
# finishes -- unrelated to provider recovery, so this is a short fixed poll
# interval rather than the exponential-with-jitter curve `_backoff_delay`
# uses for real failures, and bounded separately from `_max_attempts`
# (which limits how many *distinct credentials* are tried, not how many
# times this process waits for a slot on ones that are already eligible).
_DEFAULT_MAX_CAPACITY_WAIT_ATTEMPTS = 20
_CAPACITY_WAIT_DELAY_SECONDS = 0.05
# AIMD-decrease triggers only (spec §13's own example: "repeated 429/503")
# -- distinct from the broader set of classes that count toward tripping
# the circuit breaker (app.gemini_scheduler.health's own, wider set).
_CONGESTION_CLASSES = frozenset(
    {
        GeminiErrorClass.RATE_LIMITED,
        GeminiErrorClass.RESOURCE_EXHAUSTED,
        GeminiErrorClass.SERVER_OVERLOADED,
    }
)


class NoEligibleGeminiCandidateError(Exception):
    """Raised when zero candidates are eligible *before any dispatch is
    even attempted* -- e.g. every credential's project is currently
    daily-exhausted or circuit-OPEN. Distinct from "every attempted
    candidate failed," which re-raises the real last exception from the
    failed attempt instead (there is no real exception to re-raise here,
    since nothing was ever dispatched)."""


def _default_client_factory(credential: GeminiCredential, model: str) -> ModelClient:
    # Local import: `app.models` will import this module too (`get_model_
    # for_role`'s wiring), so a module-level import here would be circular.
    # Matches this file's own dependency's existing local-import style
    # (`app.models.GeminiModelClient.__init__`'s `from langchain_google_
    # genai import ChatGoogleGenerativeAI`), just used to break a cycle
    # instead of defer a heavy import.
    from app.models import GeminiModelClient

    # max_retries=0: the scheduler's own retry loop is the retry mechanism
    # (matches `app.models._POOLED_KEY_MAX_RETRIES`'s identical reasoning
    # for `LoadBalancedGeminiModelClient` -- a per-client internal retry
    # would burn budget retrying the same possibly-broken credential
    # before the scheduler ever gets a chance to route elsewhere).
    return GeminiModelClient(model=model, api_key=credential.api_key, max_retries=0)


class GeminiScheduler:
    """`ModelClient` for one node role's model. Constructed fresh per
    request (matches `app.models`' existing per-request `GeminiModelClient`/
    `LoadBalancedGeminiModelClient` construction pattern), reading/writing
    into the injected, process-wide `health_store`/`concurrency` singletons
    -- the same "cheap object, real state lives in shared stores" shape
    `LoadBalancedGeminiModelClient`'s `start_index_counter` already used
    for its own fairness counter."""

    def __init__(
        self,
        *,
        model: str,
        credentials: list[GeminiCredential],
        health_store: HealthStore,
        concurrency: HierarchicalConcurrency,
        client_factory: Callable[[GeminiCredential, str], ModelClient] | None = None,
        max_attempts: int | None = None,
        base_delay_seconds: float = _DEFAULT_BASE_DELAY_SECONDS,
        max_delay_seconds: float = _DEFAULT_MAX_DELAY_SECONDS,
        fairness_counter: itertools.count[int] | None = None,
        sleep_fn: Callable[[float], Awaitable[object]] = asyncio.sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        if not credentials:
            raise ValueError("GeminiScheduler requires at least one credential")
        self._model = model
        self._credentials = credentials
        self._health = health_store
        self._concurrency = concurrency
        self._client_factory = client_factory or _default_client_factory
        # Bounded, matching `_DEFAULT_MAX_KEY_ATTEMPTS`'s own reasoning in
        # app/models.py: each attempt carries its own real network timeout,
        # so trying every credential in a large pool would multiply
        # worst-case latency for a synchronous HTTP-request caller.
        self._max_attempts = (
            max_attempts
            if max_attempts is not None
            else min(len(credentials), _DEFAULT_MAX_ATTEMPTS)
        )
        self._base_delay = base_delay_seconds
        self._max_delay = max_delay_seconds
        self._fairness_counter = (
            fairness_counter if fairness_counter is not None else itertools.count()
        )
        self._sleep = sleep_fn
        self._random = random_fn

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        return await self._dispatch(
            lambda client: client.generate_structured(
                system_prompt=system_prompt, user_content=user_content, schema=schema
            )
        )

    async def generate_grounded(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> GroundedResult[T]:
        return await self._dispatch(
            lambda client: client.generate_grounded(
                system_prompt=system_prompt, user_content=user_content, schema=schema
            )
        )

    async def _dispatch(self, call: Callable[[ModelClient], Awaitable[R]]) -> R:
        tried: set[str] = set()
        last_exc: Exception | None = None
        last_action: RetryAction | None = None
        capacity_waits = 0
        attempt = 0
        while attempt < self._max_attempts:
            # Only pause when the previous failure was genuinely transient
            # (RETRY_WITH_BACKOFF) -- a durable-for-that-candidate failure
            # (RETRY_DIFFERENT_CANDIDATE: daily quota/401/403) says nothing
            # about whether a *different* project is available right now,
            # so there's no reason to make the caller wait before trying
            # one (spec §9: "429 daily quota -> No until reset," implying
            # move on immediately, not "wait then retry the same thing").
            if attempt > 0 and last_action == RetryAction.RETRY_WITH_BACKOFF:
                await self._sleep(self._backoff_delay(attempt))
            candidate, capacity_pressure = await self._select_and_acquire(exclude=tried)
            if candidate is None:
                if capacity_pressure and capacity_waits < _DEFAULT_MAX_CAPACITY_WAIT_ATTEMPTS:
                    # A candidate was eligible but every concurrency slot
                    # was momentarily full -- this resolves as soon as any
                    # in-flight call (on this or another concurrent
                    # request) finishes, so it's worth a short wait, not a
                    # credential-exhaustion failure. Doesn't consume the
                    # dispatch-attempt budget or add anything to `tried`.
                    capacity_waits += 1
                    await self._sleep(_CAPACITY_WAIT_DELAY_SECONDS)
                    continue
                if last_exc is not None:
                    # Every attempted candidate failed and none remain --
                    # re-raise the real last exception, never a wrapper
                    # (see this module's own docstring for why).
                    raise last_exc
                raise NoEligibleGeminiCandidateError(
                    f"No eligible Gemini credential/project available for model {self._model!r}"
                )
            attempt += 1
            tried.add(candidate.id)
            client = self._client_factory(candidate, self._model)
            try:
                result = await call(client)
            except Exception as exc:
                last_exc = exc
                error_class = classify_error(exc)
                await self._health.record_failure(
                    project_id=candidate.project_id,
                    model=self._model,
                    credential_id=candidate.id,
                    error_class=error_class,
                )
                if error_class in _CONGESTION_CLASSES:
                    await self._concurrency.on_congestion(candidate.project_id, self._model)
                await self._concurrency.release(candidate.project_id, self._model)
                logger.warning(
                    "gemini_scheduler.attempt_failed",
                    model=self._model,
                    project_id=candidate.project_id,
                    credential_id=candidate.id,
                    attempt=attempt,
                    max_attempts=self._max_attempts,
                    error_class=error_class,
                    error_type=type(exc).__name__,
                )
                action = retry_action_for(error_class)
                if action == RetryAction.FAIL_FAST:
                    raise
                last_action = action
                continue
            else:
                await self._health.record_success(candidate.project_id, self._model)
                await self._concurrency.on_success(candidate.project_id, self._model)
                await self._concurrency.release(candidate.project_id, self._model)
                logger.info(
                    "gemini_scheduler.dispatch_succeeded",
                    model=self._model,
                    project_id=candidate.project_id,
                    credential_id=candidate.id,
                    attempt=attempt,
                )
                return result

        logger.error(
            "gemini_scheduler.all_candidates_exhausted",
            model=self._model,
            attempts_made=self._max_attempts,
            candidates_tried=len(tried),
        )
        assert last_exc is not None
        raise last_exc

    async def _select_and_acquire(
        self, *, exclude: set[str]
    ) -> tuple[GeminiCredential | None, bool]:
        """Filters, scores, and atomically claims a candidate's circuit +
        concurrency slot. Returns `(candidate, _)` once both are claimed
        (never a candidate the caller isn't already cleared to dispatch
        against), or `(None, had_capacity_pressure)` when nothing could be
        claimed -- `had_capacity_pressure` distinguishes two different
        reasons for `None`, which `_dispatch` must treat very differently:
        a candidate was health/circuit-eligible but every concurrency slot
        was momentarily full (transient -- worth a short wait, not a
        credential-exhaustion failure), versus nothing was eligible at all
        (durable -- `NoEligibleGeminiCandidateError`/re-raise territory)."""
        eligible: list[GeminiCredential] = []
        for credential in self._credentials:
            if credential.id in exclude:
                continue
            if not credential.supports(self._model):
                continue
            if not await self._health.is_credential_healthy(credential.id):
                continue
            if await self._health.is_daily_exhausted(credential.project_id):
                continue
            eligible.append(credential)
        if not eligible:
            return None, False

        scored: list[tuple[float, int, GeminiCredential]] = []
        for index, credential in enumerate(eligible):
            snapshot = await self._health.snapshot(credential.project_id, self._model)
            if snapshot.circuit_state == CircuitState.OPEN:
                continue
            headroom = await self._concurrency.headroom(credential.project_id, self._model)
            score = snapshot.health_score * 2.0 + min(headroom, 5) * 0.1
            scored.append((score, index, credential))
        if not scored:
            return None, False
        # Primary sort: score descending. Tiebreak: a rotating offset (the
        # same shared-counter fairness trick `app.models.
        # LoadBalancedGeminiModelClient` already established, adapted from
        # "rotate the round-robin start index" to "rotate which
        # equally-scored candidate sorts first") -- computed *once* per
        # call, not per credential, so it rotates *which* tied candidate
        # wins across successive calls instead of monotonically favoring
        # whichever credential happens to be listed last (an earlier,
        # buggier version of this added an ever-increasing counter value
        # directly into the score itself, which is not a tiebreak at all).
        # `count` is precomputed, not `len(scored)` referenced from inside
        # the key function below -- CPython's `list.sort()` temporarily
        # empties the list while computing keys (a documented guard against
        # the key function mutating the list mid-sort), so `len(scored)`
        # evaluated inside the lambda itself would see 0, not the real
        # count (verified live: reproduces a real `ZeroDivisionError`).
        count = len(scored)
        rotation = next(self._fairness_counter) % count
        scored.sort(key=lambda item: (-item[0], (item[1] - rotation) % count))

        capacity_pressure = False
        for _, _, credential in scored:
            circuit_acquired = await self._health.try_acquire(credential.project_id, self._model)
            if not circuit_acquired:
                continue
            concurrency_acquired = await self._concurrency.try_acquire(
                credential.project_id, self._model
            )
            if not concurrency_acquired:
                # Never dispatched -- release the circuit's HALF_OPEN trial
                # claim without recording any outcome (see health.
                # cancel_acquire's own docstring).
                await self._health.cancel_acquire(credential.project_id, self._model)
                capacity_pressure = True
                continue
            return credential, False
        return None, capacity_pressure

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential with jitter (spec §18): `min(max_delay, base *
        2**attempt) + jitter`. Bounded total attempts (see `__init__`)
        keep this app's real constraint -- one synchronous HTTP request
        waiting on the result -- from ballooning past a sane worst case,
        unlike a background job's own, much longer retry-queue horizon."""
        delay: float = min(self._max_delay, self._base_delay * (2**attempt))
        jitter: float = self._random() * delay * 0.5
        return delay + jitter
