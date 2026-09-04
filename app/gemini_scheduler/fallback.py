"""Cross-model fallback for the Gemini Provider Scheduler (2026-09-04
addition) -- tries an ordered list of `ModelClient`s, each usually a
`GeminiScheduler` configured for a *different real model*, when the
primary model's own credential pool is genuinely capacity-exhausted (not
merely one bad response).

**Why this is real, additional capacity, not just another roll of the
dice**: each real Gemini model version is its own separate RPM/TPM/RPD
pool -- confirmed directly from the user's own AI Studio dashboard
(2026-09-04): "Gemini 3.7 Flash" showed `21/20` RPD, *already exceeded*,
while "Gemini 2.5 Flash" sat at `0/20`, completely unused, on the exact
same project at the exact same time. A different, currently-idle model
version genuinely has headroom the exhausted primary doesn't.

**Deliberately narrow trigger** (the spec's own guardrail: "do not
silently change model if doing so could materially affect output
quality"): falls back ONLY on capacity-exhaustion signals --
`NoEligibleGeminiCandidateError` (nothing eligible for the primary model
at all: every credential's project is daily-exhausted/RPD-capped/circuit-
open for *this specific model*) or a real dispatch failure whose
classified `RetryAction` is NOT `FAIL_FAST` (a provider/capacity problem,
not a request-shape one). A `FAIL_FAST` failure (bad request, schema
validation, safety block) propagates immediately, never triggering a
model change -- a different model line isn't expected to fix a malformed
request, and silently retrying one on a different model changes what was
actually generated, exactly the risk this guardrail exists to avoid.

Only ever swaps within the tier `Settings.gemini_model_fallbacks` was
configured for (flash-tier fallbacks for the analysis role, lite-tier for
utility) -- `get_model_for_role` (`app/models.py`) is the only real
construction site and deliberately never cross-wires the two."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

import structlog
from pydantic import BaseModel

from app.gemini_scheduler.errors import RetryAction, classify_error, retry_action_for
from app.gemini_scheduler.scheduler import NoEligibleGeminiCandidateError

if TYPE_CHECKING:
    from app.models import GroundedResult, ModelClient

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


class ModelFallbackClient:
    """`ModelClient` wrapping an ordered list of `ModelClient`s -- the
    first is the primary/preferred model; the rest are fallback models,
    tried in order only when the previous one is capacity-exhausted (see
    this module's own docstring for the exact trigger)."""

    def __init__(self, clients: list[ModelClient]) -> None:
        if not clients:
            raise ValueError("ModelFallbackClient requires at least one client")
        self._clients = clients

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
        last_exc: Exception | None = None
        for index, client in enumerate(self._clients):
            try:
                return await call(client)
            except NoEligibleGeminiCandidateError as exc:
                last_exc = exc
                logger.warning(
                    "gemini_scheduler.model_fallback_no_eligible_candidate",
                    client_index=index,
                    remaining_fallbacks=len(self._clients) - index - 1,
                )
                continue
            except Exception as exc:
                if retry_action_for(classify_error(exc)) == RetryAction.FAIL_FAST:
                    raise
                last_exc = exc
                logger.warning(
                    "gemini_scheduler.model_fallback_exhausted",
                    client_index=index,
                    remaining_fallbacks=len(self._clients) - index - 1,
                    error_type=type(exc).__name__,
                )
                continue
        assert last_exc is not None
        raise last_exc
