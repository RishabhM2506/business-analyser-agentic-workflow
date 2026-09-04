"""Proactive RPM/TPM/RPD admission control for the Gemini Provider
Scheduler (2026-09-04 addition) -- complements, does not replace,
`health.py`'s *reactive* daily-quota-exhaustion tracking (that module
reacts to Gemini's own real `429` daily-exhaustion response; this module
proactively stops a request from ever being sent once a *configured* cap
is reached, so a real `429` from Google should be rare in practice, not
the primary mechanism).

Token bucket implementation generalizes `app.rate_limit.RateLimiter`'s
established pattern in this codebase (continuous refill, not a fixed
window -- a fixed window unfairly bursts/denies right at a boundary) to
consume a variable amount per call, since TPM consumes N estimated tokens
per call, not always 1 the way RPM does.

**On the default numbers below**: Google's own rate-limits page
(`ai.google.dev/gemini-api/docs/rate-limits`, fetched live 2026-09-04) does
**not** publish fixed free-tier RPM/TPM/RPD numbers -- it states "Rate
limits depend on... your usage tier" and directs users to their own live
AI Studio dashboard (`aistudio.google.com/rate-limit`) instead. That same
page *does* directly confirm two facts used elsewhere in this package:
rate limits apply **per project, not per API key** (validates
`credentials.py`'s project-grouping design), and RPD quotas reset at
**midnight Pacific time** (used by `_next_pacific_midnight` below).

The numbers in `DEFAULT_RATE_LIMITS` were **updated 2026-09-04 to the
real values**, read directly from a live AI Studio dashboard screenshot
the user provided for their own project ("Default Gemini Project"), cross-
checked against real model IDs via a quota-free `client.models.list()`
call (see that constant's own comment for the full sourcing and the one
remaining inference: which real model each of Google's `-latest` aliases
currently resolves to). These replace an earlier best-effort estimate that
turned out to be too generous for the analysis-role model (guessed RPD 250,
real RPD 20) -- a reminder that a guess, however conservative-seeming, is
still a guess; prefer real dashboard numbers whenever they're available.
Still worth re-checking periodically: Google can repoint a `-latest` alias
to a different underlying model at any time, and this project's own usage
tier/plan can change -- override via `Settings.gemini_rate_limits` (env var
`GEMINI_RATE_LIMITS`) whenever either happens.

**Honest limitation on TPM specifically**: `estimate_tokens` is a rough
`len(text) // 4` character-based approximation (a standard rough estimate
for English prose), not Gemini's real tokenizer, which isn't exposed
pre-call. `GeminiModelClient.generate_structured` uses `with_structured_output`
without `include_raw=True`, so the real `AIMessage.usage_metadata` token
counts (confirmed present on the installed `langchain-google-genai`'s
response objects) aren't currently surfaced to correct this estimate
after the fact -- doing so would mean changing `generate_structured`'s
return contract at every existing call site, out of proportion for this
addition. TPM admission is therefore approximate by construction; RPM and
RPD are exact (they only ever count real requests).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import structlog
from pydantic import BaseModel

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

_PACIFIC = ZoneInfo("America/Los_Angeles")


class RateLimitConfig(BaseModel, frozen=True):
    """A plain pydantic model (not a dataclass) so it embeds directly in
    `Settings.gemini_rate_limits` without a conversion layer -- same
    convention as `app.settings.GeminiCredentialConfig`."""

    rpm: int
    tpm: int
    rpd: int


# REAL numbers (2026-09-04), not the earlier best-effort estimate -- read
# directly from the user's own AI Studio "Rate limits by model" dashboard
# for their actual project ("Default Gemini Project"), confirmed against
# real model IDs via a live, quota-free `client.models.list()` call (the
# dashboard shows display names like "Gemini 3.8 Flash"; the API needs the
# real ID, e.g. `gemini-3.8-flash`). `gemini-flash-latest`/
# `gemini-flash-lite-latest` are Google's own aliases -- the dashboard has
# no separate row for an alias, only its currently-resolved underlying
# model (`gemini-3.7-flash`/`gemini-3.8-flash` and `gemini-3.5-flash-lite`
# respectively, both confirmed by real nonzero usage in the same
# dashboard), so the alias entries below use that resolved model's real
# numbers as a working assumption -- re-check if Google repoints either
# alias to a model with a different quota shape.
#
# The non-alias entries below (idle on this account as of 2026-09-04) exist
# so `app.gemini_scheduler.fallback` can route to a *different model*, not
# just a different credential, when the primary alias's entire pool is
# genuinely capacity-exhausted -- each real model version is its own
# separate RPD/RPM/TPM pool, confirmed directly by the dashboard (e.g.
# "Gemini 3.7 Flash" showing 21/20 RPD *exceeded* while "Gemini 2.5 Flash"
# sits at 0/20, completely unused, on the very same project).
#
# An unrecognized model name not listed here falls back to
# `_FALLBACK_RATE_LIMIT` (the more conservative of the two primary
# entries), never an unlimited allowance.
DEFAULT_RATE_LIMITS: dict[str, RateLimitConfig] = {
    "gemini-flash-latest": RateLimitConfig(rpm=5, tpm=250_000, rpd=20),
    "gemini-flash-lite-latest": RateLimitConfig(rpm=15, tpm=250_000, rpd=500),
    # Idle same-tier fallback candidates for the analysis role (matches
    # gemini-flash-latest's real 5/250K/20 shape on this account).
    "gemini-2.5-flash": RateLimitConfig(rpm=5, tpm=250_000, rpd=20),
    "gemini-3-flash-preview": RateLimitConfig(rpm=5, tpm=250_000, rpd=20),
    "gemini-3.5-flash": RateLimitConfig(rpm=5, tpm=250_000, rpd=20),
    "gemini-3.6-flash": RateLimitConfig(rpm=5, tpm=250_000, rpd=20),
    "gemini-3.7-flash": RateLimitConfig(rpm=5, tpm=250_000, rpd=20),
    "gemini-3.8-flash": RateLimitConfig(rpm=5, tpm=250_000, rpd=20),
    # Idle fallback candidates for the utility role. Real numbers differ
    # meaningfully between these two -- confirmed directly from the
    # dashboard, not assumed to match gemini-flash-lite-latest's shape.
    "gemini-2.5-flash-lite": RateLimitConfig(rpm=10, tpm=250_000, rpd=20),
    "gemini-3.1-flash-lite": RateLimitConfig(rpm=15, tpm=250_000, rpd=500),
    "gemini-3.5-flash-lite": RateLimitConfig(rpm=15, tpm=250_000, rpd=500),
}
_FALLBACK_RATE_LIMIT = RateLimitConfig(rpm=5, tpm=250_000, rpd=20)

_IDLE_EVICTION_SECONDS = 300.0
_SWEEP_INTERVAL_CALLS = 500


def estimate_tokens(text: str) -> int:
    """Rough ~4-chars-per-token approximation -- see this module's own
    "Honest limitation on TPM specifically" docstring section. Never zero,
    so even an empty/near-empty prompt still reserves a nonzero amount
    rather than looking free."""
    return max(1, len(text) // 4)


class _TokenBucket:
    """Continuous-refill token bucket keyed by an arbitrary string --
    generalizes `app.rate_limit.RateLimiter` (which always consumes
    exactly 1.0 per call) to consume a caller-supplied amount, shared here
    by both the RPM bucket (amount=1) and the TPM bucket
    (amount=estimated_tokens)."""

    def __init__(
        self, *, capacity: float, refill_per_second: float, now_fn: Callable[[], float]
    ) -> None:
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._now_fn = now_fn
        self._tokens: dict[str, float] = {}
        self._last_refill: dict[str, float] = {}

    def _refill(self, key: str, now: float) -> float:
        tokens = self._tokens.get(key, self._capacity)
        last_refill = self._last_refill.get(key, now)
        elapsed = max(0.0, now - last_refill)
        tokens = min(self._capacity, tokens + elapsed * self._refill_per_second)
        self._last_refill[key] = now
        return tokens

    def headroom(self, key: str) -> float:
        return self._refill(key, self._now_fn())

    def try_consume(self, key: str, amount: float) -> bool:
        now = self._now_fn()
        tokens = self._refill(key, now)
        if tokens < amount:
            self._tokens[key] = tokens
            return False
        self._tokens[key] = tokens - amount
        return True

    def evict_idle(self, *, cutoff: float) -> None:
        idle = [key for key, last in self._last_refill.items() if last < cutoff]
        for key in idle:
            self._tokens.pop(key, None)
            self._last_refill.pop(key, None)


def _next_pacific_midnight(now_utc: datetime) -> datetime:
    """The real, confirmed reset behavior (`ai.google.dev/gemini-api/docs/
    rate-limits`: "Requests per day (RPD) quotas reset at midnight Pacific
    time") -- not a rolling 24h window."""
    now_pacific = now_utc.astimezone(_PACIFIC)
    next_midnight_pacific = (now_pacific + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return next_midnight_pacific.astimezone(UTC)


class QuotaStore:
    """Per-`(project_id, model)` RPM/TPM admission, per-`project_id` RPD
    admission (RPD, like the daily-quota tracking in `health.py`, is
    project-wide -- Google's own confirmed behavior, not per-model)."""

    def __init__(
        self,
        *,
        rate_limits: dict[str, RateLimitConfig] | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        wall_clock_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._rate_limits = rate_limits if rate_limits is not None else DEFAULT_RATE_LIMITS
        self._now = now_fn
        self._wall_clock = wall_clock_fn
        self._rpm_buckets: dict[str, _TokenBucket] = {}
        self._tpm_buckets: dict[str, _TokenBucket] = {}
        self._rpd_counts: dict[str, int] = {}
        self._rpd_reset_at: dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        self._calls_since_sweep = 0

    def _limits_for(self, model: str) -> RateLimitConfig:
        return self._rate_limits.get(model, _FALLBACK_RATE_LIMIT)

    def _rpm_bucket(self, project_id: str, model: str) -> _TokenBucket:
        key = f"{project_id}:{model}"
        bucket = self._rpm_buckets.get(key)
        if bucket is None:
            limits = self._limits_for(model)
            bucket = _TokenBucket(
                capacity=float(limits.rpm), refill_per_second=limits.rpm / 60.0, now_fn=self._now
            )
            self._rpm_buckets[key] = bucket
        return bucket

    def _tpm_bucket(self, project_id: str, model: str) -> _TokenBucket:
        key = f"{project_id}:{model}"
        bucket = self._tpm_buckets.get(key)
        if bucket is None:
            limits = self._limits_for(model)
            bucket = _TokenBucket(
                capacity=float(limits.tpm), refill_per_second=limits.tpm / 60.0, now_fn=self._now
            )
            self._tpm_buckets[key] = bucket
        return bucket

    def _rpd_remaining_locked(self, project_id: str, model: str) -> int:
        now = self._wall_clock()
        reset_at = self._rpd_reset_at.get(project_id)
        if reset_at is None or now >= reset_at:
            self._rpd_counts[project_id] = 0
            self._rpd_reset_at[project_id] = _next_pacific_midnight(now)
        limit = self._limits_for(model).rpd
        return max(0, limit - self._rpd_counts.get(project_id, 0))

    async def rpd_would_exceed(self, project_id: str, model: str) -> bool:
        """Peek only -- does not consume. Used as an eligibility filter
        (like `health.is_daily_exhausted`): a project with zero RPD
        headroom left today is excluded from candidate selection entirely,
        not retried with a short wait the way RPM/TPM pressure is."""
        async with self._lock:
            return self._rpd_remaining_locked(project_id, model) <= 0

    async def try_reserve(self, *, project_id: str, model: str, estimated_tokens: int) -> bool:
        """Atomically checks RPD, RPM, and TPM together and reserves all
        three (1 request + `estimated_tokens`) only if every one has
        headroom right now -- never a partial reservation. Returns `False`
        (reserving nothing) if any single one is exhausted."""
        async with self._lock:
            if self._rpd_remaining_locked(project_id, model) <= 0:
                return False
            rpm_bucket = self._rpm_bucket(project_id, model)
            tpm_bucket = self._tpm_bucket(project_id, model)
            if rpm_bucket.headroom(project_id) < 1.0:
                return False
            if tpm_bucket.headroom(project_id) < estimated_tokens:
                return False
            rpm_bucket.try_consume(project_id, 1.0)
            tpm_bucket.try_consume(project_id, float(estimated_tokens))
            self._rpd_counts[project_id] = self._rpd_counts.get(project_id, 0) + 1
            if self._rpd_remaining_locked(project_id, model) <= 0:
                # Logged once, on the transition to zero remaining -- not
                # on every subsequent excluded eligibility check, which
                # would spam a log line per scheduling attempt for the
                # rest of the day.
                logger.warning(
                    "gemini_scheduler.rpd_cap_reached",
                    project_id=project_id,
                    model=model,
                    rpd_limit=self._limits_for(model).rpd,
                )

            self._calls_since_sweep += 1
            if self._calls_since_sweep >= _SWEEP_INTERVAL_CALLS:
                self._calls_since_sweep = 0
                self._sweep_idle(self._now())
            return True

    def _sweep_idle(self, now: float) -> None:
        cutoff = now - _IDLE_EVICTION_SECONDS
        for bucket in (*self._rpm_buckets.values(), *self._tpm_buckets.values()):
            bucket.evict_idle(cutoff=cutoff)


_quota_store: QuotaStore | None = None


def get_quota_store(rate_limits: dict[str, RateLimitConfig] | None = None) -> QuotaStore:
    """Process-wide singleton, matching `health.get_health_store`'s own
    construct-on-first-use pattern. `rate_limits` (normally `Settings.
    gemini_rate_limits`) is only consulted the *first* time this is
    called in the process -- later calls, even with a different
    `rate_limits`, return the already-constructed singleton (the same
    "config read once at first construction" behavior every other
    process-wide store in this package already has)."""
    global _quota_store
    if _quota_store is None:
        _quota_store = QuotaStore(rate_limits=rate_limits)
    return _quota_store
