"""Gemini error classification (Gemini Provider Scheduler, Phase 1).

Verified directly against the installed `google-genai` SDK, not assumed:
`google.genai.errors.APIError` (base of `ClientError`/`ServerError`, both
confirmed live this session as what `langchain-google-genai` actually
raises unwrapped -- every `error_type` logged during real 503/504 outages
was exactly `"ServerError"`) carries a real `.code` (HTTP status, int),
`.status` (Gemini's own reason string, e.g. `"UNAVAILABLE"`,
`"RESOURCE_EXHAUSTED"`, `"DEADLINE_EXCEEDED"`), and `.message`.
`APIError.raise_error` dispatches 4xx -> `ClientError`, 5xx -> `ServerError`
(read directly from the installed package's source, not documentation).
This module classifies off `.code`/`.status`/`.message` directly rather
than re-deriving them from a caught type name.

**Known, honestly-documented limitation**: distinguishing a temporary rate
limit from daily quota exhaustion (both surface as a `429` with
`.status == "RESOURCE_EXHAUSTED"`) has no reliable structured signal from
Gemini -- only `.message` text. No real daily-exhaustion response was
captured live this session to confirm the exact wording, so
`_looks_like_daily_quota` is best-effort text matching, and defaults to
the safer `RATE_LIMITED` interpretation when it can't tell: under-
classifying as temporary costs a few extra retries (self-correcting via
backoff/the circuit breaker), while over-classifying as daily-exhausted
risks incorrectly blackholing a project that could still serve requests --
the worse failure mode. Same "verified where possible, honestly flagged
where not" convention as `app/models.py`'s `generate_grounded` docstring
already uses for citation extraction.

Similarly, `SAFETY_BLOCKED` detection is best-effort text matching, not
verified against a real blocked response (none was triggered live this
session either).
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Any

import httpx
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError


class GeminiErrorClass(StrEnum):
    """Internal classification every scheduling/health decision is made
    from -- business logic never inspects a raw HTTP status or exception
    type directly (spec requirement: "do not let business logic depend
    directly on raw HTTP status codes")."""

    NONE = "none"
    BAD_REQUEST = "bad_request"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMITED = "rate_limited"
    DAILY_QUOTA_EXHAUSTED = "daily_quota_exhausted"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    SERVER_OVERLOADED = "server_overloaded"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SAFETY_BLOCKED = "safety_blocked"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    UNKNOWN = "unknown"


class RetryAction(StrEnum):
    """What the scheduler should do about one failed attempt -- deliberately
    a 3-way split, not a boolean, because "retryable" alone conflates two
    different real situations (spec §8/§9's table): a durable-for-this-
    candidate failure (401/403/daily quota) where a *different* project
    might work right now with no wait, versus a transient failure (429
    rate limit/503/500/504) worth a backoff pause before retrying either
    the same or a different candidate."""

    FAIL_FAST = "fail_fast"
    RETRY_DIFFERENT_CANDIDATE = "retry_different_candidate"
    RETRY_WITH_BACKOFF = "retry_with_backoff"


_RETRY_POLICY: dict[GeminiErrorClass, RetryAction] = {
    GeminiErrorClass.NONE: RetryAction.FAIL_FAST,
    GeminiErrorClass.BAD_REQUEST: RetryAction.FAIL_FAST,
    GeminiErrorClass.NOT_FOUND: RetryAction.FAIL_FAST,
    GeminiErrorClass.SAFETY_BLOCKED: RetryAction.FAIL_FAST,
    GeminiErrorClass.SCHEMA_VALIDATION_FAILED: RetryAction.FAIL_FAST,
    GeminiErrorClass.CANCELLED: RetryAction.FAIL_FAST,
    GeminiErrorClass.AUTHENTICATION_FAILED: RetryAction.RETRY_DIFFERENT_CANDIDATE,
    GeminiErrorClass.PERMISSION_DENIED: RetryAction.RETRY_DIFFERENT_CANDIDATE,
    GeminiErrorClass.DAILY_QUOTA_EXHAUSTED: RetryAction.RETRY_DIFFERENT_CANDIDATE,
    GeminiErrorClass.CONFLICT: RetryAction.RETRY_WITH_BACKOFF,
    GeminiErrorClass.RATE_LIMITED: RetryAction.RETRY_WITH_BACKOFF,
    GeminiErrorClass.RESOURCE_EXHAUSTED: RetryAction.RETRY_WITH_BACKOFF,
    GeminiErrorClass.SERVER_OVERLOADED: RetryAction.RETRY_WITH_BACKOFF,
    GeminiErrorClass.INTERNAL_SERVER_ERROR: RetryAction.RETRY_WITH_BACKOFF,
    GeminiErrorClass.TIMEOUT: RetryAction.RETRY_WITH_BACKOFF,
    GeminiErrorClass.UNKNOWN: RetryAction.RETRY_WITH_BACKOFF,
}


def retry_action_for(error_class: GeminiErrorClass) -> RetryAction:
    return _RETRY_POLICY[error_class]


_DAILY_QUOTA_MARKERS = ("per day", "perday", "daily")
_SAFETY_MARKERS = ("safety", "blocked", "recitation")


def classify_error(exc: Exception) -> GeminiErrorClass:
    """Classify a real exception raised from a Gemini call. Order matters:
    Python-level concerns (cancellation, schema validation, timeout) are
    checked before attempting to interpret `exc` as a Gemini API error at
    all, since a `TimeoutError`/`ValidationError` is never a
    `google.genai.errors.APIError` instance."""
    if isinstance(exc, asyncio.CancelledError):
        # Never retried -- Python's own cancellation semantics require this
        # propagate immediately, not get caught and retried like an
        # ordinary failure.
        return GeminiErrorClass.CANCELLED
    if isinstance(exc, ValidationError | OutputParserException):
        # Matches `app/main.py`'s existing `isinstance(exc, ValidationError
        # | OutputParserException)` classification exactly -- a genuine
        # schema-validation failure is deterministic given the same
        # request, so a different credential/project can't help.
        return GeminiErrorClass.SCHEMA_VALIDATION_FAILED
    if isinstance(exc, TimeoutError | httpx.TimeoutException):
        # A client-side timeout (our own configured `timeout=` firing with
        # no response at all) -- distinct from a real Gemini-side 504,
        # which arrives as an `APIError` with `.code == 504` and is
        # classified by `_classify_api_error` below instead.
        return GeminiErrorClass.TIMEOUT

    api_error = _as_api_error(exc)
    if api_error is not None:
        return _classify_api_error(api_error)
    return GeminiErrorClass.UNKNOWN


def _as_api_error(exc: Exception) -> Any | None:
    try:
        from google.genai.errors import APIError
    except ImportError:  # pragma: no cover - real dependency, always installed
        return None
    return exc if isinstance(exc, APIError) else None


def _classify_api_error(exc: Any) -> GeminiErrorClass:
    code = exc.code
    status = (exc.status or "").upper()
    message = (exc.message or "").lower()

    if code == 400:
        if any(marker in message for marker in _SAFETY_MARKERS):
            return GeminiErrorClass.SAFETY_BLOCKED
        return GeminiErrorClass.BAD_REQUEST
    if code == 401:
        return GeminiErrorClass.AUTHENTICATION_FAILED
    if code == 403:
        return GeminiErrorClass.PERMISSION_DENIED
    if code == 404:
        return GeminiErrorClass.NOT_FOUND
    if code == 409:
        return GeminiErrorClass.CONFLICT
    if code == 429:
        if any(marker in message for marker in _DAILY_QUOTA_MARKERS):
            return GeminiErrorClass.DAILY_QUOTA_EXHAUSTED
        if status == "RESOURCE_EXHAUSTED":
            return GeminiErrorClass.RATE_LIMITED
        return GeminiErrorClass.RESOURCE_EXHAUSTED
    if code == 499:
        return GeminiErrorClass.CANCELLED
    if code == 500:
        return GeminiErrorClass.INTERNAL_SERVER_ERROR
    if code == 503:
        return GeminiErrorClass.SERVER_OVERLOADED
    if code == 504:
        return GeminiErrorClass.TIMEOUT
    if isinstance(code, int) and 500 <= code < 600:
        return GeminiErrorClass.INTERNAL_SERVER_ERROR
    return GeminiErrorClass.UNKNOWN
