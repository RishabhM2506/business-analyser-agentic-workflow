"""Unit tests for `app.gemini_scheduler.errors` -- every HTTP status in the
Gemini Provider Scheduler spec's error table, classified against the real
`google.genai.errors.APIError` shape (not a hand-rolled fake exception),
plus the non-API-error cases (cancellation, schema validation, timeout).
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from google.genai.errors import ClientError, ServerError
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

from app.gemini_scheduler.errors import (
    GeminiErrorClass,
    RetryAction,
    classify_error,
    retry_action_for,
)


def _api_error(cls: type, *, code: int, status: str, message: str) -> Exception:
    return cls(code, {"error": {"code": code, "status": status, "message": message}}, None)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("code", "status", "message", "expected"),
    [
        (
            400,
            "INVALID_ARGUMENT",
            "Request contains an invalid argument.",
            GeminiErrorClass.BAD_REQUEST,
        ),
        (
            400,
            "INVALID_ARGUMENT",
            "Response was blocked due to safety settings.",
            GeminiErrorClass.SAFETY_BLOCKED,
        ),
        (401, "UNAUTHENTICATED", "API key not valid.", GeminiErrorClass.AUTHENTICATION_FAILED),
        (
            403,
            "PERMISSION_DENIED",
            "The caller does not have permission.",
            GeminiErrorClass.PERMISSION_DENIED,
        ),
        (404, "NOT_FOUND", "Model not found.", GeminiErrorClass.NOT_FOUND),
        (409, "ABORTED", "The operation was aborted.", GeminiErrorClass.CONFLICT),
        (
            429,
            "RESOURCE_EXHAUSTED",
            "Quota exceeded for requests per minute.",
            GeminiErrorClass.RATE_LIMITED,
        ),
        (
            429,
            "RESOURCE_EXHAUSTED",
            "Quota exceeded for quota metric GenerateContent requests per day.",
            GeminiErrorClass.DAILY_QUOTA_EXHAUSTED,
        ),
        (429, "SOME_OTHER_STATUS", "Unrecognized 429 shape.", GeminiErrorClass.RESOURCE_EXHAUSTED),
        (499, "CANCELLED", "Request cancelled by client.", GeminiErrorClass.CANCELLED),
    ],
)
def test_classify_client_error(
    code: int, status: str, message: str, expected: GeminiErrorClass
) -> None:
    exc = _api_error(ClientError, code=code, status=status, message=message)
    assert classify_error(exc) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("code", "status", "message", "expected"),
    [
        (500, "INTERNAL", "Internal error.", GeminiErrorClass.INTERNAL_SERVER_ERROR),
        (
            503,
            "UNAVAILABLE",
            "This model is currently experiencing high demand.",
            GeminiErrorClass.SERVER_OVERLOADED,
        ),
        (
            504,
            "DEADLINE_EXCEEDED",
            "Deadline expired before operation could complete.",
            GeminiErrorClass.TIMEOUT,
        ),
        (599, "UNKNOWN", "Some other server error.", GeminiErrorClass.INTERNAL_SERVER_ERROR),
    ],
)
def test_classify_server_error(
    code: int, status: str, message: str, expected: GeminiErrorClass
) -> None:
    exc = _api_error(ServerError, code=code, status=status, message=message)
    assert classify_error(exc) == expected


@pytest.mark.unit
def test_classify_cancelled_error() -> None:
    assert classify_error(asyncio.CancelledError()) == GeminiErrorClass.CANCELLED


@pytest.mark.unit
def test_classify_pydantic_validation_error() -> None:
    class _Schema(BaseModel):
        x: int

    try:
        _Schema.model_validate({"x": "not an int"})
    except ValidationError as exc:
        assert classify_error(exc) == GeminiErrorClass.SCHEMA_VALIDATION_FAILED
    else:
        raise AssertionError("expected ValidationError")


@pytest.mark.unit
def test_classify_output_parser_exception() -> None:
    assert (
        classify_error(OutputParserException("could not parse"))
        == GeminiErrorClass.SCHEMA_VALIDATION_FAILED
    )


@pytest.mark.unit
def test_classify_httpx_timeout() -> None:
    assert classify_error(httpx.ReadTimeout("timed out")) == GeminiErrorClass.TIMEOUT


@pytest.mark.unit
def test_classify_asyncio_timeout_error() -> None:
    assert classify_error(TimeoutError()) == GeminiErrorClass.TIMEOUT


@pytest.mark.unit
def test_classify_unrecognized_exception_is_unknown() -> None:
    assert classify_error(RuntimeError("something else entirely")) == GeminiErrorClass.UNKNOWN


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error_class", "expected_action"),
    [
        (GeminiErrorClass.BAD_REQUEST, RetryAction.FAIL_FAST),
        (GeminiErrorClass.NOT_FOUND, RetryAction.FAIL_FAST),
        (GeminiErrorClass.SAFETY_BLOCKED, RetryAction.FAIL_FAST),
        (GeminiErrorClass.SCHEMA_VALIDATION_FAILED, RetryAction.FAIL_FAST),
        (GeminiErrorClass.CANCELLED, RetryAction.FAIL_FAST),
        (GeminiErrorClass.AUTHENTICATION_FAILED, RetryAction.RETRY_DIFFERENT_CANDIDATE),
        (GeminiErrorClass.PERMISSION_DENIED, RetryAction.RETRY_DIFFERENT_CANDIDATE),
        (GeminiErrorClass.DAILY_QUOTA_EXHAUSTED, RetryAction.RETRY_DIFFERENT_CANDIDATE),
        (GeminiErrorClass.CONFLICT, RetryAction.RETRY_WITH_BACKOFF),
        (GeminiErrorClass.RATE_LIMITED, RetryAction.RETRY_WITH_BACKOFF),
        (GeminiErrorClass.RESOURCE_EXHAUSTED, RetryAction.RETRY_WITH_BACKOFF),
        (GeminiErrorClass.SERVER_OVERLOADED, RetryAction.RETRY_WITH_BACKOFF),
        (GeminiErrorClass.INTERNAL_SERVER_ERROR, RetryAction.RETRY_WITH_BACKOFF),
        (GeminiErrorClass.TIMEOUT, RetryAction.RETRY_WITH_BACKOFF),
        (GeminiErrorClass.UNKNOWN, RetryAction.RETRY_WITH_BACKOFF),
    ],
)
def test_retry_action_for_every_error_class(
    error_class: GeminiErrorClass, expected_action: RetryAction
) -> None:
    assert retry_action_for(error_class) == expected_action


@pytest.mark.unit
def test_every_error_class_has_a_retry_policy_entry() -> None:
    """Defensive: a future new `GeminiErrorClass` member with no policy
    entry must fail loudly (KeyError) at lookup time, not silently no-op --
    this test just documents/pins that every *current* member is covered."""
    for error_class in GeminiErrorClass:
        retry_action_for(error_class)  # raises KeyError if uncovered
