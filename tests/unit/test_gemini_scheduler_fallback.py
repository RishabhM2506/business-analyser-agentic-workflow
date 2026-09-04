"""Unit tests for `app.gemini_scheduler.fallback.ModelFallbackClient` --
the cross-model fallback trigger (capacity exhaustion only, never a
FAIL_FAST request-shape failure)."""

from __future__ import annotations

from typing import TypeVar

import pytest
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel

from app.gemini_scheduler.fallback import ModelFallbackClient
from app.gemini_scheduler.scheduler import NoEligibleGeminiCandidateError
from app.models import GroundedResult

T = TypeVar("T", bound=BaseModel)


class _OneFieldSchema(BaseModel):
    value: str


def _api_error(cls: type[Exception], *, code: int, status: str, message: str = "") -> Exception:
    exc = cls(code, {"error": {"code": code, "status": status, "message": message}}, None)
    assert isinstance(exc, Exception)
    return exc


class _FakeClient:
    def __init__(self, *, error: Exception | None = None, value: str = "ok") -> None:
        self.error = error
        self.value = value
        self.call_count = 0

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return schema.model_validate({"value": self.value})

    async def generate_grounded(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> GroundedResult[T]:
        raise NotImplementedError


@pytest.mark.unit
async def test_succeeds_on_the_primary_client_without_trying_fallbacks() -> None:
    primary = _FakeClient(value="from primary")
    fallback = _FakeClient(value="from fallback")
    client = ModelFallbackClient([primary, fallback])

    result = await client.generate_structured(
        system_prompt="s", user_content="u", schema=_OneFieldSchema
    )

    assert result.value == "from primary"
    assert fallback.call_count == 0


@pytest.mark.unit
async def test_falls_back_on_no_eligible_candidate_error() -> None:
    primary = _FakeClient(error=NoEligibleGeminiCandidateError("nothing eligible"))
    fallback = _FakeClient(value="from fallback")
    client = ModelFallbackClient([primary, fallback])

    result = await client.generate_structured(
        system_prompt="s", user_content="u", schema=_OneFieldSchema
    )

    assert result.value == "from fallback"


@pytest.mark.unit
async def test_falls_back_on_a_real_capacity_exhaustion_failure() -> None:
    primary = _FakeClient(error=_api_error(ServerError, code=503, status="UNAVAILABLE"))
    fallback = _FakeClient(value="from fallback")
    client = ModelFallbackClient([primary, fallback])

    result = await client.generate_structured(
        system_prompt="s", user_content="u", schema=_OneFieldSchema
    )

    assert result.value == "from fallback"


@pytest.mark.unit
async def test_never_falls_back_on_a_fail_fast_class() -> None:
    """The spec's own guardrail: a bad-request/schema-validation-shaped
    failure must never trigger a silent model change - it's about the
    request, not provider capacity."""
    bad_request = _api_error(ClientError, code=400, status="INVALID_ARGUMENT")
    primary = _FakeClient(error=bad_request)
    fallback = _FakeClient(value="from fallback")
    client = ModelFallbackClient([primary, fallback])

    with pytest.raises(ClientError):
        await client.generate_structured(
            system_prompt="s", user_content="u", schema=_OneFieldSchema
        )

    assert fallback.call_count == 0


@pytest.mark.unit
async def test_raises_the_real_last_exception_when_every_client_is_exhausted() -> None:
    exc_a = _api_error(ServerError, code=503, status="UNAVAILABLE")
    exc_b = _api_error(ServerError, code=504, status="DEADLINE_EXCEEDED")
    client = ModelFallbackClient([_FakeClient(error=exc_a), _FakeClient(error=exc_b)])

    with pytest.raises(ServerError) as excinfo:
        await client.generate_structured(
            system_prompt="s", user_content="u", schema=_OneFieldSchema
        )

    assert excinfo.value is exc_b


@pytest.mark.unit
async def test_tries_clients_strictly_in_order() -> None:
    first = _FakeClient(error=NoEligibleGeminiCandidateError("x"))
    second = _FakeClient(error=NoEligibleGeminiCandidateError("x"))
    third = _FakeClient(value="from third")
    client = ModelFallbackClient([first, second, third])

    result = await client.generate_structured(
        system_prompt="s", user_content="u", schema=_OneFieldSchema
    )

    assert result.value == "from third"
    assert first.call_count == 1
    assert second.call_count == 1
    assert third.call_count == 1


@pytest.mark.unit
def test_requires_at_least_one_client() -> None:
    with pytest.raises(ValueError, match="at least one client"):
        ModelFallbackClient([])
