"""Unit tests for `app.models` — the `MockLLM` provider (master brief §6:
mandatory, zero-token-spend, used by all CI/unit tests). No network call is
possible from these tests: `provider="mock"` never touches
`langchain_google_genai` at all.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from app.guardrails import extract_numbers
from app.models import GeminiModelClient, MockLLM, get_model_for_role


class _OneFieldSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str


class _TwoFieldSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    body: str


class _UnsupportedSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int


@pytest.mark.unit
async def test_mock_llm_returns_schema_valid_instance() -> None:
    result = await MockLLM().generate_structured(
        system_prompt="sys", user_content="some content", schema=_OneFieldSchema
    )
    assert isinstance(result, _OneFieldSchema)
    assert isinstance(result.description, str)
    assert len(result.description) > 0


@pytest.mark.unit
async def test_mock_llm_is_deterministic_for_same_input() -> None:
    a = await MockLLM().generate_structured(
        system_prompt="sys", user_content="fixed content", schema=_OneFieldSchema
    )
    b = await MockLLM().generate_structured(
        system_prompt="sys", user_content="fixed content", schema=_OneFieldSchema
    )
    assert a == b


@pytest.mark.unit
async def test_mock_llm_output_echoes_only_numbers_present_in_input() -> None:
    user_content = "IMPORTS table: USA cumulative 1,992,456 USD, year 2023: 1,992,456"
    result = await MockLLM().generate_structured(
        system_prompt="sys", user_content=user_content, schema=_OneFieldSchema
    )
    output_numbers = extract_numbers(result.description)
    input_numbers = set(extract_numbers(user_content))
    # every number in the mock output must have come from the input verbatim
    assert all(n in input_numbers for n in output_numbers)


@pytest.mark.unit
async def test_mock_llm_falls_back_to_excerpt_when_no_numbers_present() -> None:
    result = await MockLLM().generate_structured(
        system_prompt="sys", user_content="no digits in this text at all", schema=_OneFieldSchema
    )
    assert "no digits in this text" in result.description


@pytest.mark.unit
async def test_mock_llm_handles_multi_field_schema() -> None:
    result = await MockLLM().generate_structured(
        system_prompt="sys", user_content="hello world", schema=_TwoFieldSchema
    )
    assert isinstance(result.title, str)
    assert isinstance(result.body, str)


@pytest.mark.unit
async def test_mock_llm_raises_for_unsupported_field_type() -> None:
    with pytest.raises(NotImplementedError):
        await MockLLM().generate_structured(
            system_prompt="sys", user_content="hello", schema=_UnsupportedSchema
        )


@pytest.mark.unit
def test_get_model_for_role_mock_provider_returns_mock_llm() -> None:
    assert isinstance(get_model_for_role("utility", provider="mock"), MockLLM)
    assert isinstance(get_model_for_role("analysis", provider="mock"), MockLLM)


@pytest.mark.unit
def test_get_model_for_role_gemini_provider_returns_gemini_client_without_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Constructing ChatGoogleGenerativeAI does not itself make a network
    # call (verified directly against the installed package) - safe to
    # exercise in CI with a placeholder key and no network access.
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-placeholder-key")
    from app.settings import get_settings

    get_settings.cache_clear()
    try:
        client = get_model_for_role("utility", provider="gemini")
        assert isinstance(client, GeminiModelClient)
    finally:
        get_settings.cache_clear()
