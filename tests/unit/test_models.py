"""Unit tests for `app.models` — the `MockLLM` provider (master brief §6:
mandatory, zero-token-spend, used by all CI/unit tests). No network call is
possible from these tests: `provider="mock"` never touches
`langchain_google_genai` at all.
"""

from __future__ import annotations

from typing import Any, TypeVar

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.gemini_scheduler.fallback import ModelFallbackClient
from app.gemini_scheduler.scheduler import GeminiScheduler
from app.guardrails import extract_numbers
from app.models import (
    GeminiModelClient,
    GroundedResult,
    GroundingCitation,
    MockLLM,
    _extract_citations,
    get_model_for_role,
)

T = TypeVar("T", bound=BaseModel)


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


class _NormalizedQuerySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    normalized_query: str


class _FloatFieldSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relevance_score: float


class _HsCodeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hs_code: str
    relevance_score: float


class _HsCodeListSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ranked_candidates: list[_HsCodeCandidate] = Field(min_length=1, max_length=8)


class _HsCodeCandidateWithUnsupportedField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hs_code: str
    weight: int  # unsupported nested-field type


class _HsCodeListWithUnsupportedNestedFieldSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ranked_candidates: list[_HsCodeCandidateWithUnsupportedField]


class _ListOfNonHsCodeModelSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[_TwoFieldSchema]  # a list[Model], but the model has no hs_code field


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
async def test_mock_llm_normalized_query_field_is_a_deterministic_passthrough() -> None:
    """`app.search.normalize.NormalizedQuery.normalized_query` must echo
    `user_content` unchanged under `LLM_PROVIDER=mock` — normalization is
    always a no-op in mock mode, which is what keeps
    `app.search.service.search_products`'s BM25-empty retry path from ever
    firing in mock-based tests/CI (see `_mock_text_for`'s own docstring)."""
    result = await MockLLM().generate_structured(
        system_prompt="sys", user_content="posta dana", schema=_NormalizedQuerySchema
    )
    assert result.normalized_query == "posta dana"


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
async def test_mock_llm_float_field_returns_schema_valid_float() -> None:
    result = await MockLLM().generate_structured(
        system_prompt="sys", user_content="anything", schema=_FloatFieldSchema
    )
    assert isinstance(result.relevance_score, float)
    assert 0.0 <= result.relevance_score <= 1.0


@pytest.mark.unit
async def test_mock_llm_hs_code_list_field_extracts_codes_from_user_content() -> None:
    user_content = "Candidates:\n1. 090111: Coffee, not roasted\n2. 090121: Coffee, roasted"
    result = await MockLLM().generate_structured(
        system_prompt="sys", user_content=user_content, schema=_HsCodeListSchema
    )
    codes = [c.hs_code for c in result.ranked_candidates]
    assert codes == ["090111", "090121"]
    assert all(0.0 <= c.relevance_score <= 1.0 for c in result.ranked_candidates)


@pytest.mark.unit
async def test_mock_llm_hs_code_list_field_is_grounded_by_construction() -> None:
    """Every code MockLLM produces for this field must have come from
    `user_content` verbatim - the same "grounded by construction" property
    `_mock_text_for` already guarantees for `summarize`'s numbers, now
    checked for `app.search.rerank`'s code-identity guardrail instead."""
    user_content = "The candidates are 010121 and 271012, nothing else."
    result = await MockLLM().generate_structured(
        system_prompt="sys", user_content=user_content, schema=_HsCodeListSchema
    )
    produced_codes = {c.hs_code for c in result.ranked_candidates}
    assert produced_codes <= {"010121", "271012"}
    assert produced_codes  # and it did produce at least one


@pytest.mark.unit
async def test_mock_llm_hs_code_list_field_respects_max_length() -> None:
    codes = [f"{100000 + i * 11}" for i in range(12)]  # 12 distinct 6-digit codes
    user_content = "Candidates: " + ", ".join(codes)
    result = await MockLLM().generate_structured(
        system_prompt="sys", user_content=user_content, schema=_HsCodeListSchema
    )
    assert len(result.ranked_candidates) == 8  # _HsCodeListSchema's Field(max_length=8)


@pytest.mark.unit
async def test_mock_llm_hs_code_list_field_deduplicates_repeated_codes() -> None:
    user_content = "090111 appears here and again as 090111, plus 090121 once."
    result = await MockLLM().generate_structured(
        system_prompt="sys", user_content=user_content, schema=_HsCodeListSchema
    )
    codes = [c.hs_code for c in result.ranked_candidates]
    assert codes == ["090111", "090121"]  # not ["090111", "090111", "090121"]


@pytest.mark.unit
async def test_mock_llm_hs_code_list_field_raises_when_no_codes_present() -> None:
    with pytest.raises(ValueError, match="no 6-digit HS codes"):
        await MockLLM().generate_structured(
            system_prompt="sys",
            user_content="no codes mentioned anywhere in this text",
            schema=_HsCodeListSchema,
        )


@pytest.mark.unit
async def test_mock_llm_raises_for_unsupported_nested_field_type() -> None:
    with pytest.raises(NotImplementedError):
        await MockLLM().generate_structured(
            system_prompt="sys",
            user_content="candidate 090111 here",
            schema=_HsCodeListWithUnsupportedNestedFieldSchema,
        )


@pytest.mark.unit
async def test_mock_llm_raises_for_list_of_model_without_hs_code_field() -> None:
    """A `list[NestedModel]` field only gets MockLLM's special handling when
    `NestedModel` has an `hs_code` field - anything else must still fall
    through to the generic NotImplementedError, not be silently mishandled
    as if it were a candidate list."""
    with pytest.raises(NotImplementedError):
        await MockLLM().generate_structured(
            system_prompt="sys", user_content="hello", schema=_ListOfNonHsCodeModelSchema
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
    # Fallback disabled here to isolate the single-credential/no-fallback
    # case this test is actually about - the default-fallback case is
    # covered separately below.
    monkeypatch.setenv("GEMINI_MODEL_FALLBACKS", "{}")
    from app.settings import get_settings

    get_settings.cache_clear()
    try:
        client = get_model_for_role("utility", provider="gemini")
        assert isinstance(client, GeminiModelClient)
    finally:
        get_settings.cache_clear()


@pytest.mark.unit
def test_get_model_for_role_returns_gemini_scheduler_when_multiple_keys_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-placeholder-key-a")
    monkeypatch.setenv(
        "GEMINI_API_KEYS_EXTRA", '["unit-test-placeholder-key-b", "unit-test-placeholder-key-c"]'
    )
    monkeypatch.setenv("GEMINI_MODEL_FALLBACKS", "{}")
    from app.settings import get_settings

    get_settings.cache_clear()
    try:
        client = get_model_for_role("analysis", provider="gemini")
        assert isinstance(client, GeminiScheduler)
    finally:
        get_settings.cache_clear()


@pytest.mark.unit
def test_get_model_for_role_shares_the_fairness_counter_across_separate_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Architect-review regression (2026-08-26, originally against
    `LoadBalancedGeminiModelClient`, the class `GeminiScheduler` replaced
    2026-09-04): `get_model_for_role` builds a brand new scheduler on every
    call (matching the real per-request call pattern in app/main.py and
    every node) - without a fairness counter shared *across* those
    instances, every single request's first attempt would always favor the
    same credential, silently defeating the "spread load across the pool"
    property (failover would still work; fairness would not). This proves
    the shared, role-keyed counter (`_role_fairness_counters`) actually
    persists across separate `get_model_for_role` calls, without ever
    making a real network call."""
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-placeholder-key-a")
    monkeypatch.setenv(
        "GEMINI_API_KEYS_EXTRA", '["unit-test-placeholder-key-b", "unit-test-placeholder-key-c"]'
    )
    monkeypatch.setenv("GEMINI_MODEL_FALLBACKS", "{}")
    from app.settings import get_settings

    get_settings.cache_clear()
    try:
        first = get_model_for_role("analysis", provider="gemini")
        second = get_model_for_role("analysis", provider="gemini")
        assert isinstance(first, GeminiScheduler)
        assert isinstance(second, GeminiScheduler)
        assert first is not second  # confirms this is genuinely a fresh instance per call
        assert (
            first._fairness_counter is second._fairness_counter
        )  # ...sharing the same counter regardless
    finally:
        get_settings.cache_clear()


@pytest.mark.unit
def test_get_model_for_role_wraps_in_fallback_client_when_fallbacks_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-placeholder-key-a")
    monkeypatch.setenv(
        "GEMINI_API_KEYS_EXTRA", '["unit-test-placeholder-key-b", "unit-test-placeholder-key-c"]'
    )
    monkeypatch.setenv(
        "GEMINI_MODEL_FALLBACKS", '{"analysis": ["fallback-model-a", "fallback-model-b"]}'
    )
    from app.settings import get_settings

    get_settings.cache_clear()
    try:
        client = get_model_for_role("analysis", provider="gemini")
        assert isinstance(client, ModelFallbackClient)
        assert len(client._clients) == 3  # primary + 2 configured fallbacks
        primary, fallback_a, fallback_b = client._clients
        assert isinstance(primary, GeminiScheduler)
        assert primary._model == get_settings().model_analysis
        assert isinstance(fallback_a, GeminiScheduler)
        assert fallback_a._model == "fallback-model-a"
        assert isinstance(fallback_b, GeminiScheduler)
        assert fallback_b._model == "fallback-model-b"
    finally:
        get_settings.cache_clear()


@pytest.mark.unit
def test_get_model_for_role_fallback_wrapping_works_for_a_single_credential_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback wrapping is independent of credential-pool size - even a
    single-credential deployment benefits from a currently-idle model
    version having real headroom the primary one doesn't."""
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-placeholder-key")
    monkeypatch.setenv("GEMINI_MODEL_FALLBACKS", '{"utility": ["fallback-model-c"]}')
    from app.settings import get_settings

    get_settings.cache_clear()
    try:
        client = get_model_for_role("utility", provider="gemini")
        assert isinstance(client, ModelFallbackClient)
        assert len(client._clients) == 2
        primary, fallback = client._clients
        assert isinstance(primary, GeminiModelClient)
        assert isinstance(fallback, GeminiModelClient)
    finally:
        get_settings.cache_clear()


# --- generate_grounded / GroundedResult (2026-09-02, Step 4 hardening) -------


@pytest.mark.unit
async def test_mock_llm_generate_grounded_returns_deterministic_citation() -> None:
    result = await MockLLM().generate_grounded(
        system_prompt="sys", user_content="hello", schema=_OneFieldSchema
    )
    assert isinstance(result, GroundedResult)
    assert isinstance(result.value, _OneFieldSchema)
    assert len(result.citations) == 1
    assert result.citations[0].source_url.startswith("https://")


# --- _extract_citations (2026-09-02, Step 4 hardening) -----------------------


@pytest.mark.unit
def test_extract_citations_empty_when_no_grounding_metadata_key() -> None:
    assert _extract_citations({"finish_reason": "STOP"}) == []


@pytest.mark.unit
def test_extract_citations_empty_when_grounding_metadata_is_empty() -> None:
    assert _extract_citations({"grounding_metadata": {}}) == []
    assert _extract_citations({"grounding_metadata": None}) == []


@pytest.mark.unit
def test_extract_citations_parses_real_shaped_grounding_chunks() -> None:
    metadata = {
        "grounding_metadata": {
            "grounding_chunks": [
                {"web": {"uri": "https://example.test/a", "title": "Source A"}},
                {"web": {"uri": "https://example.test/b", "title": "Source B"}},
            ]
        }
    }
    citations = _extract_citations(metadata)
    assert citations == [
        GroundingCitation(source_url="https://example.test/a", title="Source A"),
        GroundingCitation(source_url="https://example.test/b", title="Source B"),
    ]


@pytest.mark.unit
def test_extract_citations_parses_the_camel_case_groundingchunks_variant() -> None:
    metadata = {
        "grounding_metadata": {"groundingChunks": [{"web": {"uri": "https://example.test/c"}}]}
    }
    assert _extract_citations(metadata) == [GroundingCitation(source_url="https://example.test/c")]


@pytest.mark.unit
def test_extract_citations_title_is_none_when_not_present() -> None:
    metadata = {
        "grounding_metadata": {"grounding_chunks": [{"web": {"uri": "https://example.test/d"}}]}
    }
    citations = _extract_citations(metadata)
    assert citations == [GroundingCitation(source_url="https://example.test/d", title=None)]


@pytest.mark.unit
def test_extract_citations_skips_a_chunk_with_no_web_field() -> None:
    metadata: dict[str, Any] = {
        "grounding_metadata": {"grounding_chunks": [{"retrievedContext": {}}]}
    }
    assert _extract_citations(metadata) == []


@pytest.mark.unit
def test_extract_citations_skips_a_chunk_whose_web_has_no_uri() -> None:
    metadata = {"grounding_metadata": {"grounding_chunks": [{"web": {"title": "No URI here"}}]}}
    assert _extract_citations(metadata) == []
