"""Unit tests for `app.models` — the `MockLLM` provider (master brief §6:
mandatory, zero-token-spend, used by all CI/unit tests). No network call is
possible from these tests: `provider="mock"` never touches
`langchain_google_genai` at all.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.guardrails import extract_numbers
from app.models import (
    _RETRY_BACKOFF_SECONDS,
    GeminiModelClient,
    GroundedResult,
    GroundingCitation,
    LoadBalancedGeminiModelClient,
    MockLLM,
    ModelClient,
    UngroundedSearchError,
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
    from app.settings import get_settings

    get_settings.cache_clear()
    try:
        client = get_model_for_role("utility", provider="gemini")
        assert isinstance(client, GeminiModelClient)
    finally:
        get_settings.cache_clear()


@pytest.mark.unit
def test_get_model_for_role_returns_load_balanced_client_when_multiple_keys_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-placeholder-key-a")
    monkeypatch.setenv(
        "GEMINI_API_KEYS_EXTRA", '["unit-test-placeholder-key-b", "unit-test-placeholder-key-c"]'
    )
    from app.settings import get_settings

    get_settings.cache_clear()
    try:
        client = get_model_for_role("analysis", provider="gemini")
        assert isinstance(client, LoadBalancedGeminiModelClient)
    finally:
        get_settings.cache_clear()


@pytest.mark.unit
def test_get_model_for_role_round_robins_the_start_key_across_separate_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Architect-review regression (2026-08-26): `get_model_for_role` builds
    a brand new `LoadBalancedGeminiModelClient` on every call (matching the
    real per-request call pattern in app/main.py and every node) - without a
    round-robin counter shared *across* those instances, every single
    request's first attempt would always hit key 0, silently defeating the
    "spread load across the pool" half of the load balancer (failover would
    still work; fairness would not). This proves the shared, role-keyed
    counter (`_role_key_start_indices`) actually persists across separate
    `get_model_for_role` calls, without ever making a real network call."""
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-placeholder-key-a")
    monkeypatch.setenv(
        "GEMINI_API_KEYS_EXTRA", '["unit-test-placeholder-key-b", "unit-test-placeholder-key-c"]'
    )
    from app.settings import get_settings

    get_settings.cache_clear()
    try:
        first = get_model_for_role("analysis", provider="gemini")
        second = get_model_for_role("analysis", provider="gemini")
        assert isinstance(first, LoadBalancedGeminiModelClient)
        assert isinstance(second, LoadBalancedGeminiModelClient)
        assert first is not second  # confirms this is genuinely a fresh instance per call
        assert first._next_index is second._next_index  # ...sharing the same counter regardless
    finally:
        get_settings.cache_clear()


# --- LoadBalancedGeminiModelClient ---------------------------------------------
#
# Exercised entirely through the generic `clients=` constructor with plain
# fake `ModelClient`s — no real network client, no real API key, ever
# required to test the round-robin/failover logic itself (see
# `LoadBalancedGeminiModelClient.__init__`'s own docstring for why the
# generic shape exists).


class _FakeModelClient:
    """A `ModelClient` test double that either always succeeds (recording
    each call) or always raises `RuntimeError` — enough to drive every
    round-robin/failover scenario below without a real Gemini client.
    `generate_structured`'s signature mirrors `ModelClient`'s own generic
    shape exactly (a bare `type[_OneFieldSchema]` would not structurally
    satisfy the Protocol under mypy) — every scenario below only ever
    passes `_OneFieldSchema`, so the `cast` is safe in practice, not just in
    principle."""

    def __init__(self, *, label: str, fail: bool = False) -> None:
        self.label = label
        self.fail = fail
        self.calls: list[str] = []

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        self.calls.append(user_content)
        if self.fail:
            raise RuntimeError(f"simulated failure on {self.label}")
        return cast(T, _OneFieldSchema(description=f"handled by {self.label}"))

    async def generate_grounded(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> GroundedResult[T]:
        self.calls.append(user_content)
        if self.fail:
            raise RuntimeError(f"simulated failure on {self.label}")
        return GroundedResult(
            value=cast(T, _OneFieldSchema(description=f"handled by {self.label}")),
            citations=[GroundingCitation(source_url=f"https://example.test/{self.label}")],
        )


async def _no_op_sleep(seconds: float) -> None:
    """Replaces the real inter-attempt backoff (`_RETRY_BACKOFF_SECONDS`) in
    every test below except the one that specifically verifies it — mirrors
    `app.pipeline.comtrade_mirror.fetch_with_retry`'s own injectable
    `sleep_fn` testability pattern, so this suite stays fast and
    deterministic rather than incurring real wall-clock delays."""


def _pool(
    *labels_and_fail: tuple[str, bool],
    max_key_attempts: int = 3,
    sleep_fn: Callable[[float], Awaitable[object]] = _no_op_sleep,
) -> tuple[LoadBalancedGeminiModelClient, list[_FakeModelClient]]:
    fakes = [_FakeModelClient(label=label, fail=fail) for label, fail in labels_and_fail]
    clients: list[ModelClient] = list(fakes)
    pool = LoadBalancedGeminiModelClient(
        model="test-model", clients=clients, max_key_attempts=max_key_attempts, sleep_fn=sleep_fn
    )
    return pool, fakes


@pytest.mark.unit
def test_load_balanced_client_rejects_an_empty_client_list() -> None:
    with pytest.raises(ValueError, match="at least one client"):
        LoadBalancedGeminiModelClient(model="test-model", clients=[])


@pytest.mark.unit
async def test_load_balanced_client_uses_the_first_healthy_key() -> None:
    pool, fakes = _pool(("a", False), ("b", False), ("c", False))
    result = await pool.generate_structured(
        system_prompt="sys", user_content="hello", schema=_OneFieldSchema
    )
    assert result.description == "handled by a"
    assert fakes[0].calls == ["hello"]
    assert fakes[1].calls == []
    assert fakes[2].calls == []


@pytest.mark.unit
async def test_load_balanced_client_fails_over_to_the_next_key_on_failure() -> None:
    pool, fakes = _pool(("a", True), ("b", False), ("c", False))
    result = await pool.generate_structured(
        system_prompt="sys", user_content="hello", schema=_OneFieldSchema
    )
    assert result.description == "handled by b"
    assert fakes[0].calls == ["hello"]  # attempted, failed
    assert fakes[1].calls == ["hello"]  # attempted, succeeded
    assert fakes[2].calls == []  # never needed


@pytest.mark.unit
async def test_load_balanced_client_round_robins_the_starting_key_across_calls() -> None:
    # Real load distribution, not just failover: successive top-level calls
    # start from different keys, not just retries within one call.
    pool, fakes = _pool(("a", False), ("b", False), ("c", False))
    await pool.generate_structured(system_prompt="sys", user_content="1", schema=_OneFieldSchema)
    await pool.generate_structured(system_prompt="sys", user_content="2", schema=_OneFieldSchema)
    await pool.generate_structured(system_prompt="sys", user_content="3", schema=_OneFieldSchema)
    assert fakes[0].calls == ["1"]
    assert fakes[1].calls == ["2"]
    assert fakes[2].calls == ["3"]


@pytest.mark.unit
async def test_load_balanced_client_wraps_around_the_pool() -> None:
    pool, fakes = _pool(("a", False), ("b", False))
    await pool.generate_structured(system_prompt="sys", user_content="1", schema=_OneFieldSchema)
    await pool.generate_structured(system_prompt="sys", user_content="2", schema=_OneFieldSchema)
    await pool.generate_structured(system_prompt="sys", user_content="3", schema=_OneFieldSchema)
    assert fakes[0].calls == ["1", "3"]
    assert fakes[1].calls == ["2"]


@pytest.mark.unit
async def test_load_balanced_client_raises_when_every_attempted_key_fails() -> None:
    pool, fakes = _pool(("a", True), ("b", True), ("c", True))
    with pytest.raises(RuntimeError, match="simulated failure on c"):
        await pool.generate_structured(
            system_prompt="sys", user_content="hello", schema=_OneFieldSchema
        )
    assert all(fake.calls == ["hello"] for fake in fakes)


@pytest.mark.unit
async def test_load_balanced_client_reraises_the_real_last_exception_directly_not_wrapped() -> None:
    pool, _fakes = _pool(("a", True))
    with pytest.raises(RuntimeError) as exc_info:
        await pool.generate_structured(
            system_prompt="sys", user_content="hello", schema=_OneFieldSchema
        )
    assert "simulated failure on a" in str(exc_info.value)


class _FakeSchemaValidationError(Exception):
    """Stands in for the real `pydantic.ValidationError`/`langchain_core.
    exceptions.OutputParserException` that `app/main.py`'s `post_message`
    classifies via `isinstance(exc, ValidationError | OutputParserException)`
    (line ~756) to decide `SCHEMA_VALIDATION_FAILED` (non-retryable) vs.
    `INTERNAL_ERROR` (retryable). A distinct local type, not the real
    pydantic one, is enough here: the property under test is "the exact
    exception type survives the pool unwrapped," which doesn't depend on
    which type it is."""


@pytest.mark.unit
async def test_load_balanced_client_preserves_the_exact_exception_type_on_the_last_attempt() -> (
    None
):
    """Regression test for a real, live bug (backend-architect-review
    finding, 2026-08-26): the pool used to wrap total exhaustion in a
    `LoadBalancerExhaustedError`, which broke `app/main.py`'s isinstance-based
    error classification for a genuine schema-validation failure — the
    client would still be told to retry a request that fails deterministically
    every time. This proves the pool is a truly transparent passthrough: the
    caller sees the *exact* real exception type, not a wrapper, not even one
    that chains it via `__cause__`."""

    class _FailingClient:
        async def generate_structured(
            self, *, system_prompt: str, user_content: str, schema: type[T]
        ) -> T:
            raise _FakeSchemaValidationError("the model returned an ungrounded field")

    pool = LoadBalancedGeminiModelClient(
        model="test-model",
        clients=[_FailingClient()],
        max_key_attempts=1,
        sleep_fn=_no_op_sleep,
    )
    with pytest.raises(_FakeSchemaValidationError, match="ungrounded field"):
        await pool.generate_structured(
            system_prompt="sys", user_content="hello", schema=_OneFieldSchema
        )


@pytest.mark.unit
async def test_load_balanced_client_backs_off_between_attempts_but_not_before_the_first() -> None:
    sleep_calls: list[float] = []

    async def spying_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    pool, _fakes = _pool(("a", True), ("b", True), ("c", False), sleep_fn=spying_sleep)
    await pool.generate_structured(
        system_prompt="sys", user_content="hello", schema=_OneFieldSchema
    )
    # 3 attempts (a fails, b fails, c succeeds) -> exactly 2 pauses, one
    # before each retry, none before the very first attempt.
    assert sleep_calls == [_RETRY_BACKOFF_SECONDS, _RETRY_BACKOFF_SECONDS]


@pytest.mark.unit
async def test_load_balanced_client_never_exceeds_max_key_attempts() -> None:
    # A pool of 5 keys with max_key_attempts=2 must never try more than 2,
    # even though 3 more healthy keys are sitting right there unused -
    # bounds worst-case latency (see _DEFAULT_MAX_KEY_ATTEMPTS's own
    # comment) rather than exhaustively trying the whole pool.
    pool, fakes = _pool(
        ("a", True), ("b", True), ("c", False), ("d", False), ("e", False), max_key_attempts=2
    )
    with pytest.raises(RuntimeError, match="simulated failure on b"):
        await pool.generate_structured(
            system_prompt="sys", user_content="hello", schema=_OneFieldSchema
        )
    assert fakes[0].calls == ["hello"]
    assert fakes[1].calls == ["hello"]
    assert fakes[2].calls == []  # never reached - would have succeeded, but the cap stops first


@pytest.mark.unit
async def test_load_balanced_client_caps_max_key_attempts_at_the_pool_size() -> None:
    # max_key_attempts=10 against a 2-client pool must not loop forever or
    # index out of range - capped at len(clients).
    pool, fakes = _pool(("a", True), ("b", True), max_key_attempts=10)
    with pytest.raises(RuntimeError, match="simulated failure on b"):
        await pool.generate_structured(
            system_prompt="sys", user_content="hello", schema=_OneFieldSchema
        )
    assert fakes[0].calls == ["hello"]
    assert fakes[1].calls == ["hello"]


@pytest.mark.unit
def test_load_balanced_client_for_gemini_keys_builds_one_client_per_key_without_network_call() -> (
    None
):
    pool = LoadBalancedGeminiModelClient.for_gemini_keys(
        model="gemini-flash-lite-latest",
        api_keys=["placeholder-a", "placeholder-b", "placeholder-c"],
    )
    assert len(pool._clients) == 3  # whitebox: confirms one real client per key
    assert all(isinstance(client, GeminiModelClient) for client in pool._clients)


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


@pytest.mark.unit
async def test_load_balanced_client_generate_grounded_uses_the_first_healthy_key() -> None:
    pool, fakes = _pool(("a", False), ("b", False), ("c", False))
    result = await pool.generate_grounded(
        system_prompt="sys", user_content="hello", schema=_OneFieldSchema
    )
    assert result.value.description == "handled by a"
    assert result.citations == [GroundingCitation(source_url="https://example.test/a")]
    assert fakes[0].calls == ["hello"]
    assert fakes[1].calls == []


@pytest.mark.unit
async def test_load_balanced_client_generate_grounded_fails_over_to_the_next_key_on_failure() -> (
    None
):
    pool, fakes = _pool(("a", True), ("b", False), ("c", False))
    result = await pool.generate_grounded(
        system_prompt="sys", user_content="hello", schema=_OneFieldSchema
    )
    assert result.value.description == "handled by b"
    assert fakes[0].calls == ["hello"]
    assert fakes[1].calls == ["hello"]
    assert fakes[2].calls == []


@pytest.mark.unit
async def test_load_balanced_client_generate_grounded_raises_when_every_attempted_key_fails() -> (
    None
):
    pool, fakes = _pool(("a", True), ("b", True), ("c", True))
    with pytest.raises(RuntimeError, match="simulated failure on c"):
        await pool.generate_grounded(
            system_prompt="sys", user_content="hello", schema=_OneFieldSchema
        )
    assert all(fake.calls == ["hello"] for fake in fakes)


class _AlwaysUngroundedClient:
    """A `ModelClient` whose `generate_grounded` always raises
    `UngroundedSearchError` — proves the pool rotates to the next key on
    this failure exactly like any other exception (a key that couldn't
    find a real citation is exactly as worth retrying on a different key
    as a transient network error would be)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        raise NotImplementedError

    async def generate_grounded(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> GroundedResult[T]:
        self.calls.append(user_content)
        raise UngroundedSearchError("no real citation found")


@pytest.mark.unit
async def test_load_balanced_client_generate_grounded_rotates_past_an_ungrounded_result() -> None:
    ungrounded = _AlwaysUngroundedClient()
    healthy = _FakeModelClient(label="b", fail=False)
    pool = LoadBalancedGeminiModelClient(
        model="test-model", clients=[ungrounded, healthy], sleep_fn=_no_op_sleep
    )
    result = await pool.generate_grounded(
        system_prompt="sys", user_content="hello", schema=_OneFieldSchema
    )
    assert result.value.description == "handled by b"
    assert ungrounded.calls == ["hello"]
    assert healthy.calls == ["hello"]


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
    metadata = {"grounding_metadata": {"grounding_chunks": [{"retrievedContext": {}}]}}
    assert _extract_citations(metadata) == []


@pytest.mark.unit
def test_extract_citations_skips_a_chunk_whose_web_has_no_uri() -> None:
    metadata = {"grounding_metadata": {"grounding_chunks": [{"web": {"title": "No URI here"}}]}}
    assert _extract_citations(metadata) == []
