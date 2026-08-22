"""Integration tests for `app.search.service.search_products` — the
top-level find -> (normalize retry ->) rerank -> threshold orchestrator.

The most important test here is
`test_search_products_normalizes_a_bm25_empty_query_and_finds_correct_code`:
it directly encodes a live, user-reported bug (2026-08-21) as a permanent
regression test, mirroring this project's established pattern of turning a
real found failure into the guardrail's own test
(`tests/integration/test_rerank.py`'s `_InventedCodeModelClient`).

Real bug: searching "posta dana" (Hindi for poppy seed, HS6 120791)
returned results about meat instead. Root cause, verified live against the
real backend + real Gemini key + real embeddings corpus: BM25 found zero
lexical overlap for "posta dana" (expected - it's not English), and the
top vector-search result was noise-like (postage stamps / bovine meat
codes scored higher than the correct code, which ranked 623rd of 5613) -
noisy enough to clear `HybridSearchProvider`'s pre-reranker floor but not
a real match. A verified fix: translate the query to standard English
trade terminology ("poppy seeds") before searching, which makes 120791 the
unambiguous #1 result in both BM25 and vector search. This suite tests
`search_products`'s orchestration of that fix via fakes, not live calls.
"""

from __future__ import annotations

import re
from typing import TypeVar

import pytest
from pydantic import BaseModel

from app.budget import BudgetExceededError, BudgetTracker
from app.search.candidates import SearchCandidate
from app.search.normalize import NormalizedQuery
from app.search.rerank import RerankOutput
from app.search.service import search_products

T = TypeVar("T", bound=BaseModel)

_HS6_PATTERN = re.compile(r"\b\d{6}\b")


class _QueryDependentProvider:
    """Fake `ProductSearchProvider` whose result depends on the exact query
    text it receives - stands in for the verified real-world before/after
    candidate sets (noisy for the raw vernacular query, correct for the
    translated one) without depending on the real embeddings corpus."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def find_candidates(self, query_text: str, *, top_k: int = 8) -> list[SearchCandidate]:
        self.calls.append(query_text)
        if query_text == "posta dana":
            # The real, live-observed noisy result: postage stamps and
            # bovine meat, cleared the pre-reranker vector-similarity floor
            # by coincidence, but neither is the correct answer.
            return [
                SearchCandidate(hs_code="490700", description="Postage stamps", fusion_score=0.9),
                SearchCandidate(
                    hs_code="020110", description="Bovine meat; carcasses", fusion_score=0.85
                ),
            ]
        if query_text == "poppy seeds":
            return [
                SearchCandidate(
                    hs_code="120791",
                    description="Oil seeds; poppy seeds, whether or not broken",
                    fusion_score=0.9,
                )
            ]
        return []


class _AlwaysFindsProvider:
    """Fake `ProductSearchProvider` returning a fixed candidate list for
    any query - used where the specific candidates don't matter, only
    whether/how many times `find_candidates` was called."""

    def __init__(self, candidates: list[SearchCandidate]) -> None:
        self._candidates = candidates
        self.calls: list[str] = []

    async def find_candidates(self, query_text: str, *, top_k: int = 8) -> list[SearchCandidate]:
        self.calls.append(query_text)
        return list(self._candidates)


class _TranslatingModelClient:
    """Fake `ModelClient` covering both real model calls `search_products`
    can make: translates "posta dana" -> "poppy seeds" for the
    normalization schema; for the rerank schema, scores by position *only
    if* the "Search text" it was given is one it actually recognizes -
    live-reproduced (2026-08-21): the real Gemini reranker, given retrieved
    poppy-seed candidates but the untranslated "posta dana" as the search
    text, could not relate the two and scored every candidate 0.0. This
    double reproduces that exact failure mode instead of the more
    permissive "just echo whatever codes are present" a first version of
    this test used - which passed even while `search_products` had this
    live bug, since it never checked whether the search text made sense
    for the candidates it was scoring."""

    _RECOGNIZED_SEARCH_TEXT = re.compile(r'Search text: "(coffee|poppy seeds)"')

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        if schema is NormalizedQuery:
            normalized = "poppy seeds" if user_content.strip() == "posta dana" else user_content
            return schema.model_validate({"normalized_query": normalized})
        if schema is RerankOutput:
            codes = list(dict.fromkeys(_HS6_PATTERN.findall(user_content)))
            if not self._RECOGNIZED_SEARCH_TEXT.search(user_content):
                ranked = [{"hs_code": code, "relevance_score": 0.0} for code in codes]
                return schema.model_validate({"ranked_candidates": ranked})
            # Mirrors `test_rerank.py`'s `_EchoingModelClient`: scores by
            # position, descending - whichever candidate the provider
            # listed first (the only one, in every case this double is used
            # for) clears `HIGH_CONFIDENCE_THRESHOLD`, regardless of which
            # code it happens to be.
            ranked = [
                {"hs_code": code, "relevance_score": max(0.1, 0.95 - 0.3 * i)}
                for i, code in enumerate(codes)
            ]
            return schema.model_validate({"ranked_candidates": ranked})
        raise AssertionError(f"unexpected schema in test double: {schema}")


class _EchoNormalizeModelClient:
    """Fake `ModelClient` whose normalization call always echoes the input
    back unchanged (simulating genuine gibberish/untranslatable input, or
    an already-standard query) - the rerank schema is never expected to be
    reached in tests using this double, since the provider these tests pair
    it with returns no candidates either way."""

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        if schema is NormalizedQuery:
            return schema.model_validate({"normalized_query": user_content.strip()})
        raise AssertionError(f"unexpected schema in test double: {schema}")


def _tracker(*, max_calls_per_thread: int = 10) -> BudgetTracker:
    return BudgetTracker(max_calls_per_thread=max_calls_per_thread, max_calls_per_day=100)


@pytest.mark.integration
async def test_search_products_normalizes_a_bm25_empty_query_and_finds_correct_code() -> None:
    provider = _QueryDependentProvider()

    result = await search_products(
        "posta dana",
        thread_id="thread-1",
        tenant_id="default",
        search_provider=provider,
        model_client=_TranslatingModelClient(),
        budget_tracker=_tracker(),
    )

    assert result.outcome == "auto_selected"
    assert result.selected_hs_code == "120791"
    # Retried with the translated term after the raw term's real BM25
    # lookup found zero lexical overlap (verified against the real,
    # checked-in taxonomy - "posta dana" has none).
    assert provider.calls == ["posta dana", "poppy seeds"]


@pytest.mark.integration
async def test_search_products_skips_normalization_when_bm25_already_finds_something() -> None:
    """A query with real lexical overlap (BM25 non-empty) must never pay
    for a normalization call - the "well-formed query stays free" property
    `search_products`'s own docstring already documents for the pre-reranker
    floor, preserved here for the common case."""
    provider = _AlwaysFindsProvider(
        [SearchCandidate(hs_code="090111", description="Coffee, not roasted", fusion_score=0.9)]
    )
    model_client = _TranslatingModelClient()

    result = await search_products(
        "coffee",
        thread_id="thread-1",
        tenant_id="default",
        search_provider=provider,
        model_client=model_client,
        budget_tracker=_tracker(),
    )

    assert result.outcome in {"auto_selected", "disambiguate"}
    # find_candidates called exactly once, with the raw query - no retry.
    assert provider.calls == ["coffee"]


@pytest.mark.integration
async def test_search_products_skips_retry_search_when_normalization_is_a_no_op() -> None:
    """When the model determines nothing needed translating (returns the
    query unchanged), `search_products` must not re-call `find_candidates`
    a second time - a pointless, redundant search."""
    provider = _AlwaysFindsProvider([])  # BM25-empty gibberish genuinely finds nothing either way

    result = await search_products(
        "zzzqqqxxx nonsensegibberish",
        thread_id="thread-1",
        tenant_id="default",
        search_provider=provider,
        model_client=_EchoNormalizeModelClient(),
        budget_tracker=_tracker(),
    )

    assert result.outcome == "no_candidates_found"
    assert provider.calls == ["zzzqqqxxx nonsensegibberish"]


@pytest.mark.integration
async def test_search_products_budget_exhausted_before_normalize_raises_early() -> None:
    """The budget check gating the normalization call must fail closed
    exactly like the one gating rerank (`app.nodes.describe_item`'s
    established sequencing) - a thread with zero remaining calls must never
    reach the model at all."""
    provider = _QueryDependentProvider()

    with pytest.raises(BudgetExceededError):
        await search_products(
            "posta dana",
            thread_id="thread-1",
            tenant_id="default",
            search_provider=provider,
            model_client=_TranslatingModelClient(),
            budget_tracker=_tracker(max_calls_per_thread=0),
        )
