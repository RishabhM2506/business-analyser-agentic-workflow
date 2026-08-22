"""Unit tests for `app.search.bm25` — the hand-rolled Okapi BM25 index.

Two kinds of coverage: a synthetic, hand-computed-score corpus (precise,
independent of the real taxonomy's exact content) for the scoring formula
itself, and a handful of sanity checks against the real, checked-in
taxonomy for the end-to-end `search_bm25` entry point.
"""

from __future__ import annotations

import math

import pytest

import app.search.bm25 as bm25_module
from app.knowledge.provider import TaxonomyEntry
from app.search.bm25 import _tokenize, search_bm25

_SYNTHETIC_ENTRIES = [
    TaxonomyEntry(
        hs_code="000001", description="apple banana apple", section="1", parent="0", level="6"
    ),
    TaxonomyEntry(
        hs_code="000002", description="banana cherry", section="1", parent="0", level="6"
    ),
    TaxonomyEntry(hs_code="000003", description="cherry date", section="1", parent="0", level="6"),
]


def _patch_synthetic_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bm25_module,
        "get_hs6_taxonomy_entries",
        lambda *, taxonomy_path: _SYNTHETIC_ENTRIES,
    )


@pytest.mark.unit
def test_tokenize_lowercases_and_splits_on_non_word_characters() -> None:
    assert _tokenize("Coffee, roasted; not decaffeinated!") == [
        "coffee",
        "roasted",
        "not",
        "decaffeinated",
    ]


@pytest.mark.unit
def test_tokenize_empty_string_returns_empty_list() -> None:
    assert _tokenize("") == []


@pytest.mark.unit
def test_search_bm25_hand_computed_scores_and_length_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hand-computed against the synthetic 3-doc corpus above (`k1=1.5`,
    `b=0.75`): "banana" appears once in doc `000001` (length 3 tokens) and
    once in doc `000002` (length 2 tokens, `avg_doc_length` = 7/3). BM25's
    length normalization must rank the *shorter* matching doc higher for an
    equal raw term frequency — this is exactly what distinguishes BM25 from
    plain TF-IDF, so a passing test here is real evidence of that behavior,
    not just "some positive score came back"."""
    _patch_synthetic_corpus(monkeypatch)

    results = search_bm25("banana", top_k=10, taxonomy_path="synthetic:test-hand-computed")

    assert [hs_code for hs_code, _score in results] == ["000002", "000001"]

    idf_banana = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
    avg_len = 7 / 3  # doc lengths 3, 2, 2
    expected_score_000002 = idf_banana * (1 * 2.5) / (1 + 1.5 * (0.25 + 0.75 * 2 / avg_len))
    expected_score_000001 = idf_banana * (1 * 2.5) / (1 + 1.5 * (0.25 + 0.75 * 3 / avg_len))

    scores_by_code = dict(results)
    assert scores_by_code["000002"] == pytest.approx(expected_score_000002, rel=1e-9)
    assert scores_by_code["000001"] == pytest.approx(expected_score_000001, rel=1e-9)


@pytest.mark.unit
def test_search_bm25_excludes_docs_with_zero_term_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_corpus(monkeypatch)

    results = search_bm25("apple", top_k=10, taxonomy_path="synthetic:test-zero-overlap")

    # Only doc "000001" contains "apple" at all; "000002"/"000003" must not
    # appear, not even with a score of 0 (a zero-overlap doc is not a
    # low-relevance match — see `search_bm25`'s own docstring).
    assert [hs_code for hs_code, _score in results] == ["000001"]


@pytest.mark.unit
def test_search_bm25_repeated_term_scores_higher_than_single_occurrence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "apple" occurs twice in doc `000001` and zero times elsewhere —
    term-frequency saturation still means a repeated term contributes a
    strictly positive marginal score versus a hypothetical single
    occurrence, which this asserts indirectly by checking the score is
    strictly greater than the single-occurrence IDF term alone would be
    (i.e., the `numerator/denominator` factor from `_score`'s formula
    exceeds 1 for `f_td=2`)."""
    _patch_synthetic_corpus(monkeypatch)

    results = search_bm25("apple", top_k=10, taxonomy_path="synthetic:test-tf-saturation")
    idf_apple = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)

    assert results[0][1] > idf_apple  # score > bare IDF: TF component contributed > 1x


@pytest.mark.unit
def test_search_bm25_ranks_coffee_query_top_results_as_coffee_codes() -> None:
    results = search_bm25("coffee", top_k=5)

    assert len(results) == 5
    hs_codes = [hs_code for hs_code, _score in results]
    assert all(hs_code.startswith("0901") for hs_code in hs_codes)


@pytest.mark.unit
def test_search_bm25_results_are_sorted_descending_by_score() -> None:
    results = search_bm25("coffee beans roasted", top_k=10)
    scores = [score for _hs_code, score in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_search_bm25_respects_top_k() -> None:
    results = search_bm25("coffee tea rice wheat", top_k=3)
    assert len(results) <= 3


@pytest.mark.unit
def test_search_bm25_empty_query_returns_no_results() -> None:
    assert search_bm25("", top_k=10) == []


@pytest.mark.unit
def test_search_bm25_query_with_no_term_overlap_returns_no_results() -> None:
    assert search_bm25("zzzqqqxxx", top_k=10) == []
