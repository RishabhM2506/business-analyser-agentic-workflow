"""Unit tests for `app.search.fusion.reciprocal_rank_fusion` — hand-computed
against the standard RRF formula (`k=60`)."""

from __future__ import annotations

import pytest

from app.search.fusion import _RRF_K, reciprocal_rank_fusion


@pytest.mark.unit
def test_rrf_hand_computed_scores_two_rankings() -> None:
    bm25 = [("A", 9.0), ("B", 5.0), ("C", 1.0)]
    vector = [("B", 0.9), ("D", 0.8), ("A", 0.5)]

    results = reciprocal_rank_fusion(bm25, vector, top_k=10)
    scores = dict(results)

    assert scores["A"] == pytest.approx(1 / (60 + 1) + 1 / (60 + 3))
    assert scores["B"] == pytest.approx(1 / (60 + 2) + 1 / (60 + 1))
    assert scores["C"] == pytest.approx(1 / (60 + 3))
    assert scores["D"] == pytest.approx(1 / (60 + 2))
    assert set(scores) == {"A", "B", "C", "D"}


@pytest.mark.unit
def test_rrf_orders_descending_by_fused_score() -> None:
    bm25 = [("A", 9.0), ("B", 5.0), ("C", 1.0)]
    vector = [("B", 0.9), ("D", 0.8), ("A", 0.5)]

    results = reciprocal_rank_fusion(bm25, vector, top_k=10)

    scores = [score for _code, score in results]
    assert scores == sorted(scores, reverse=True)
    # B is rank 2 in bm25 + rank 1 in vector; A is rank 1 in bm25 + rank 3 in
    # vector -- 1/62 + 1/61 (B) > 1/61 + 1/63 (A), so B must be fused first.
    assert results[0][0] == "B"


@pytest.mark.unit
def test_rrf_a_doc_present_in_only_one_ranking_still_contributes() -> None:
    bm25 = [("A", 9.0)]
    vector: list[tuple[str, float]] = []

    results = reciprocal_rank_fusion(bm25, vector, top_k=10)

    assert results == [("A", pytest.approx(1 / (_RRF_K + 1)))]


@pytest.mark.unit
def test_rrf_empty_rankings_return_empty_result() -> None:
    assert reciprocal_rank_fusion([], [], top_k=10) == []


@pytest.mark.unit
def test_rrf_respects_top_k_truncation() -> None:
    bm25 = [(f"code{i}", float(10 - i)) for i in range(5)]

    results = reciprocal_rank_fusion(bm25, top_k=2)

    assert len(results) == 2
    assert [code for code, _score in results] == ["code0", "code1"]


@pytest.mark.unit
def test_rrf_supports_more_than_two_rankings() -> None:
    ranking_a = [("A", 1.0), ("B", 2.0)]
    ranking_b = [("B", 1.0), ("A", 2.0)]
    ranking_c = [("A", 1.0), ("C", 2.0)]

    results = reciprocal_rank_fusion(ranking_a, ranking_b, ranking_c, top_k=10)
    scores = dict(results)

    # A: rank1 in a, rank2 in b, rank1 in c
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 62 + 1 / 61)
    # B: rank2 in a, rank1 in b
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)
    # C: rank2 in c only
    assert scores["C"] == pytest.approx(1 / 62)
