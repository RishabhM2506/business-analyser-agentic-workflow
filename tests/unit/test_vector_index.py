"""Unit tests for `app.search.vector_index` — synthetic vectors only, no
real embeddings and no network. Each test writes its own tiny `.npy`/
`.hscodes.txt`/`.meta.json` fixture triple to a temp directory (via
`tmp_path`) rather than depending on the real, offline-generated
`data/hs_taxonomy_embeddings.*` files (`scripts/embed_taxonomy.py`) so
these stay fast and independent of whether that one-time script has been
run in a given checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.search.vector_index import EmbeddingsFileMismatchError, search_vector


def _write_fixture(
    tmp_path: Path,
    *,
    stem: str,
    vectors: np.ndarray,
    hs_codes: list[str],
    count: int | None = None,
    dims: int | None = None,
) -> str:
    npy_path = tmp_path / f"{stem}.npy"
    hscodes_path = tmp_path / f"{stem}.hscodes.txt"
    meta_path = tmp_path / f"{stem}.meta.json"

    np.save(npy_path, vectors.astype(np.float32))
    hscodes_path.write_text("\n".join(hs_codes), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "count": count if count is not None else len(hs_codes),
                "dims": dims if dims is not None else vectors.shape[1],
            }
        ),
        encoding="utf-8",
    )
    return str(tmp_path / stem)


def _unit_vectors() -> tuple[np.ndarray, list[str]]:
    """Three mutually orthogonal-ish unit vectors in 3-D so cosine
    similarity is trivial to reason about by hand."""
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return vectors, ["100000", "200000", "300000"]


@pytest.mark.unit
def test_search_vector_exact_match_scores_one(tmp_path: Path) -> None:
    vectors, hs_codes = _unit_vectors()
    embeddings_path = _write_fixture(
        tmp_path, stem="fixture_exact", vectors=vectors, hs_codes=hs_codes
    )

    results = search_vector([1.0, 0.0, 0.0], top_k=3, embeddings_path=embeddings_path)

    assert results[0] == ("100000", pytest.approx(1.0))


@pytest.mark.unit
def test_search_vector_orthogonal_vectors_score_zero(tmp_path: Path) -> None:
    vectors, hs_codes = _unit_vectors()
    embeddings_path = _write_fixture(
        tmp_path, stem="fixture_orthogonal", vectors=vectors, hs_codes=hs_codes
    )

    results = search_vector([1.0, 0.0, 0.0], top_k=3, embeddings_path=embeddings_path)
    scores = dict(results)

    assert scores["200000"] == pytest.approx(0.0, abs=1e-6)
    assert scores["300000"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_search_vector_normalizes_unnormalized_query(tmp_path: Path) -> None:
    vectors, hs_codes = _unit_vectors()
    embeddings_path = _write_fixture(
        tmp_path, stem="fixture_unnormalized", vectors=vectors, hs_codes=hs_codes
    )

    # A query vector scaled by 5x must still score an exact-direction match
    # as 1.0 -- cosine similarity is scale-invariant, and `search_vector`
    # must normalize the query itself (the caller only guarantees whatever
    # `EmbeddingsClient.embed_query` returned, not that it's unit-length).
    results = search_vector([5.0, 0.0, 0.0], top_k=1, embeddings_path=embeddings_path)

    assert results == [("100000", pytest.approx(1.0))]


@pytest.mark.unit
def test_search_vector_respects_top_k_and_descending_order(tmp_path: Path) -> None:
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    hs_codes = ["100000", "200000", "300000"]
    embeddings_path = _write_fixture(
        tmp_path, stem="fixture_topk", vectors=vectors, hs_codes=hs_codes
    )

    results = search_vector([1.0, 0.0], top_k=2, embeddings_path=embeddings_path)

    assert len(results) == 2
    assert [hs_code for hs_code, _score in results] == ["100000", "200000"]
    scores = [score for _hs_code, score in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_search_vector_row_count_mismatch_raises(tmp_path: Path) -> None:
    vectors, hs_codes = _unit_vectors()
    embeddings_path = _write_fixture(
        tmp_path,
        stem="fixture_row_mismatch",
        vectors=vectors,
        hs_codes=hs_codes[:2],  # deliberately fewer codes than vector rows
    )

    with pytest.raises(EmbeddingsFileMismatchError):
        search_vector([1.0, 0.0, 0.0], top_k=3, embeddings_path=embeddings_path)


@pytest.mark.unit
def test_search_vector_meta_count_mismatch_raises(tmp_path: Path) -> None:
    vectors, hs_codes = _unit_vectors()
    embeddings_path = _write_fixture(
        tmp_path, stem="fixture_meta_count", vectors=vectors, hs_codes=hs_codes, count=999
    )

    with pytest.raises(EmbeddingsFileMismatchError):
        search_vector([1.0, 0.0, 0.0], top_k=3, embeddings_path=embeddings_path)


@pytest.mark.unit
def test_search_vector_meta_dims_mismatch_raises(tmp_path: Path) -> None:
    vectors, hs_codes = _unit_vectors()
    embeddings_path = _write_fixture(
        tmp_path, stem="fixture_meta_dims", vectors=vectors, hs_codes=hs_codes, dims=999
    )

    with pytest.raises(EmbeddingsFileMismatchError):
        search_vector([1.0, 0.0, 0.0], top_k=3, embeddings_path=embeddings_path)


@pytest.mark.unit
def test_search_vector_query_dimensionality_mismatch_raises(tmp_path: Path) -> None:
    vectors, hs_codes = _unit_vectors()
    embeddings_path = _write_fixture(
        tmp_path, stem="fixture_query_dims", vectors=vectors, hs_codes=hs_codes
    )

    # 2-dim query against a 3-dim corpus.
    with pytest.raises(EmbeddingsFileMismatchError):
        search_vector([1.0, 0.0], top_k=3, embeddings_path=embeddings_path)
