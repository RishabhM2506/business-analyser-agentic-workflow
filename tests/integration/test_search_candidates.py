"""Integration tests for `app.search.candidates.HybridSearchProvider` — BM25
+ vector + RRF fusion under `MockEmbeddingsClient`. Uses the real, checked-in
`data/harmonized-system.csv` taxonomy for BM25 (small, fast, already the
shared corpus every other search-module test uses) but a small synthetic
`.npy`/`.hscodes.txt`/`.meta.json` fixture for the vector side, so this
suite never depends on `scripts/embed_taxonomy.py` having been run in a
given checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.search.candidates import HybridSearchProvider
from app.search.embeddings import MockEmbeddingsClient

_DIMS = 8


async def _write_vector_fixture(tmp_path: Path, *, hs_codes: list[str]) -> str:
    """A synthetic corpus whose vectors are the *same* deterministic
    `MockEmbeddingsClient` vectors as each code's own taxonomy description
    would produce — this makes vector search meaningfully exercisable
    (non-trivial similarity structure) without depending on real semantic
    embeddings, matching `MockEmbeddingsClient`'s own "deterministic, not
    semantically meaningful" contract."""
    mock_client = MockEmbeddingsClient(dimensions=_DIMS)
    embedded = await mock_client.embed_documents(
        [f"synthetic description for {code}" for code in hs_codes]
    )
    vectors = np.array(embedded, dtype=np.float32)

    stem = tmp_path / "fixture_embeddings"
    np.save(stem.with_suffix(".npy"), vectors)
    (tmp_path / "fixture_embeddings.hscodes.txt").write_text("\n".join(hs_codes), encoding="utf-8")
    (tmp_path / "fixture_embeddings.meta.json").write_text(
        json.dumps({"count": len(hs_codes), "dims": _DIMS}), encoding="utf-8"
    )
    return str(stem)


@pytest.fixture
async def coffee_vector_fixture(tmp_path: Path) -> str:
    # A handful of real HS6 codes (coffee-family plus a few unrelated ones)
    # so BM25's real term-overlap signal on "coffee" is meaningful even
    # though the vector side is non-semantic mock noise.
    hs_codes = ["090111", "090112", "090121", "090190", "010121", "271012"]
    return await _write_vector_fixture(tmp_path, hs_codes=hs_codes)


@pytest.mark.integration
async def test_find_candidates_returns_fused_results_for_coffee_query(
    coffee_vector_fixture: str,
) -> None:
    provider = HybridSearchProvider(
        embeddings_client=MockEmbeddingsClient(dimensions=_DIMS),
        embeddings_path=coffee_vector_fixture,
    )

    results = await provider.find_candidates("coffee", top_k=8)

    hs_codes = [c.hs_code for c in results]
    # BM25 must surface the real coffee codes even though the vector side is
    # non-semantic noise - the fused list should still contain them.
    assert "090111" in hs_codes
    assert all(isinstance(c.description, str) and c.description for c in results)


@pytest.mark.integration
async def test_find_candidates_results_sorted_descending_by_fusion_score(
    coffee_vector_fixture: str,
) -> None:
    provider = HybridSearchProvider(
        embeddings_client=MockEmbeddingsClient(dimensions=_DIMS),
        embeddings_path=coffee_vector_fixture,
    )

    results = await provider.find_candidates("coffee", top_k=8)

    scores = [c.fusion_score for c in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
async def test_find_candidates_respects_top_k(coffee_vector_fixture: str) -> None:
    provider = HybridSearchProvider(
        embeddings_client=MockEmbeddingsClient(dimensions=_DIMS),
        embeddings_path=coffee_vector_fixture,
    )

    results = await provider.find_candidates("coffee", top_k=2)

    assert len(results) <= 2


@pytest.mark.integration
async def test_find_candidates_no_bm25_hit_and_low_vector_similarity_returns_empty(
    tmp_path: Path,
) -> None:
    """The pre-reranker "no real match" floor: a query with zero BM25
    overlap and (engineered here) a below-floor top vector similarity must
    short-circuit to an empty candidate list *before* any reranking would
    ever be attempted - this is what makes a nonsense query's cost
    genuinely zero (`app.search.candidates`'s own module docstring)."""
    hs_codes = ["090111", "090112"]
    # Vectors deliberately orthogonal to whatever the mock query embedding
    # will be, by construction: same trick as test_vector_index's synthetic
    # fixtures, just wired through the full HybridSearchProvider this time.
    vectors = np.zeros((2, _DIMS), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    stem = tmp_path / "fixture_orthogonal"
    np.save(stem.with_suffix(".npy"), vectors)
    (tmp_path / "fixture_orthogonal.hscodes.txt").write_text("\n".join(hs_codes), encoding="utf-8")
    (tmp_path / "fixture_orthogonal.meta.json").write_text(
        json.dumps({"count": len(hs_codes), "dims": _DIMS}), encoding="utf-8"
    )

    class _OrthogonalQueryEmbeddingsClient(MockEmbeddingsClient):
        async def embed_query(self, text: str) -> list[float]:
            # Orthogonal to both fixture rows above -> cosine similarity 0.0
            # for everything, well below the floor.
            vector = [0.0] * _DIMS
            vector[_DIMS - 1] = 1.0
            return vector

    provider = HybridSearchProvider(
        embeddings_client=_OrthogonalQueryEmbeddingsClient(dimensions=_DIMS),
        embeddings_path=str(stem),
    )

    results = await provider.find_candidates("zzzqqqxxx nonsense", top_k=8)

    assert results == []


@pytest.mark.integration
async def test_find_candidates_rejects_a_high_but_flat_vector_ranking(tmp_path: Path) -> None:
    """Regression for a real, measured false positive in the old static
    0.35 absolute-score floor: real gibberish queries against the actual
    embedding corpus scored 0.55-0.61 top-1 cosine similarity (comfortably
    clearing 0.35) because of embedding-space "hubness" — a handful of
    vectors sit unusually close to everything. Engineered here: every
    candidate scores the *same* high similarity (0.6, well above the old
    0.35 floor) — a flat ranking with no real standout, exactly the shape
    that broke the old floor. The new margin-based floor must reject it."""
    hs_codes = ["090111", "090112", "090121", "090190", "010121", "271012"]
    # All six vectors identical -> every one scores exactly the same
    # similarity against any query, however high - zero margin.
    vectors = np.zeros((6, _DIMS), dtype=np.float32)
    vectors[:, 0] = 1.0
    stem = tmp_path / "fixture_flat"
    np.save(stem.with_suffix(".npy"), vectors)
    (tmp_path / "fixture_flat.hscodes.txt").write_text("\n".join(hs_codes), encoding="utf-8")
    (tmp_path / "fixture_flat.meta.json").write_text(
        json.dumps({"count": len(hs_codes), "dims": _DIMS}), encoding="utf-8"
    )

    class _MatchingQueryEmbeddingsClient(MockEmbeddingsClient):
        async def embed_query(self, text: str) -> list[float]:
            # Identical direction to every fixture row -> a real 0.6+
            # cosine similarity, same shape as the live-measured gibberish
            # scores this test reproduces.
            vector = [0.0] * _DIMS
            vector[0] = 0.6
            vector[1] = 0.8  # keeps the vector non-degenerate (unit-ish norm)
            return vector

    provider = HybridSearchProvider(
        embeddings_client=_MatchingQueryEmbeddingsClient(dimensions=_DIMS),
        embeddings_path=str(stem),
    )

    results = await provider.find_candidates("zzxxqqbbvvnn flibbertigibbet", top_k=8)

    assert results == []


@pytest.mark.integration
async def test_find_candidates_accepts_a_vector_hit_that_stands_out_from_its_neighbors(
    tmp_path: Path,
) -> None:
    """The margin floor's other side: one real standout hit, well above
    its neighbors, must still pass through to the reranker even though its
    own absolute score is unremarkable — the point of the fix is
    discriminating by shape, not by raising the bar."""
    hs_codes = ["090111", "090112", "090121", "090190", "010121", "271012"]
    vectors = np.zeros((6, _DIMS), dtype=np.float32)
    vectors[0, 0] = 1.0  # the one real standout
    for i in range(1, 6):
        vectors[i, 1] = 1.0  # everything else orthogonal to the query
    stem = tmp_path / "fixture_standout"
    np.save(stem.with_suffix(".npy"), vectors)
    (tmp_path / "fixture_standout.hscodes.txt").write_text("\n".join(hs_codes), encoding="utf-8")
    (tmp_path / "fixture_standout.meta.json").write_text(
        json.dumps({"count": len(hs_codes), "dims": _DIMS}), encoding="utf-8"
    )

    class _StandoutQueryEmbeddingsClient(MockEmbeddingsClient):
        async def embed_query(self, text: str) -> list[float]:
            vector = [0.0] * _DIMS
            vector[0] = 1.0  # matches only the first fixture row
            return vector

    provider = HybridSearchProvider(
        embeddings_client=_StandoutQueryEmbeddingsClient(dimensions=_DIMS),
        embeddings_path=str(stem),
    )

    results = await provider.find_candidates("qqzzxxwwvvnn blorptastic", top_k=8)

    assert any(c.hs_code == "090111" for c in results)
