"""`ProductSearchProvider` Protocol + `HybridSearchProvider` (v1 impl) — the
free-text-to-candidate-codes half of the search feature. `app.search.rerank`
handles the second half (candidates -> a selection).

Mirrors `app.knowledge.provider.KnowledgeProvider`'s shape (a `Protocol`,
one v1 impl, no vector DB) but is the inverse operation — deliberately not
merged with it (see `app/search/__init__.py`'s module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.knowledge.provider import get_hs6_taxonomy_entries
from app.search.bm25 import search_bm25
from app.search.embeddings import EmbeddingsClient
from app.search.fusion import reciprocal_rank_fusion
from app.search.vector_index import search_vector

_BM25_TOP_K = 30
_VECTOR_TOP_K = 30
_FUSED_TOP_K = 8

# Pre-reranker, code-level "no real match" floor (docs/PLAN.md's rollout
# section, flagged must-verify — no measured score distribution exists yet):
# BM25 found literally zero term-overlapping docs *and* the single closest
# vector match is below this raw cosine similarity -> nothing here is worth
# spending a model call to rerank. Lives here (not in `app.search.rerank`)
# because it needs the two *raw*, pre-fusion signals, which only this
# module computes; RRF's own fused score has no comparable "zero signal"
# reading once even one ranking contributes.
_MIN_VECTOR_SIMILARITY_FLOOR = 0.35


@dataclass(frozen=True)
class SearchCandidate:
    """One fused candidate: an HS6 code, its taxonomy description (for
    display and for the reranker prompt), and its RRF fusion score."""

    hs_code: str
    description: str
    fusion_score: float


class ProductSearchProvider(Protocol):
    """Interface every free-text-search implementation must satisfy."""

    async def find_candidates(
        self, query_text: str, *, top_k: int = _FUSED_TOP_K
    ) -> list[SearchCandidate]: ...


@lru_cache(maxsize=2)
def _hs6_description_lookup(taxonomy_path: str) -> dict[str, str]:
    """`hs_code -> description` for every HS6 row, cached the same way
    `app.knowledge.provider._load_taxonomy` caches the parsed CSV itself —
    this is just a re-keyed view over that same cached data, not a second
    file read."""
    entries = get_hs6_taxonomy_entries(taxonomy_path=taxonomy_path)
    return {entry.hs_code: entry.description for entry in entries}


class HybridSearchProvider:
    """v1 `ProductSearchProvider`: BM25 + vector search, fused via
    reciprocal rank fusion. No vector DB (see
    `app.search.vector_index`'s module docstring for the verified-sub-5ms
    reasoning) — two fixed-corpus, in-process lookups, not a distributed
    system."""

    def __init__(
        self,
        *,
        embeddings_client: EmbeddingsClient,
        taxonomy_path: str = "data/harmonized-system.csv",
        embeddings_path: str = "data/hs_taxonomy_embeddings",
    ) -> None:
        self._embeddings_client = embeddings_client
        self._taxonomy_path = taxonomy_path
        self._embeddings_path = embeddings_path

    async def find_candidates(
        self, query_text: str, *, top_k: int = _FUSED_TOP_K
    ) -> list[SearchCandidate]:
        bm25_results = search_bm25(query_text, top_k=_BM25_TOP_K, taxonomy_path=self._taxonomy_path)

        query_vector = await self._embeddings_client.embed_query(query_text)
        vector_results = search_vector(
            query_vector, top_k=_VECTOR_TOP_K, embeddings_path=self._embeddings_path
        )

        top_vector_similarity = vector_results[0][1] if vector_results else 0.0
        if not bm25_results and top_vector_similarity < _MIN_VECTOR_SIMILARITY_FLOOR:
            return []

        fused = reciprocal_rank_fusion(bm25_results, vector_results, top_k=top_k)
        descriptions = _hs6_description_lookup(self._taxonomy_path)
        return [
            SearchCandidate(
                hs_code=hs_code, description=descriptions.get(hs_code, ""), fusion_score=score
            )
            for hs_code, score in fused
        ]
