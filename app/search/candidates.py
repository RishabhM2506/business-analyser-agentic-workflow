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

# Pre-reranker, code-level "no real match" floor for the BM25-empty case
# (BM25 lives here — not in `app.search.rerank` — because this check needs
# the two *raw*, pre-fusion signals, which only this module computes; RRF's
# own fused score has no comparable "zero signal" reading once even one
# ranking contributes).
#
# A margin between the top hit and the rest of the ranking, not a static
# absolute score (2026-09-02 fix, replacing the original 0.35 absolute
# floor — real, measured evidence this floor was live-broken: genuine
# gibberish queries against the real 3072-dim embedding corpus scored
# 0.55-0.61 top-1 cosine similarity ("zzzqqqxxx nonsense gibberish": 0.5869,
# "asdfghjkl": 0.6149, "XQ-99": 0.5592) — comfortably *above* 0.35, so the
# floor essentially never triggered for realistic nonsense, an instance of
# high-dimensional embedding-space "hubness" (a handful of vectors sit
# unusually close to *everything*, unrelated queries included). The
# discriminating signal turned out to be the *margin* between the top hit
# and the rest of the ranking, not its absolute value: the same real
# queries measured margin (top-1 minus the mean of ranks 2-6) of only
# 0.0064-0.0133 for nonsense, versus 0.0433-0.0822 for genuine matches
# ("poppy seeds", "green coffee beans", "mango") — nonsense scores high but
# *flat* (every candidate looks equally plausible/implausible), a real
# match stands out from its own neighbors. Verified this does not regress
# the vernacular-query fix ("posta dana"): its own pre-translation margin
# (0.0041) is *also* low, correctly failing this floor exactly like
# nonsense would — but `app.search.service.search_products`'s separate
# BM25-empty-triggered translation retry still fires independently and
# supersedes this result, so "posta dana" -> "poppy seeds" -> the correct
# code is unaffected. Threshold picked with real headroom on both sides of
# the two measured clusters — flagged, not exhaustively researched, same
# honesty convention as every other reasoned-not-verified threshold in this
# pipeline (the old 0.35 value, the 60% HHI concentration threshold).
_MIN_VECTOR_MARGIN = 0.02
# Backstop only: used when there are too few vector results to compute any
# margin against (fewer than 2 total — never happens against the real
# 5,613-code corpus, `_VECTOR_TOP_K=30` always returns 30, but a small test
# fixture or a corpus this small could hit it).
_MIN_VECTOR_SIMILARITY_FLOOR = 0.35
# How many of the next-best results to average against the top hit when
# computing the margin.
_MARGIN_COMPARISON_WIDTH = 5


def _has_sufficient_vector_signal(vector_results: list[tuple[str, float]]) -> bool:
    """True iff the top vector hit stands out enough from its own
    neighbors to be worth reranking — see `_MIN_VECTOR_MARGIN`'s own
    comment for the real, measured evidence behind this replacing a static
    absolute-score floor."""
    if not vector_results:
        return False
    top1 = vector_results[0][1]
    rest = vector_results[1 : 1 + _MARGIN_COMPARISON_WIDTH]
    if not rest:
        return top1 >= _MIN_VECTOR_SIMILARITY_FLOOR
    rest_mean = sum(score for _, score in rest) / len(rest)
    return (top1 - rest_mean) >= _MIN_VECTOR_MARGIN


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

        if not bm25_results and not _has_sufficient_vector_signal(vector_results):
            return []

        fused = reciprocal_rank_fusion(bm25_results, vector_results, top_k=top_k)
        descriptions = _hs6_description_lookup(self._taxonomy_path)
        return [
            SearchCandidate(
                hs_code=hs_code, description=descriptions.get(hs_code, ""), fusion_score=score
            )
            for hs_code, score in fused
        ]
