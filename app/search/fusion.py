"""Reciprocal Rank Fusion (RRF) — combines the BM25 and vector-search
rankings into one ranked list without needing to calibrate their
incompatible score scales (a BM25 score and a cosine similarity are not on
the same axis and averaging them directly would be meaningless).

Standard formula, `k=60` — the original Cormack et al. (2009) default, also
Elasticsearch's own built-in RRF default: `score(doc) = sum(1 / (k + rank))`
over every ranking the doc appears in, 1-indexed rank. A doc missing from
one of the two input rankings simply doesn't contribute that term — it is
not penalized beyond not getting the bonus.
"""

from __future__ import annotations

_RRF_K = 60


def reciprocal_rank_fusion(
    *rankings: list[tuple[str, float]], top_k: int
) -> list[tuple[str, float]]:
    """Fuse any number of `(hs_code, score)` rankings (score only used to
    determine each ranking's own relative order — RRF itself only looks at
    rank position, not the score magnitude) into one `(hs_code, rrf_score)`
    list, descending, truncated to `top_k`."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (hs_code, _score) in enumerate(ranking, start=1):
            fused[hs_code] = fused.get(hs_code, 0.0) + 1.0 / (_RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
    return ordered[:top_k]
