"""Loader for the offline-precomputed HS6 taxonomy embeddings + brute-force
cosine similarity search over them.

No vector DB / pgvector (docs/PLAN.md's rollout section for this feature):
real operational overhead (a second datastore, index builds, migrations) to
save microseconds on a query that's already faster than the network
round-trip needed to fetch the query embedding itself. Verified math: one
`(5613, 3072) @ (3072,)` dot product is ~34.5M FLOPs, sub-5ms wall clock on
any modern CPU via numpy's BLAS backend — this corpus is small and fixed
(only changes when the taxonomy is re-vendored and
`scripts/embed_taxonomy.py` is re-run), exactly the case where a real vector
index is pure overhead.

The three files loaded here are produced once, offline, by
`scripts/embed_taxonomy.py` (never at request time, never in CI/Docker
build) and committed to `data/`:

    data/hs_taxonomy_embeddings.npy           float32 (N, D), L2-normalized
    data/hs_taxonomy_embeddings.hscodes.txt   N lines, row i <-> hs_code i
    data/hs_taxonomy_embeddings.meta.json     model, task_type, dims, count,
                                               source_csv_sha256, generated_at
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EMBEDDINGS_STEM = "data/hs_taxonomy_embeddings"


def _resolve_path(path_str: str) -> Path:
    """Same repo-root-relative resolution convention as
    `app.knowledge.provider._resolve_path` — robust regardless of the
    process's working directory (pytest, docker WORKDIR, ...)."""
    path = Path(path_str)
    return path if path.is_absolute() else _REPO_ROOT / path


@dataclass(frozen=True)
class VectorIndex:
    """Precomputed, L2-normalized corpus embeddings plus their row<->hs_code
    mapping. `vectors` shape is `(num_docs, dims)`; row `i` corresponds to
    `hs_codes[i]`."""

    hs_codes: list[str]
    vectors: np.ndarray
    dims: int


class EmbeddingsFileMismatchError(RuntimeError):
    """Raised when the three embeddings files don't agree with each other
    (row counts, dimensionality) — fails loudly rather than silently
    truncating/misaligning `hs_codes[i]` against `vectors[i]`."""


@lru_cache(maxsize=2)
def _load_vector_index(embeddings_stem: str) -> VectorIndex:
    """Cached, lazy on first use (matches
    `app.knowledge.provider._load_taxonomy`'s exact pattern) — loading a
    ~69MB `.npy` is not something to redo per request, but it's also not
    worth an eager warm-up in the app's `lifespan` for a feature that may
    never be called in a given process lifetime."""
    base = _resolve_path(embeddings_stem)
    npy_path = base.with_suffix(".npy")
    hscodes_path = base.parent / f"{base.name}.hscodes.txt"
    meta_path = base.parent / f"{base.name}.meta.json"

    vectors = np.load(npy_path)
    hs_codes = hscodes_path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if vectors.shape[0] != len(hs_codes):
        raise EmbeddingsFileMismatchError(
            f"{npy_path.name} has {vectors.shape[0]} rows but {hscodes_path.name} "
            f"has {len(hs_codes)} lines — row<->hs_code alignment cannot be trusted"
        )
    if int(meta.get("count", -1)) != len(hs_codes):
        raise EmbeddingsFileMismatchError(
            f"{meta_path.name}'s count ({meta.get('count')!r}) does not match "
            f"{hscodes_path.name}'s {len(hs_codes)} lines"
        )
    if int(meta.get("dims", -1)) != vectors.shape[1]:
        raise EmbeddingsFileMismatchError(
            f"{meta_path.name}'s dims ({meta.get('dims')!r}) does not match "
            f"{npy_path.name}'s actual dimensionality {vectors.shape[1]}"
        )

    return VectorIndex(hs_codes=hs_codes, vectors=vectors, dims=vectors.shape[1])


def search_vector(
    query_vector: list[float],
    *,
    top_k: int,
    embeddings_path: str = _DEFAULT_EMBEDDINGS_STEM,
) -> list[tuple[str, float]]:
    """Return up to `top_k` `(hs_code, cosine_similarity)` pairs for
    `query_vector`, ranked descending. `query_vector` need not be
    pre-normalized — normalized here so callers only need to hand over
    whatever `EmbeddingsClient.embed_query` returned."""
    index = _load_vector_index(embeddings_path)
    if len(query_vector) != index.dims:
        raise EmbeddingsFileMismatchError(
            f"query vector has {len(query_vector)} dims but the loaded corpus "
            f"index has {index.dims} — likely an embeddings-model mismatch "
            f"between the query embedder and {embeddings_path}"
        )

    query = np.asarray(query_vector, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm > 0:
        query = query / query_norm

    similarities = index.vectors @ query
    if top_k >= len(index.hs_codes):
        top_indices = np.argsort(-similarities)
    else:
        # argpartition is O(n) vs argsort's O(n log n); only the top_k need
        # to end up sorted, not the full 5,613-row ranking.
        top_indices = np.argpartition(-similarities, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

    return [(index.hs_codes[i], float(similarities[i])) for i in top_indices[:top_k]]
