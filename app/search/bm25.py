"""Hand-rolled Okapi BM25 over the HS6 taxonomy descriptions.

Not a dependency: `rank-bm25` (PyPI) last released 2022-02-16 — effectively
unmaintained, and this project's own `dep-audit` (pip-audit) CI gate plus
its habit of citing exact verified dependency behavior argue against taking
on an unmaintained package for a ~40-year-old, extremely well-specified
formula. `bm25s` is actively maintained but its value (Numba/JAX-accelerated
indexing, memory-mapped corpora) targets corpora many orders of magnitude
larger than this one — 5,613 fixed, short taxonomy strings is exactly the
case where a general-purpose IR library's flexibility is pure overhead.
This corpus is also uniquely favorable to hand-rolling: fixed (only changes
when the taxonomy CSV is re-vendored), small, and short (median description
a few words) — no stemming/stopword list for v1, simplest thing that could
plausibly work; revisit only if real queries expose a specific gap.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from app.knowledge.provider import TaxonomyEntry, get_hs6_taxonomy_entries

# Classic Robertson/Sparck-Jones defaults — not tuned to this corpus yet
# (see docs/PLAN.md's rollout section: "revisit only if real queries expose
# a specific gap"), but these are the standard, well-documented starting
# values used across essentially every real-world BM25 deployment.
_K1 = 1.5
_B = 0.75

_TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + `\\w+` split — no stemming, no stopwords (see module
    docstring). Deliberately the simplest tokenizer that could work for
    short, technical taxonomy descriptions."""
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class Bm25Index:
    """Precomputed BM25 index over the HS6-level taxonomy rows."""

    hs_codes: list[str]  # doc id -> hs_code, row order fixed at build time
    doc_term_freqs: list[Counter[str]]  # doc id -> {term: count in that doc}
    doc_lengths: list[int]  # doc id -> token count
    avg_doc_length: float
    doc_freq: dict[str, int]  # term -> number of docs containing it
    num_docs: int


def _build_index(entries: list[TaxonomyEntry]) -> Bm25Index:
    hs_codes: list[str] = []
    doc_term_freqs: list[Counter[str]] = []
    doc_lengths: list[int] = []
    doc_freq: Counter[str] = Counter()

    for entry in entries:
        tokens = _tokenize(entry.description)
        term_freq = Counter(tokens)
        hs_codes.append(entry.hs_code)
        doc_term_freqs.append(term_freq)
        doc_lengths.append(len(tokens))
        doc_freq.update(term_freq.keys())  # each term counted once per doc

    num_docs = len(hs_codes)
    avg_doc_length = sum(doc_lengths) / num_docs if num_docs else 0.0
    return Bm25Index(
        hs_codes=hs_codes,
        doc_term_freqs=doc_term_freqs,
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
        doc_freq=dict(doc_freq),
        num_docs=num_docs,
    )


@lru_cache(maxsize=2)
def _build_bm25_index(taxonomy_path: str) -> Bm25Index:
    """Cached, lazy on first use (matches
    `app.knowledge.provider._load_taxonomy`'s exact `@lru_cache` pattern,
    keyed on the path string here since `get_hs6_taxonomy_entries` already
    handles its own repo-root-relative resolution) — tokenizing and
    indexing 5,613 short strings is tens of milliseconds, one-time, not
    worth an eager warm-up in the app's `lifespan`."""
    return _build_index(get_hs6_taxonomy_entries(taxonomy_path=taxonomy_path))


def _idf(index: Bm25Index, term: str) -> float:
    n_t = index.doc_freq.get(term, 0)
    # Lucene/BM25+-style IDF: never negative, unlike the classic Robertson
    # formula, which can go negative for terms present in >50% of documents.
    return math.log((index.num_docs - n_t + 0.5) / (n_t + 0.5) + 1)


def _score(index: Bm25Index, doc_id: int, query_terms: list[str]) -> float:
    term_freq = index.doc_term_freqs[doc_id]
    doc_length = index.doc_lengths[doc_id]
    score = 0.0
    for term in query_terms:
        f_td = term_freq.get(term)
        if not f_td:
            continue
        idf = _idf(index, term)
        numerator = f_td * (_K1 + 1)
        denominator = f_td + _K1 * (1 - _B + _B * doc_length / index.avg_doc_length)
        score += idf * numerator / denominator
    return score


def search_bm25(
    query_text: str, *, top_k: int, taxonomy_path: str = "data/harmonized-system.csv"
) -> list[tuple[str, float]]:
    """Return up to `top_k` `(hs_code, bm25_score)` pairs for `query_text`,
    ranked descending, restricted to docs scoring > 0 (a doc with zero
    query-term overlap is not a match, not a low-relevance match)."""
    index = _build_bm25_index(taxonomy_path)
    query_terms = _tokenize(query_text)
    if not query_terms:
        return []

    scores: list[tuple[str, float]] = []
    for doc_id, hs_code in enumerate(index.hs_codes):
        score = _score(index, doc_id, query_terms)
        if score > 0:
            scores.append((hs_code, score))

    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores[:top_k]
