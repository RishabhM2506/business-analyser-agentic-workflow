"""Top-level orchestrator for free-text product search
(`POST /threads/{thread_id}/search`, `app/main.py`) — composes
`app.search.candidates` (find), `app.search.rerank` (rank + ground), and
`app.budget` into the three-outcome contract
(`auto_selected` / `disambiguate` / `no_candidates_found`) that
`app.schemas.response.ProductSearchResponse` exposes.

Deliberately returns a plain dataclass, not a schema instance — mirrors
`app.graph` returning `AnalysisState` (not a response schema) to
`app.main`, keeping this module free of any FastAPI/schema-layer
dependency. `app.main` is responsible for catching `BudgetExceededError`
(`app.budget`) and `UngroundedRerankError` (`app.search.rerank`) around the
call into this module — neither is caught here, matching
`app.nodes.describe_item`'s own "budget check happens immediately before
the spend it would fund" sequencing, made explicit rather than swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.budget import BudgetTracker
from app.models import ModelClient
from app.search.bm25 import search_bm25
from app.search.candidates import ProductSearchProvider
from app.search.normalize import normalize_query
from app.search.rerank import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_FLOOR,
    RankedCandidate,
    rerank_candidates,
)

SearchOutcome = Literal["auto_selected", "disambiguate", "no_candidates_found"]


@dataclass(frozen=True)
class ProductSearchResult:
    """Everything `app.main`'s route needs to build a `ProductSearchResponse`.

    `ranked_candidates` is already in final display order (descending
    `relevance_score`, computed in code by `rerank_candidates` — never the
    model's own list order) and empty for `no_candidates_found`, where no
    model call was ever made. `description_by_hs_code` lets the route
    attach each candidate's taxonomy description without a second lookup
    against `app.knowledge.provider`."""

    outcome: SearchOutcome
    selected_hs_code: str | None
    ranked_candidates: list[RankedCandidate]
    description_by_hs_code: dict[str, str]


async def search_products(
    query_text: str,
    *,
    thread_id: str,
    tenant_id: str,
    search_provider: ProductSearchProvider,
    model_client: ModelClient,
    budget_tracker: BudgetTracker,
    taxonomy_path: str = "data/harmonized-system.csv",
) -> ProductSearchResult:
    """Run the full find -> (normalize retry ->) (budget check ->) rerank ->
    threshold pipeline.

    Two independent "no real match" floors, not one — live smoke-testing
    (2026-08-20, real embeddings corpus + real Gemini) confirmed both are
    necessary:

    1. Pre-reranker, inside `search_provider.find_candidates` itself (see
       `app.search.candidates.HybridSearchProvider`'s module docstring) —
       an empty `candidates` list here short-circuits straight to
       `no_candidates_found` *before* the budget check or any model spend,
       which is what makes a nonsense query's cost genuinely zero rather
       than just "usually cheap."
    2. Post-reranker (`app.search.rerank.LOW_CONFIDENCE_FLOOR`, below) —
       real embedding-space geometry means an unrelated query's raw vector
       similarity can still clear the pre-reranker floor, reaching the
       reranker; if the model's own best score for what it was given is
       still this low, that is itself the signal nothing really matches,
       just discovered one step later and at the cost of one model call
       rather than zero.

    A third case, found live (2026-08-21, "posta dana" -> poppy seed,
    HS6 120791 not picked, meat codes shown instead): a query with zero
    lexical overlap against the taxonomy (BM25 empty) can still clear
    floor 1 above on vector similarity alone, but with a *noise-like*
    ranking rather than a genuine match (the correct code buried at rank
    623/5613; unrelated codes scoring higher from embedding-space
    artifacts). `search_bm25` is called here again (cheap, `@lru_cache`d,
    no I/O) purely as a free trigger for this case — not to duplicate
    `HybridSearchProvider`'s own internal BM25 call for real ranking
    purposes — deliberately kept out of the `ProductSearchProvider`
    Protocol so `HybridSearchProvider` doesn't need `budget_tracker`/
    `thread_id`/`tenant_id` threaded into its otherwise free, no-budget
    `find_candidates` contract.
    """
    candidates = await search_provider.find_candidates(query_text)
    # The query text actually used for retrieval - starts as the raw query,
    # becomes the normalized one below if normalization fires and changes
    # it. Must also be what's sent to the reranker (not the raw query):
    # live-reproduced (2026-08-21) - reranking retrieved-via-"poppy seeds"
    # candidates against the *original* "posta dana" search text made the
    # real model unable to relate the (untranslated) search text to the
    # (correctly retrieved) candidates at all, scoring every one 0.0. The
    # reranker must see the same terms retrieval actually used.
    effective_query_text = query_text

    if not search_bm25(query_text, top_k=1, taxonomy_path=taxonomy_path):
        # Budget-checked immediately before the call it funds (matches
        # `app.nodes.describe_item`'s sequencing) — this is a real model
        # spend, unlike `find_candidates` itself.
        await budget_tracker.check_and_increment(thread_id=thread_id, tenant_id=tenant_id)
        normalized = await normalize_query(query_text, model_client=model_client)
        if normalized != query_text.strip():
            # Full replacement, including replacement with empty: falling
            # back to the original (possibly noise-like) candidates here
            # would risk resurfacing exactly the bug this fixes. The
            # translated term finding nothing is a stronger "no real
            # match" signal than the untranslated term clearing the
            # vector-similarity floor by coincidence.
            candidates = await search_provider.find_candidates(normalized)
            effective_query_text = normalized

    if not candidates:
        return ProductSearchResult(
            outcome="no_candidates_found",
            selected_hs_code=None,
            ranked_candidates=[],
            description_by_hs_code={},
        )

    description_by_hs_code = {candidate.hs_code: candidate.description for candidate in candidates}

    # Checked immediately before the one model call this pipeline can make
    # (docs/PLAN.md §5.5: fails closed), not pre-checked earlier — finding
    # candidates itself is free (BM25 + one embedding call, not generative).
    await budget_tracker.check_and_increment(thread_id=thread_id, tenant_id=tenant_id)

    ranked_candidates = await rerank_candidates(
        effective_query_text, candidates, model_client=model_client
    )

    top = ranked_candidates[0]
    if top.relevance_score >= HIGH_CONFIDENCE_THRESHOLD:
        return ProductSearchResult(
            outcome="auto_selected",
            selected_hs_code=top.hs_code,
            ranked_candidates=ranked_candidates,
            description_by_hs_code=description_by_hs_code,
        )

    if top.relevance_score < LOW_CONFIDENCE_FLOOR:
        # The reranker's own best-scored candidate still isn't a real
        # match (see this function's docstring, point 2) — discard the
        # candidates entirely rather than returning `disambiguate` with a
        # list the model itself rated as not relevant; matches the
        # pre-reranker floor's identical empty-candidates shape so both
        # floors produce one consistent `no_candidates_found` contract.
        return ProductSearchResult(
            outcome="no_candidates_found",
            selected_hs_code=None,
            ranked_candidates=[],
            description_by_hs_code={},
        )

    return ProductSearchResult(
        outcome="disambiguate",
        selected_hs_code=None,
        ranked_candidates=ranked_candidates,
        description_by_hs_code=description_by_hs_code,
    )
