"""Query normalization: one `MODEL_UTILITY` call that translates/
standardizes a free-text product search query into standard English HS
trade terminology, before BM25/vector search runs on it.

Root cause this exists to fix (verified live, 2026-08-21, real Gemini +
real embeddings corpus): a query with zero lexical overlap against the
English taxonomy (e.g. "posta dana", Hindi for poppy seed) produces an
empty BM25 result and a *noise-like* vector-search ranking — the correct
code (120791) ranked 623rd of 5613, while unrelated codes (postage stamps,
bovine meat) scored higher purely from embedding-space artifacts. Search
`app.search.service.search_products` gates this call on that same "BM25
found nothing" signal, since translating an already-English query was
verified to be a near-no-op (not worth paying for on every search).

This output is never shown to the user (`ProductSearchResponse.query_text`
is sourced directly from the raw request in `app/main.py`, not from
anything returned here) and is only ever fed back into `search_bm25`/
`embed_query` as a plain search string — unlike `app.search.rerank`
(which can invent an HS6 code) or `app.nodes.describe_item` (which can
invent a number), there is nothing here for a code-level grounding
guardrail to check.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.models import ModelClient

PROMPT_VERSION = "normalize_query-v1"
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "normalize_query.md"


class NormalizedQuery(BaseModel):
    """Schema-constrained structured output for the MODEL_UTILITY
    normalization call. Same length bound as
    `app.schemas.query.ProductSearchQuery.query_text` — this is a rewrite
    of that same field, not a new kind of input."""

    model_config = ConfigDict(extra="forbid")

    normalized_query: str = Field(min_length=1, max_length=200)


def _load_system_prompt() -> str:
    """Read `prompts/normalize_query.md` and strip its leading HTML-comment
    block, matching `app.nodes.describe_item._load_system_prompt` and
    `app.search.rerank._load_system_prompt` exactly."""
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    if text.startswith("<!--"):
        end = text.find("-->")
        if end != -1:
            text = text[end + 3 :]
    return text.strip()


async def normalize_query(query_text: str, *, model_client: ModelClient) -> str:
    """Return `query_text` rewritten as standard English HS trade
    terminology, or `query_text` itself (verified: near-unchanged) when it
    already is."""
    result = await model_client.generate_structured(
        system_prompt=_load_system_prompt(),
        user_content=query_text,
        schema=NormalizedQuery,
    )
    return result.normalized_query.strip()
