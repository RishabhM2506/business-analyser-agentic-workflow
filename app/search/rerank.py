"""LLM reranker for free-text product search candidates
(`app.search.candidates.HybridSearchProvider`'s output) — the second half
of the search feature. Structured-output schema, prompt loading, and the
code-level grounding guard that makes it structurally impossible for the
model to return an HS6 code that search itself didn't find (the
code-identity equivalent of `app.guardrails`' number-grounding check,
docs/PLAN.md's rollout section: "the LLM never produces a number/fact it
wasn't given," now applied to code identity).
"""

from __future__ import annotations

from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.guardrails import check_hs_codes_grounded, find_ungrounded_hs_codes
from app.models import ModelClient
from app.search.candidates import SearchCandidate

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

PROMPT_VERSION = "rerank_hs_code-v1"
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "rerank_hs_code.md"

# Auto-select the top-scoring candidate (sorted by score in code below,
# never trusting the model's own list order) only above this score.
# Conservative on purpose (flagged must-verify in docs/PLAN.md — no
# measured score distribution exists yet): auto-select skips straight into
# the trade-data pipeline with no confirmation, while disambiguation costs
# the user one extra click — "ask when in doubt" is the safer default.
HIGH_CONFIDENCE_THRESHOLD = 0.75

# Live-reproduced 2026-08-20 (real Gemini + real embeddings corpus, first
# genuine end-to-end smoke test): a deliberately nonsense query
# ("zzzqqqxxx asdkjhaskjdh nonsense gibberish") was NOT caught by
# `app.search.candidates`' pre-reranker floor (real embedding-space
# geometry means an unrelated query's top raw cosine similarity can still
# exceed that floor's 0.35 — embedding spaces are not uniformly "far apart"
# for unrelated text the way that floor's threshold assumes), so it reached
# the reranker, which correctly scored every single candidate 0.0 — but
# `search_products` had no rule for "the reranker's own judgment says
# nothing matches," so it still returned `disambiguate` with 8 candidates
# the model itself rated 0% relevant. This floor is the missing symmetric
# check on the *reranked* scores, catching exactly this case regardless of
# whether the pre-reranker floor did: if even the best candidate scores
# below this, the reranker is telling us nothing in the candidate set is a
# real match, and that's `no_candidates_found`, not a disambiguation
# prompt full of junk. Same constant value as `HybridSearchProvider`'s own
# floor for consistency, not because the two are mechanically related.
LOW_CONFIDENCE_FLOOR = 0.35


class RankedCandidate(BaseModel):
    """One reranked candidate exactly as returned by the model — internal
    to `app.search`; `app.schemas.response.RankedCandidateOut` is the
    public, display-ready shape assembled from this plus each candidate's
    taxonomy description (the model is never asked to echo the description
    back, only the code and a score)."""

    model_config = ConfigDict(extra="forbid")

    hs_code: str = Field(pattern=r"^\d{6}$")
    relevance_score: float = Field(ge=0.0, le=1.0)


class RerankOutput(BaseModel):
    """Schema-constrained structured output for the MODEL_UTILITY rerank
    call (docs/PLAN.md §2.2: schema-constrained, not free text parsed
    later)."""

    model_config = ConfigDict(extra="forbid")

    ranked_candidates: list[RankedCandidate] = Field(min_length=1, max_length=8)


class UngroundedRerankError(Exception):
    """Raised when the model returns an HS6 code that was not among the
    candidates it was given — the reranker inventing a code, the exact
    failure mode `check_hs_codes_grounded` exists to catch. Mapped to
    `RERANK_INVALID_CANDIDATE` (502) at the API edge (`app/main.py`)."""

    def __init__(self, ungrounded_codes: list[str]) -> None:
        self.ungrounded_codes = ungrounded_codes
        super().__init__(f"reranker returned code(s) not in the candidate set: {ungrounded_codes}")


def _load_system_prompt() -> str:
    """Read `prompts/rerank_hs_code.md` and strip its leading HTML-comment
    block, matching `app.nodes.describe_item._load_system_prompt` exactly."""
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    if text.startswith("<!--"):
        end = text.find("-->")
        if end != -1:
            text = text[end + 3 :]
    return text.strip()


def _build_user_content(query_text: str, candidates: list[SearchCandidate]) -> str:
    lines = [f'Search text: "{query_text}"', "", "Candidates:"]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(f"{index}. {candidate.hs_code}: {candidate.description}")
    return "\n".join(lines)


async def rerank_candidates(
    query_text: str,
    candidates: list[SearchCandidate],
    *,
    model_client: ModelClient,
) -> list[RankedCandidate]:
    """Rerank `candidates` (already found by `app.search.candidates`) via
    one MODEL_UTILITY call, returned sorted descending by
    `relevance_score` (computed in code, never trusting the model's own
    list order).

    Raises `UngroundedRerankError` — never silently drops or substitutes —
    if the model returns any code outside the candidate set. Callers are
    responsible for budget-checking before calling this, mirroring
    `app.nodes.describe_item`'s sequencing: the budget check happens
    immediately before the spend it would fund, not after.
    """
    candidate_codes = {candidate.hs_code for candidate in candidates}
    result = await model_client.generate_structured(
        system_prompt=_load_system_prompt(),
        user_content=_build_user_content(query_text, candidates),
        schema=RerankOutput,
    )

    returned_codes = [ranked.hs_code for ranked in result.ranked_candidates]
    if not check_hs_codes_grounded(returned_codes, candidate_codes):
        ungrounded = find_ungrounded_hs_codes(returned_codes, candidate_codes)
        logger.warning(
            "rerank.guardrail_rejected",
            query_text=query_text,
            candidate_codes=sorted(candidate_codes),
            ungrounded_codes=ungrounded,
        )
        raise UngroundedRerankError(ungrounded)

    return sorted(result.ranked_candidates, key=lambda ranked: ranked.relevance_score, reverse=True)
