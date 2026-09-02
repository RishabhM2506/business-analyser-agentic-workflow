"""Decides whether the domain-specific agriculture sources (Agmarknet,
MSP, FAOSTAT production) are relevant to a given HS6 commodity —
user-directed (2026-08-25): "today poppy seed, tomorrow cement" — a
source must not be silently queried, or reported as `NOT_FOUND`, for a
commodity it categorically doesn't apply to. `NOT_FOUND` means "this
source is relevant but has no data" (e.g. Agmarknet's real, confirmed
lack of poppy-seed coverage); a commodity these sources don't apply to at
all gets `NOT_APPLICABLE` instead (`app.report.facts`'s own concern, not
this module's — this module only answers yes/no).

**Hybrid design, per the user's own explicit choice** over a pure rule or
a pure per-query model call:

- HS chapters 01-24 (Sections I-IV of the HS nomenclature: live animals,
  vegetable products, animal/vegetable fats, prepared foodstuffs/
  beverages/tobacco) are unambiguously agriculture/food — `RELEVANT`, no
  model call, no cost.
- A small set of **boundary chapters** (50-53: silk, wool/animal hair,
  cotton, other vegetable textile fibres) are real agricultural raw
  materials *outside* 01-24 that this pipeline's own MSP data already
  confirms genuine coverage for (a real row: `"Cotton (Medium Staple)"`,
  HS chapter 52) — a fixed chapter rule can't safely resolve these either
  way, so exactly this narrow set triggers one real `MODEL_UTILITY` call.
  **Flagged, not exhaustively researched** — the same honesty convention
  as this pipeline's other reasoned-not-verified thresholds (the 0.35
  vector-similarity floor, the 60% HHI concentration threshold).
- Every other chapter (the overwhelming majority — industrial goods,
  machinery, electronics, chemicals, etc.) is `NOT_APPLICABLE`, no model
  call — this must stay free and instant, matching this pipeline's
  existing "a well-formed, clearly out-of-scope query costs nothing"
  design goal (`app.search.service`'s own stated principle, applied here
  to a different decision).

Budget-checked only immediately before the one real call this can make
(the boundary-chapter case), matching `app.search.service.search_products`'s
identical sequencing.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.budget import BudgetTracker
from app.models import ModelClient

PROMPT_VERSION = "agriculture_relevance-v1"
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "agriculture_relevance.md"

# Sections I-IV of the HS nomenclature - unambiguously agriculture/food.
AGRICULTURE_CHAPTERS = frozenset(f"{i:02d}" for i in range(1, 25))

# Real agricultural raw materials outside chapters 01-24 - flagged, not
# exhaustively researched. See module docstring.
BOUNDARY_CHAPTERS = frozenset({"50", "51", "52", "53"})


class AgricultureRelevanceCheck(BaseModel):
    """Schema-constrained structured output for the one real
    `MODEL_UTILITY` call this module can make (boundary chapters only)."""

    model_config = ConfigDict(extra="forbid")

    is_agricultural: bool


def _load_system_prompt() -> str:
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    if text.startswith("<!--"):
        end = text.find("-->")
        if end != -1:
            text = text[end + 3 :]
    return text.strip()


async def is_agriculture_relevant(
    hs6: str,
    *,
    commodity_description: str,
    model_client: ModelClient,
    budget_tracker: BudgetTracker,
    thread_id: str,
    tenant_id: str,
) -> bool:
    """`True` if Agmarknet/MSP/FAOSTAT-production are relevant to `hs6` -
    a chapter-rule lookup for the overwhelming majority of codes, one real
    model call only for the narrow `BOUNDARY_CHAPTERS` set."""
    chapter = hs6[:2]
    if chapter in AGRICULTURE_CHAPTERS:
        return True
    if chapter not in BOUNDARY_CHAPTERS:
        return False

    await budget_tracker.check_and_increment(thread_id=thread_id, tenant_id=tenant_id)
    result = await model_client.generate_structured(
        system_prompt=_load_system_prompt(),
        user_content=f"HS6 code: {hs6}\nDescription: {commodity_description}",
        schema=AgricultureRelevanceCheck,
    )
    return result.is_agricultural
