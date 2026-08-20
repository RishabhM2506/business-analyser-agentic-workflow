"""`describe_item` node: MODEL_UTILITY call, schema-constrained output
(docs/PLAN.md §2.2, §5.1 call #1). Cacheable purely by
`(hs_code, prompt_version)` — never depends on year range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.budget import BudgetExceededError, get_budget_tracker
from app.guardrails import extract_numbers
from app.models import get_model_for_role
from app.schemas.errors import ErrorResponse
from app.settings import get_settings
from app.state import AnalysisState, get_or_mint_trace_id, has_error

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

PROMPT_VERSION = "describe_item-v1"
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "describe_item.md"


class DescribeItemOutput(BaseModel):
    """Schema-constrained output for the MODEL_UTILITY call — a single
    short prose field, nothing else (docs/PLAN.md §2.2: "schema-constrained
    output," not free text later parsed with regex)."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=800)


def _load_system_prompt() -> str:
    """Read `prompts/describe_item.md` and strip its leading HTML-comment
    block (human-readable `PROMPT_VERSION` note, not part of the prompt
    itself — see that file) before sending the rest to the model."""
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    if text.startswith("<!--"):
        end = text.find("-->")
        if end != -1:
            text = text[end + 3 :]
    return text.strip()


async def describe_item(state: AnalysisState) -> dict[str, Any]:
    """Write `item_description` from `state["taxonomy_text"]` via
    MODEL_UTILITY."""
    if has_error(state):
        return {}
    query = state.get("query")
    taxonomy_text = state.get("taxonomy_text")
    thread_id = state.get("thread_id")
    if query is None or taxonomy_text is None or thread_id is None:
        return {}

    try:
        await get_budget_tracker().check_and_increment(
            thread_id=thread_id, tenant_id=query.tenant_id
        )
    except BudgetExceededError:
        # Checked immediately before the call it would fund, not after
        # (docs/PLAN.md §5.5: fails closed) — no model spend happens on
        # this branch.
        return {
            "error": ErrorResponse(
                error_code="BUDGET_EXCEEDED",
                message="The model-call budget for this thread or day has been reached.",
                retryable=True,
                trace_id=get_or_mint_trace_id(state),
            )
        }

    settings = get_settings()
    client = get_model_for_role("utility", provider=settings.llm_provider)
    result = await client.generate_structured(
        system_prompt=_load_system_prompt(),
        user_content=f"HS code: {query.hs_code}\n\n{taxonomy_text}",
        schema=DescribeItemOutput,
    )

    if extract_numbers(result.description):
        # Code-level enforcement, not prompt-wording alone (finding
        # M2/AWR-04 — master brief §8 explicitly forbids relying on a
        # prompt instruction as the only control). Unlike `summarize`'s
        # output, which legitimately describes real table values,
        # `item_description` has no legitimate reason to contain *any*
        # number at all: prompts/describe_item.md's own "Hard rules"
        # already instruct the model never to mention one (prices, costs,
        # growth, or "any number"), so this check can be strict —
        # membership-in-a-table doesn't even apply here — rather than the
        # table-membership check `check_numbers_grounded` does for the
        # analytical summary.
        #
        # QA finding (2026-08-20): mirrors the identical, previously-missing
        # log line added to summarize.py's UNGROUNDED_SUMMARY path — see
        # that file's comment for the full rationale (a rejection here was
        # previously undiagnosable after the fact; nothing about the
        # rejected content was ever logged).
        logger.warning(
            "describe_item.guardrail_rejected",
            hs_code=query.hs_code,
            trace_id=get_or_mint_trace_id(state),
            found_numbers=extract_numbers(result.description),
            rejected_description=result.description,
        )
        return {
            "error": ErrorResponse(
                error_code="UNGROUNDED_DESCRIPTION",
                message=(
                    "The generated item description could not be verified as number-free "
                    "and was discarded rather than shown."
                ),
                retryable=True,
                trace_id=get_or_mint_trace_id(state),
            )
        }

    return {"item_description": result.description}
