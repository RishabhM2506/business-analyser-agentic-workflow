"""`summarize` node: MODEL_ANALYSIS call over the pre-aggregated table only
— never raw Comtrade JSON (docs/PLAN.md §2.2, §5.1 call #2, master brief
§7.1). Output must pass `app.guardrails.check_numbers_grounded` before
`assemble_response` runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.budget import BudgetExceededError, get_budget_tracker
from app.guardrails import check_numbers_grounded
from app.models import get_model_for_role
from app.schemas.errors import ErrorResponse
from app.schemas.response import TradeTable
from app.settings import get_settings
from app.state import AnalysisState, get_or_mint_trace_id, has_error

PROMPT_VERSION = "summarize-v1"
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "summarize.md"


class SummarizeOutput(BaseModel):
    """Schema-constrained output for the MODEL_ANALYSIS call — a single
    short prose field (docs/PLAN.md §2.2)."""

    model_config = ConfigDict(extra="forbid")

    analytical_summary: str = Field(min_length=1, max_length=1500)


def _load_system_prompt() -> str:
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    if text.startswith("<!--"):
        end = text.find("-->")
        if end != -1:
            text = text[end + 3 :]
    return text.strip()


def render_table(label: str, table: TradeTable) -> str:
    """Compact, plain-text rendering of one `TradeTable` for the model
    prompt — the *only* shape of trade data a model ever sees (master brief
    §7.1: "a 40 KB JSON blob passed to a model is ~10k billed input
    tokens... pre-aggregation in Python is what keeps this call under
    1,200 tokens"). Every number rendered here is copied straight from the
    table's own fields, whole-unit rounded for readability — the same
    rounding `app.guardrails.check_numbers_grounded` tolerates, so a model
    that copies a rendered figure verbatim always passes the guardrail.
    Pure function — no I/O, easily unit tested independent of any model
    call.
    """
    lines = [f"{label} (USD, top {len(table.rows)} partner countries):"]
    for row in table.rows:
        year_values = ", ".join(
            (
                f"{year}: {row.values_by_year[year]:,.0f}"
                if row.values_by_year[year] is not None
                else f"{year}: no data"
            )
            for year in table.years
        )
        lines.append(
            f"  #{row.rank} {row.partner_country} — 5yr cumulative {row.cumulative_5yr:,.0f} "
            f"USD | by year: {year_values}"
        )
    if not table.rows:
        lines.append("  (no partner-country data available)")
    if table.excluded_partner_codes:
        lines.append(
            "  (excluded aggregate/unspecified-partner codes, not shown above: "
            + ", ".join(table.excluded_partner_codes)
            + ")"
        )
    unfinalized = [year for year in table.years if year not in table.years_finalized]
    if unfinalized:
        lines.append(
            "  (provisional / not yet finalized years — subject to revision: "
            + ", ".join(str(year) for year in unfinalized)
            + ")"
        )
    return "\n".join(lines)


async def summarize(state: AnalysisState) -> dict[str, Any]:
    """Write `analytical_summary` from `imports_table`/`exports_table` via
    MODEL_ANALYSIS, then run the output guardrail before returning."""
    if has_error(state):
        return {}
    query = state.get("query")
    imports_table = state.get("imports_table")
    exports_table = state.get("exports_table")
    thread_id = state.get("thread_id")
    if query is None or imports_table is None or exports_table is None or thread_id is None:
        return {}

    try:
        await get_budget_tracker().check_and_increment(
            thread_id=thread_id, tenant_id=query.tenant_id
        )
    except BudgetExceededError:
        return {
            "error": ErrorResponse(
                error_code="BUDGET_EXCEEDED",
                message="The model-call budget for this thread or day has been reached.",
                retryable=True,
                trace_id=get_or_mint_trace_id(state),
            )
        }

    settings = get_settings()
    client = get_model_for_role("analysis", provider=settings.llm_provider)
    user_content = (
        f"HS code {query.hs_code}. Analyze the following pre-aggregated trade tables.\n\n"
        f"{render_table('IMPORTS', imports_table)}\n\n"
        f"{render_table('EXPORTS', exports_table)}"
    )
    result = await client.generate_structured(
        system_prompt=_load_system_prompt(),
        user_content=user_content,
        schema=SummarizeOutput,
    )

    if not check_numbers_grounded(
        result.analytical_summary, imports_table, exports_table, hs_code=query.hs_code
    ):
        # The concrete v1 instance of "the LLM never produces a number"
        # (master brief §2.2) — a defined error fallback, never a silent
        # partial render (docs/PLAN.md §3.2).
        return {
            "error": ErrorResponse(
                error_code="UNGROUNDED_SUMMARY",
                message=(
                    "The generated summary could not be verified against the underlying data "
                    "and was discarded rather than shown."
                ),
                retryable=True,
                trace_id=get_or_mint_trace_id(state),
            )
        }

    return {"analytical_summary": result.analytical_summary}
