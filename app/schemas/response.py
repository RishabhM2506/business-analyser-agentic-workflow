"""Response envelope: `TradeAnalysisResponse` and its constituent shapes.

Exact field set per docs/PLAN.md §3.2. `values_by_year` uses `None` for a
missing year — the frontend renders that as `"—"`, never interpolated
(master brief §2.2: "Missing data is shown as missing").
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Provenance(BaseModel):
    """Data provenance attached to every analysis response."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["UN Comtrade (comtradeapi.un.org)"]
    retrieved_at: datetime
    period_type: Literal["calendar_year"]  # explicit: NOT Indian fiscal year — Gate 0 finding
    currency: Literal["USD"]
    prompt_version: str
    # Finding M22/PBO-04 (schema half): the product never stated anywhere
    # that this is India's trade data specifically, even though the entire
    # system is hard-coded to India as the fixed reporter
    # (`INDIA_REPORTER_CODE`, app/tools/comtrade_client.py). The only way
    # "India" could previously reach the user was incidentally, inside
    # unconstrained model prose, or — worse — confusingly, as an unexplained
    # partner-table row (M20/PBO-02). A `Literal["India"]` (not a plain
    # `str`) matches the existing pattern for `source`/`period_type`/
    # `currency`: a schema-enforced fact about this deployment, not
    # something any node computes or a model could contradict. Frontend
    # rendering of this field is out of scope here — this is the contract
    # half only.
    reporter_country: Literal["India"]


class CountryRow(BaseModel):
    """One partner-country row within a `TradeTable`."""

    model_config = ConfigDict(extra="forbid")

    partner_country: str
    partner_code: str
    values_by_year: dict[int, float | None]
    cumulative_5yr: float
    rank: int  # ranked by cumulative_5yr, per Gate 0 answer


class TradeTable(BaseModel):
    """Top-10 partner-country table for one trade flow (imports or exports)."""

    model_config = ConfigDict(extra="forbid")

    unit: Literal["USD"]
    years: list[int]
    years_finalized: list[int]  # subset of `years` NOT flagged provisional by Comtrade
    # Finding M21/PBO-03: a year absent from `years_finalized` was previously
    # presented identically to the user whether it was genuinely
    # still-settling (some records exist, at least one marked provisional)
    # or structurally had zero records at all (commonly because this HS6
    # code did not exist in that year's HS nomenclature edition — see
    # `app.nodes.aggregate.flag_years_no_data`). This field isolates the
    # zero-records case so the frontend can use accurate, non-promissory
    # copy ("no data recorded") instead of "not yet finalized" for a year
    # that may never have data. Always disjoint from `years_finalized`.
    # Defaulted (not required like its sibling) since this is a newer,
    # purely additive field — every real construction path
    # (`build_trade_table`) sets it explicitly regardless.
    years_no_data: list[int] = Field(default_factory=list)
    excluded_partner_codes: list[str]  # transparency: aggregate/"nes" codes stripped
    rows: list[CountryRow]  # top 10


class TradeAnalysisResponse(BaseModel):
    """The full analysis result returned from `POST /threads/{id}/messages`."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    message_id: str
    hs_code: str
    item_description: str
    imports: TradeTable
    exports: TradeTable
    analytical_summary: str
    provenance: Provenance
