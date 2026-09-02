"""Response envelope: `TradeAnalysisResponse` and its constituent shapes.

Exact field set per docs/PLAN.md §3.2. `values_by_year` uses `None` for a
missing year — the frontend renders that as `"—"`, never interpolated
(master brief §2.2: "Missing data is shown as missing").
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.report.facts import Facts


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
    # 2026-08-20, live user-reported finding: a year whose Comtrade fetch
    # itself failed (rate-limited/timed-out/etc. after every retry attempt)
    # is a *third*, distinct case from both `years_finalized`'s complement
    # and `years_no_data` — we don't actually know whether real data exists
    # for that year, only that we couldn't retrieve it just now. Conflating
    # it with `years_no_data` (which means Comtrade was successfully asked
    # and genuinely had nothing) would repeat the exact class of mistake
    # finding M21/PBO-03 already fixed for a different pair of cases. Each
    # entry is a real, honest, one-line note (the real caught exception's
    # own message, never a paraphrase) — computed here in code and placed
    # directly in the response, never passed through the LLM, matching how
    # every other footnote field on this model is grounded. Empty in the
    # overwhelmingly common case (nothing failed) — defaulted, not
    # required, since every pre-existing construction path predates this
    # field.
    fetch_issues: list[str] = Field(default_factory=list)
    # The `year` half of each `fetch_issues` entry, structured — the
    # frontend needs this to correctly exclude a fetch-failed year from the
    # "provisional" `*` marker it renders per year-column (a fetch failure
    # is not "check back later, still settling"), and parsing a year back
    # out of `fetch_issues`' free-text messages would be fragile. Always
    # exactly `len(fetch_issues)` long, same order.
    fetch_issue_years: list[int] = Field(default_factory=list)
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


class RankedCandidateOut(BaseModel):
    """One reranked, display-ready HS6 candidate returned from
    `POST /threads/{thread_id}/search` (`app.search.service`)."""

    model_config = ConfigDict(extra="forbid")

    hs_code: str
    description: str
    relevance_score: float


class ProductSearchResponse(BaseModel):
    """The full result of a free-text product search. `outcome` drives the
    frontend's branch: `disambiguate` asks the user to pick from
    `candidates` (at most `app.search.service.MAX_DISAMBIGUATE_CANDIDATES`,
    always accompanied client-side by an "or describe it again" option —
    every search that finds anything real ends here, never auto-navigates,
    2026-09-02 product decision: see `app.search.service`'s own module
    docstring for why); `no_candidates_found` is a normal 200 (a nonsense
    query returning nothing is a legitimate outcome, not an error — the
    same principle as `years_no_data` rendering "no data recorded," not a
    failure)."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    query_text: str
    outcome: Literal["disambiguate", "no_candidates_found"]
    candidates: list[RankedCandidateOut]


class TradeReportResponse(BaseModel):
    """The full result of `POST /threads/{thread_id}/trade-report`
    (`app.report.facts`/`app.report.narrative`) — India trade-analysis
    pipeline, D14. `facts` is the complete frozen contract document
    (`docs/PLAN.md` §14) every numeral in `narrative` must trace back to;
    exposed directly (not summarized/re-shaped) so a caller can render a
    full data view independent of the narrative prose. `narrative_source`
    tells the caller whether the prose came straight from the model, a
    grounding-retry, or the deterministic template fallback — never hidden
    from the API consumer (§14's own "reject -> regenerate once ->
    template fallback" policy is a real, visible outcome, not an internal
    implementation detail)."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    facts: Facts
    narrative: str
    narrative_source: Literal["model", "model_retry", "template_fallback"]
