"""Facts JSON assembly (`docs/PLAN.md` §14, the frozen LLM contract, D4).
Every numeral the narrative model is later allowed to state must trace to
a field in this document — `report/narrative.py`'s post-validator (not yet
built) will extract every number from the model's prose and assert
membership against a flattened set of every numeric leaf here.

**Read-only over the analytics/ref layers, by design.** `assemble_facts`
never touches `normalized_trade_flows` directly and never recomputes a
metric — it only reads `analytics_partner_rankings`,
`analytics_unit_value_series`, `analytics_mismatch_checks`,
`analytics_coverage_summary`, `ref_duty_components` (via
`ManualDutySource`), `ref_regulatory_notes`, and `ref_hs6_hs8_crosswalk`.
This matches `schema.py`'s own stated architecture ("Analytics layer:
precomputed on ingest, API reads only this") and the ingestion-vs-query
plane separation (D13): every one of those tables must already be
populated by its own precompute job (`rankings.py`, `unit_value.py`,
`mismatch.py`, `coverage_gate.py`) before this function is called, never
computed request-time.

Scoped to one `flow` at a time, matching every other module built this
session (`mismatch.py`, `rankings.py`, `unit_value.py`,
`coverage_gate.py`) — a caller wanting a bidirectional report calls this
twice.

**Real, flagged deviations from §14's literal example JSON**, each
because fabricating the example's exact shape would require inventing
data that does not exist yet, which is exactly what this pipeline's whole
evidence-first mandate forbids:

- `month_wise_current_year` is always `[]` right now — `§13`'s own
  design has DGCIS's *monthly* ingestion job write
  `analytics_monthly_current_year` "as part of its normal monthly run",
  and that job doesn't exist yet (open item, `docs/STATE.md`). An empty
  list is the honest, correct output of "no rows exist yet" — not a
  special case to work around.
- `landed_cost.as_of_period` (renamed from the example's `as_of_month`):
  this pipeline has no monthly-grain CIF figure yet (`analytics_landed_cost`
  has no writer), so the most recent real annual CIF/kg figure is used
  instead, labeled by year, not a fabricated month. `landed_cost` is
  `None` entirely when no year in the window has a real, gate-passed
  unit-value figure to use as CIF.
- `coverage` is `None` when no `analytics_coverage_summary` row exists for
  the caller's exact window — this pipeline's coverage gate is
  window-specific (§9), so a window nobody has run the gate for yet has
  no honest number to report, not a zero.
- Per-year `status` (on `annual_series` entries and `all_other_partners`)
  uses a severity ordering across `cell_status`'s 10 values that `§5`
  does not itself define a total order for — `_STATUS_SEVERITY` below is
  this module's own reasoned ordering (worse-status-wins), flagged the
  same way `§14`'s own 60% HHI/concentration threshold is flagged as
  reasoned-not-verified.
- `regulatory_note_missing_warning`'s "top-1 partner share > 60%" trigger
  is computed as the literal top-1 share (not re-derived from the HHI
  value, which measures a different, related but distinct thing) against
  the most recent year in the window that has real ranking data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.knowledge.provider import get_taxonomy_entry
from app.pipeline.duty_source import ManualDutySource
from app.report.landed_cost import LandedCostResult, compute_landed_cost
from app.report.rankings import PartnerRanking, compute_hhi
from app.warehouse.schema import (
    analytics_coverage_summary,
    analytics_mismatch_checks,
    analytics_partner_rankings,
    analytics_unit_value_series,
    ref_hs6_hs8_crosswalk,
    ref_regulatory_notes,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARTNER_AREAS_CSV = "data/comtrade-partner-areas.csv"
_ALL_PARTNERS = "ALL_PARTNERS"
_UNMAPPED = "UNMAPPED"

# This module's own reasoned worse-status-wins ordering across cell_status's
# 10 values (§5 defines each status's producing condition but not a total
# order across them) - flagged the same way §14 flags its own invented
# 60% concentration threshold.
_STATUS_SEVERITY = (
    "OK",
    "ZERO",
    "PROVISIONAL",
    "QTY_MISSING",
    "UNIT_MISMATCH",
    "NOT_YET_PUBLISHED",
    "SUPPRESSED",
    "NOT_REPORTED",
    "CODE_RETIRED",
    "FETCH_FAILED",
)

# §14: "top-1 partner share > 60%" - this module's own reasoned starting
# point, not empirically validated (same flagged status as §14's own
# statement of this number).
_CONCENTRATION_WARNING_THRESHOLD = Decimal("60")


@lru_cache(maxsize=1)
def _load_partner_names(csv_path: str = _PARTNER_AREAS_CSV) -> dict[str, str]:
    resolved = Path(csv_path)
    resolved = resolved if resolved.is_absolute() else _REPO_ROOT / resolved
    names: dict[str, str] = {}
    with resolved.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            names[row["partner_code"]] = row["partner_desc"]
    return names


def _display_name(partner_country_code: str) -> str:
    if partner_country_code in (_ALL_PARTNERS, _UNMAPPED):
        return partner_country_code
    return _load_partner_names().get(partner_country_code, partner_country_code)


def _worst_status(statuses: list[str]) -> str:
    return max(statuses, key=_STATUS_SEVERITY.index)


class PartnerFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    country: str
    value_inr_paise: int
    status: str


class AllOtherPartnersFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value_inr_paise: int
    status: str


class AnnualSeriesYear(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int
    flow: str
    total_inr_paise: int | None
    status: str
    partners: list[PartnerFact]
    all_other_partners: AllOtherPartnersFact


class UnitValueTrendYear(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int
    inr_paise_per_kg: Decimal | None
    delta_qty_pct: Decimal | None
    delta_price_pct: Decimal | None
    delta_fx_pct: Decimal | None


class HhiYear(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int
    hhi: Decimal | None


class MismatchCheckFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    year: int
    partner: str
    gap_pct: Decimal
    severity: str


class CoverageFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_cells: int
    present_cells: int
    not_yet_published: int
    suppressed: int
    fetch_failed: int
    degraded: bool


class Window(BaseModel):
    model_config = ConfigDict(extra="forbid")

    years: int
    start_year: int
    end_year: int


class Facts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hs6: str
    product_label: str
    window: Window
    top_n: int
    annual_series: list[AnnualSeriesYear]
    month_wise_current_year: list[dict[str, object]]
    unit_value_trend: list[UnitValueTrendYear]
    hhi_by_year: list[HhiYear]
    landed_cost: LandedCostResult | None
    landed_cost_as_of_period: str | None
    mismatch_checks: list[MismatchCheckFact]
    regulatory_note: str | None
    regulatory_note_missing_warning: bool
    coverage: CoverageFact | None
    hs8_split_note: str


@dataclass(frozen=True)
class _RankingRow:
    partner_country_code: str
    rank: int | None
    value_inr_paise: int | None
    status: str


async def _fetch_rankings_by_year(
    engine: AsyncEngine, *, hs6: str, flow: str, years: list[int]
) -> dict[int, list[_RankingRow]]:
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(analytics_partner_rankings).where(
                        analytics_partner_rankings.c.hs6 == hs6,
                        analytics_partner_rankings.c.flow == flow,
                        analytics_partner_rankings.c.year.in_(years),
                    )
                )
            )
            .mappings()
            .all()
        )
    by_year: dict[int, list[_RankingRow]] = {year: [] for year in years}
    for r in rows:
        by_year[r["year"]].append(
            _RankingRow(
                partner_country_code=r["partner_country_code"],
                rank=r["rank"],
                value_inr_paise=r["value_inr_paise"],
                status=r["status"],
            )
        )
    return by_year


def _to_partner_facts(ranked_rows: list[_RankingRow]) -> list[PartnerFact]:
    facts = []
    for r in ranked_rows:
        assert (
            r.rank is not None and r.value_inr_paise is not None
        )  # by construction: ranked_rows is pre-filtered to rank is not None
        facts.append(
            PartnerFact(
                rank=r.rank,
                country=_display_name(r.partner_country_code),
                value_inr_paise=r.value_inr_paise,
                status=r.status,
            )
        )
    return facts


def _build_annual_series_year(
    *, year: int, flow: str, top_n: int, rows: list[_RankingRow]
) -> AnnualSeriesYear:
    if not rows:
        return AnnualSeriesYear(
            year=year,
            flow=flow,
            total_inr_paise=None,
            status="NOT_REPORTED",
            partners=[],
            all_other_partners=AllOtherPartnersFact(value_inr_paise=0, status="NOT_REPORTED"),
        )

    ranked = [r for r in rows if r.rank is not None]
    top = sorted(ranked, key=lambda r: r.rank if r.rank is not None else 0)[:top_n]
    beyond_top_n = [r for r in ranked if r.rank is not None and r.rank > top_n]
    unranked = [r for r in rows if r.rank is None]

    total_inr_paise = sum(r.value_inr_paise for r in ranked if r.value_inr_paise is not None)
    other_value = sum(r.value_inr_paise for r in beyond_top_n if r.value_inr_paise is not None)
    other_status = (
        _worst_status([r.status for r in beyond_top_n + unranked])
        if (beyond_top_n or unranked)
        else "OK"
    )
    year_status = _worst_status([r.status for r in rows])

    return AnnualSeriesYear(
        year=year,
        flow=flow,
        total_inr_paise=total_inr_paise,
        status=year_status,
        partners=_to_partner_facts(top),
        all_other_partners=AllOtherPartnersFact(value_inr_paise=other_value, status=other_status),
    )


async def _fetch_unit_value_trend(
    engine: AsyncEngine, *, hs6: str, flow: str, years: list[int]
) -> list[UnitValueTrendYear]:
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(analytics_unit_value_series).where(
                        analytics_unit_value_series.c.hs6 == hs6,
                        analytics_unit_value_series.c.flow == flow,
                        analytics_unit_value_series.c.year.in_(years),
                    )
                )
            )
            .mappings()
            .all()
        )
    by_year = {r["year"]: r for r in rows}
    return [
        UnitValueTrendYear(
            year=year,
            inr_paise_per_kg=(
                by_year[year]["unit_value_inr_paise_per_kg"] if year in by_year else None
            ),
            delta_qty_pct=(by_year[year]["delta_from_qty_pct"] if year in by_year else None),
            delta_price_pct=(by_year[year]["delta_from_price_pct"] if year in by_year else None),
            delta_fx_pct=(by_year[year]["delta_from_fx_pct"] if year in by_year else None),
        )
        for year in years
    ]


async def _fetch_mismatch_checks(
    engine: AsyncEngine, *, hs6: str, flow: str, years: list[int]
) -> list[MismatchCheckFact]:
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(analytics_mismatch_checks).where(
                        analytics_mismatch_checks.c.hs6 == hs6,
                        analytics_mismatch_checks.c.flow == flow,
                        analytics_mismatch_checks.c.year.in_(years),
                    )
                )
            )
            .mappings()
            .all()
        )
    return [
        MismatchCheckFact(
            check=r["check_name"],
            year=r["year"],
            partner=_display_name(r["partner_country_code"]),
            gap_pct=r["gap_pct"],
            severity=r["severity"],
        )
        for r in rows
    ]


async def _fetch_coverage(
    engine: AsyncEngine, *, hs6: str, flow: str, window_start: date, window_end: date
) -> CoverageFact | None:
    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    select(analytics_coverage_summary).where(
                        analytics_coverage_summary.c.hs6 == hs6,
                        analytics_coverage_summary.c.flow == flow,
                        analytics_coverage_summary.c.window_start == window_start,
                        analytics_coverage_summary.c.window_end == window_end,
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
    if row is None:
        return None
    return CoverageFact(
        expected_cells=row["expected_cells"],
        present_cells=row["present_cells"],
        not_yet_published=row["not_yet_published_cells"],
        suppressed=row["suppressed_cells"],
        fetch_failed=row["fetch_failed_cells"],
        degraded=row["degraded"],
    )


async def _fetch_hs8_split_note(engine: AsyncEngine, *, hs6: str) -> tuple[str, list[str]]:
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(ref_hs6_hs8_crosswalk.c.hs8).where(
                        ref_hs6_hs8_crosswalk.c.hs6 == hs6,
                        ref_hs6_hs8_crosswalk.c.effective_to.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
    active_hs8 = sorted(rows)
    if len(active_hs8) == 1:
        note = (
            f"{active_hs8[0]} is the only ITC-HS8 line beneath {hs6} as of this vintage "
            f"(from ref_hs6_hs8_crosswalk)."
        )
    elif active_hs8:
        note = (
            f"{len(active_hs8)} ITC-HS8 lines beneath {hs6} as of this vintage: "
            f"{', '.join(active_hs8)}."
        )
    else:
        note = f"No ITC-HS8 line has been observed beneath {hs6} yet."
    return note, active_hs8


async def _fetch_regulatory_note(engine: AsyncEngine, *, hs6: str) -> str | None:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                select(ref_regulatory_notes.c.note).where(ref_regulatory_notes.c.hs6 == hs6)
            )
        ).scalar_one_or_none()


async def _build_landed_cost(
    engine: AsyncEngine, *, hs8: str | None, years: list[int], as_of: date
) -> tuple[LandedCostResult | None, str | None]:
    if hs8 is None:
        return None, None
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(
                    analytics_unit_value_series.c.year,
                    analytics_unit_value_series.c.unit_value_inr_paise_per_kg,
                ).where(
                    analytics_unit_value_series.c.hs6 == hs8[:6],
                    analytics_unit_value_series.c.year.in_(years),
                    analytics_unit_value_series.c.coverage_gate_passed.is_(True),
                    analytics_unit_value_series.c.unit_value_inr_paise_per_kg.is_not(None),
                )
            )
        ).all()
    if not rows:
        return None, None
    most_recent_year, cif_per_kg = max(rows, key=lambda r: r[0])
    cif_paise = int(cif_per_kg.to_integral_value())
    duty_source = ManualDutySource(engine=engine)
    evidence = await duty_source.get_duty_evidence(hs8, as_of=as_of)
    result = compute_landed_cost(cif_paise, evidence)
    return result, str(most_recent_year)


async def assemble_facts(
    engine: AsyncEngine,
    *,
    hs6: str,
    flow: str,
    window_start: date,
    window_end: date,
    top_n: int,
    as_of: date,
) -> Facts:
    years = list(range(window_start.year, window_end.year + 1))

    rankings_by_year = await _fetch_rankings_by_year(engine, hs6=hs6, flow=flow, years=years)
    annual_series = [
        _build_annual_series_year(year=year, flow=flow, top_n=top_n, rows=rankings_by_year[year])
        for year in years
    ]

    hhi_by_year = [
        HhiYear(
            year=year,
            hhi=compute_hhi(
                [
                    PartnerRanking(
                        hs6=hs6,
                        flow=flow,
                        year=year,
                        partner_country_code=r.partner_country_code,
                        rank=r.rank,
                        value_inr_paise=r.value_inr_paise,
                        status=r.status,
                    )
                    for r in rankings_by_year[year]
                ]
            ),
        )
        for year in years
    ]

    latest_hhi_year = next((y for y in reversed(hhi_by_year) if y.hhi is not None), None)
    top1_share_exceeds_threshold = False
    if latest_hhi_year is not None:
        year_rows = rankings_by_year[latest_hhi_year.year]
        valued = [r.value_inr_paise for r in year_rows if r.value_inr_paise is not None]
        total = sum(valued)
        if total > 0:
            top1_share_exceeds_threshold = (
                Decimal(max(valued)) / Decimal(total) * 100
            ) > _CONCENTRATION_WARNING_THRESHOLD

    regulatory_note = await _fetch_regulatory_note(engine, hs6=hs6)
    regulatory_note_missing_warning = regulatory_note is None and top1_share_exceeds_threshold

    hs8_split_note, active_hs8 = await _fetch_hs8_split_note(engine, hs6=hs6)
    landed_cost, landed_cost_as_of_period = await _build_landed_cost(
        engine, hs8=active_hs8[0] if len(active_hs8) == 1 else None, years=years, as_of=as_of
    )

    taxonomy_entry = get_taxonomy_entry(hs6)
    product_label = taxonomy_entry.description if taxonomy_entry is not None else hs6

    return Facts(
        hs6=hs6,
        product_label=product_label,
        window=Window(years=len(years), start_year=years[0], end_year=years[-1]),
        top_n=top_n,
        annual_series=annual_series,
        month_wise_current_year=[],
        unit_value_trend=await _fetch_unit_value_trend(engine, hs6=hs6, flow=flow, years=years),
        hhi_by_year=hhi_by_year,
        landed_cost=landed_cost,
        landed_cost_as_of_period=landed_cost_as_of_period,
        mismatch_checks=await _fetch_mismatch_checks(engine, hs6=hs6, flow=flow, years=years),
        regulatory_note=regulatory_note,
        regulatory_note_missing_warning=regulatory_note_missing_warning,
        coverage=await _fetch_coverage(
            engine, hs6=hs6, flow=flow, window_start=window_start, window_end=window_end
        ),
        hs8_split_note=hs8_split_note,
    )
