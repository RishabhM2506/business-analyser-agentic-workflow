"""D8 three-way qty/price/FX decomposition + unit-value precompute:
`analytics_unit_value_series` (`docs/PLAN.md` §4 DDL, §11, §9's coverage
gate).

Sourced from Comtrade's reporter-role World-aggregate rows
(`partner_country_code='0'`, `COMTRADE_DATASET_VERSION_REPORTER_ROLE`) —
a deliberate departure from `rankings.py`'s DGCIS-backbone convention,
stated explicitly why: `raw_dgcis_annual` structurally has no quantity
column at all (verified live, §1 — this DGCIS report never returns one),
so neither a unit-value figure nor a qty/price/FX decomposition can ever
be computed from DGCIS data. Comtrade is the only source in this pipeline
that carries both value and quantity on the same row, so this one
specific metric is Comtrade-sourced by structural necessity, not
preference — `mismatch.py`'s D9 checks (which stay DGCIS-vs-Comtrade)
already own which source is authoritative for value totals; this module
only ever describes the *shape* of a change (how much is quantity vs.
price vs. FX), gated by real coverage.

`coverage_gate_passed` here is computed directly from this exact row's
own presence, **not** by calling `warehouse.coverage_gate.evaluate_coverage`
— that function's `qty_missing_pct` scope is "every tracked partner's row
for this `(hs6, flow, source)` in-window," which would silently count
every per-partner Comtrade row (both query shapes) rather than this one
specific World-aggregate cell, understating the real missing-rate for
this narrower series. For a single aggregate row per year there is
exactly one cell to gate on, so the gate correctly reduces to "is a real,
non-null value/quantity/FX triple present for this year" — computed
directly, not borrowed from a function built for a different scope.

`delta_*_pct` is only computed between **calendar-adjacent** years (year
`N` vs. year `N-1`) — never "vs. the most recent prior year with data".
HS6 120791's real Comtrade data has a genuine gap (no reporter-role World
row for imports in 2021), and silently comparing 2022 against 2020 in
that case would misreport a two-year change as a one-year one. A gap
year means the delta is un-computable for the following year too, not
approximated by skipping over it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.fx.decomposition import decompose
from app.pipeline.normalize import COMTRADE_DATASET_VERSION_REPORTER_ROLE
from app.warehouse.schema import analytics_unit_value_series, normalized_trade_flows

_WORLD_AGGREGATE_PARTNER_CODE = "0"


@dataclass(frozen=True)
class _YearData:
    value_inr_paise: int
    value_original_currency_paise: int
    fx_rate_used: Decimal
    quantity_kg: Decimal


@dataclass(frozen=True)
class UnitValueYear:
    hs6: str
    flow: str
    year: int
    unit_value_inr_paise_per_kg: Decimal | None
    delta_value_pct: Decimal | None
    delta_from_qty_pct: Decimal | None
    delta_from_price_pct: Decimal | None
    delta_from_fx_pct: Decimal | None
    coverage_gate_passed: bool


async def _fetch_world_aggregate_years(
    engine: AsyncEngine, *, hs6: str, flow: str
) -> dict[int, _YearData]:
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(
                        normalized_trade_flows.c.period_month,
                        normalized_trade_flows.c.value_inr_paise,
                        normalized_trade_flows.c.value_original_currency_paise,
                        normalized_trade_flows.c.fx_rate_used,
                        normalized_trade_flows.c.quantity_kg,
                    ).where(
                        normalized_trade_flows.c.hs6 == hs6,
                        normalized_trade_flows.c.flow == flow,
                        normalized_trade_flows.c.source == "comtrade",
                        normalized_trade_flows.c.dataset_version
                        == COMTRADE_DATASET_VERSION_REPORTER_ROLE,
                        normalized_trade_flows.c.partner_country_code
                        == _WORLD_AGGREGATE_PARTNER_CODE,
                    )
                )
            )
            .mappings()
            .all()
        )
    by_year: dict[int, _YearData] = {}
    for r in rows:
        if (
            r["value_inr_paise"] is None
            or r["value_original_currency_paise"] is None
            or r["fx_rate_used"] is None
            or r["quantity_kg"] is None
            or r["quantity_kg"] == 0
        ):
            continue
        by_year[r["period_month"].year] = _YearData(
            value_inr_paise=r["value_inr_paise"],
            value_original_currency_paise=r["value_original_currency_paise"],
            fx_rate_used=r["fx_rate_used"],
            quantity_kg=r["quantity_kg"],
        )
    return by_year


async def compute_unit_value_series(
    engine: AsyncEngine, *, hs6: str, flow: str, years: list[int]
) -> list[UnitValueYear]:
    by_year = await _fetch_world_aggregate_years(engine, hs6=hs6, flow=flow)

    results = []
    for year in sorted(years):
        current = by_year.get(year)
        # A single-cell gate (§9 reduced to this series' one aggregate row
        # per year, see module docstring): passed iff a real, non-null
        # value/quantity/FX triple exists for this year.
        gate_passed = current is not None
        unit_value = (
            Decimal(current.value_inr_paise) / current.quantity_kg if current is not None else None
        )

        delta_value_pct = delta_qty_pct = delta_price_pct = delta_fx_pct = None
        previous = by_year.get(year - 1)
        if current is not None and previous is not None:
            decomposition = decompose(
                qty_start=previous.quantity_kg,
                qty_end=current.quantity_kg,
                price_native_start=Decimal(previous.value_original_currency_paise)
                / previous.quantity_kg,
                price_native_end=Decimal(current.value_original_currency_paise)
                / current.quantity_kg,
                fx_start=previous.fx_rate_used,
                fx_end=current.fx_rate_used,
            )
            delta_value_pct = decomposition.delta_value_pct
            delta_qty_pct = decomposition.delta_qty_pct
            delta_price_pct = decomposition.delta_price_pct
            delta_fx_pct = decomposition.delta_fx_pct

        results.append(
            UnitValueYear(
                hs6=hs6,
                flow=flow,
                year=year,
                unit_value_inr_paise_per_kg=unit_value,
                delta_value_pct=delta_value_pct,
                delta_from_qty_pct=delta_qty_pct,
                delta_from_price_pct=delta_price_pct,
                delta_from_fx_pct=delta_fx_pct,
                coverage_gate_passed=gate_passed,
            )
        )
    return results


async def upsert_unit_value_series(engine: AsyncEngine, series: list[UnitValueYear]) -> int:
    if not series:
        return 0
    rows = [
        {
            "hs6": r.hs6,
            "flow": r.flow,
            "year": r.year,
            "unit_value_inr_paise_per_kg": r.unit_value_inr_paise_per_kg,
            "delta_value_pct": r.delta_value_pct,
            "delta_from_qty_pct": r.delta_from_qty_pct,
            "delta_from_price_pct": r.delta_from_price_pct,
            "delta_from_fx_pct": r.delta_from_fx_pct,
            "coverage_gate_passed": r.coverage_gate_passed,
        }
        for r in series
    ]
    async with engine.begin() as conn:
        stmt = insert(analytics_unit_value_series).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["hs6", "flow", "year"],
            set_={
                "unit_value_inr_paise_per_kg": stmt.excluded.unit_value_inr_paise_per_kg,
                "delta_value_pct": stmt.excluded.delta_value_pct,
                "delta_from_qty_pct": stmt.excluded.delta_from_qty_pct,
                "delta_from_price_pct": stmt.excluded.delta_from_price_pct,
                "delta_from_fx_pct": stmt.excluded.delta_from_fx_pct,
                "coverage_gate_passed": stmt.excluded.coverage_gate_passed,
            },
        )
        await conn.execute(stmt)
    return len(rows)
