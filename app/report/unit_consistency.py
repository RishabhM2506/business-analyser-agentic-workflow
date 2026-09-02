"""D10 unit-consistency gate (`docs/PLAN.md` §5, §15) — runs at HS8->HS6
rollup time, before any HS6-level aggregate is computed from multiple HS8
sibling lines.

Only `raw_dgcis_annual` carries a per-row, source-stated unit
(`"KGS"`, etc, verified live from the report's own header, §1) at HS8
granularity. The Comtrade mirror is HS6-level only (`hs8 IS NULL` in
`normalized_trade_flows`) and always reports net weight in kg by Comtrade
convention, so it never participates in this specific check — there are
no HS8 siblings to compare on that side.

If an HS6's HS8 siblings report unit not IN ('KGS',) (right now DGCIS
happens to be homogeneously KGS for every code observed) or, more to the
point, report *different* units from each other, D5's table (§5) requires
`UNIT_MISMATCH` — never silently picking one sibling's unit as "the" unit
for the rolled-up HS6 series."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import distinct, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.warehouse.schema import normalized_trade_flows, raw_dgcis_annual

# Statuses that already carry a stronger, unrelated signal than "the value
# looks fine but its unit is inconsistent" - UNIT_MISMATCH must never
# overwrite one of these (§5: each status has its own, distinct producing
# condition; overwriting FETCH_FAILED with UNIT_MISMATCH would hide a
# fetch failure behind an unrelated finding).
_STATUSES_ELIGIBLE_FOR_UNIT_MISMATCH = ("OK", "ZERO", "QTY_MISSING")


@dataclass(frozen=True)
class UnitConsistencyResult:
    hs6: str
    is_consistent: bool
    units_by_hs8: dict[str, set[str]]


async def check_unit_consistency(engine: AsyncEngine, *, hs6: str) -> UnitConsistencyResult:
    """Every distinct (hs8, unit) pair `raw_dgcis_annual` has ever recorded
    for this `hs6`'s siblings, grouped by hs8 (a single hs8 reporting two
    different units across its own historical rows is exactly as much a
    mismatch as two different hs8 siblings disagreeing — both mean "the
    unit isn't a safe constant to roll up under," so both are tracked, not
    collapsed to whichever row happened to be seen last). A `NULL` unit is
    excluded from the comparison (never a data point, not "the unit is
    null" — a genuinely unknown unit isn't evidence of a mismatch either
    way)."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(distinct(raw_dgcis_annual.c.hs8), raw_dgcis_annual.c.unit).where(
                    raw_dgcis_annual.c.hs8.startswith(hs6),
                    raw_dgcis_annual.c.unit.is_not(None),
                )
            )
        ).all()

    units_by_hs8: dict[str, set[str]] = {}
    for hs8, unit in rows:
        units_by_hs8.setdefault(hs8, set()).add(unit)

    distinct_units = {unit for units in units_by_hs8.values() for unit in units}
    return UnitConsistencyResult(
        hs6=hs6, is_consistent=len(distinct_units) <= 1, units_by_hs8=units_by_hs8
    )


async def mark_unit_mismatch(engine: AsyncEngine, *, hs6: str, flow: str) -> int:
    """Runs `check_unit_consistency` and, only if it fails, stamps every
    DGCIS-sourced `normalized_trade_flows` row for this `(hs6, flow)` whose
    status doesn't already carry a stronger signal (see
    `_STATUSES_ELIGIBLE_FOR_UNIT_MISMATCH`) with `status='UNIT_MISMATCH'`
    and a `status_detail` naming the observed, disagreeing units — never
    guessing which sibling's unit is the "real" one. Returns the number of
    rows updated (0 if consistent)."""
    result = await check_unit_consistency(engine, hs6=hs6)
    if result.is_consistent:
        return 0

    status_detail = "unit mismatch across HS8 siblings: " + ", ".join(
        f"{hs8}={sorted(units)}" for hs8, units in sorted(result.units_by_hs8.items())
    )
    async with engine.begin() as conn:
        outcome = await conn.execute(
            update(normalized_trade_flows)
            .where(
                normalized_trade_flows.c.hs6 == hs6,
                normalized_trade_flows.c.flow == flow,
                normalized_trade_flows.c.source == "dgcis",
                normalized_trade_flows.c.status.in_(_STATUSES_ELIGIBLE_FOR_UNIT_MISMATCH),
            )
            .values(status="UNIT_MISMATCH", status_detail=status_detail)
        )
    return outcome.rowcount
