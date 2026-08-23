"""D11 coverage gate (`docs/PLAN.md` §9) — refuses metric computation
below threshold, rather than approximating over sparse data:

```
gate(hs6, flow, window) -> passed: bool, reason: str
  expected_cells = months_in_window x tracked_partners_in_scope
  qty_missing_pct = count(status == QTY_MISSING) / expected_cells
  if qty_missing_pct > 0.30: unit_value metric is NOT emitted ... refuse, don't approximate.
```

`tracked_partners_in_scope` is deliberately a caller-supplied count, not
something this module infers from whatever happens to already be in the
database. Right now only one real partner country (Turkey) has been
ingested for the canonical scenario — inferring "tracked partners" from
present rows would silently conflate "how much of our intended scope has
data" with "how much of the data we happen to have has data," making the
gate meaningless during the current partial-coverage phase of ingestion
(the same real caveat already flagged for `report/mismatch.py`'s check A).
The caller (the eventual ingestion-job orchestrator, §7) is the one place
that actually knows the intended tracked-partner list.

`periods_expected` is computed from the window's calendar span at annual
grain (`window_end.year - window_start.year + 1`) — this pipeline's only
currently-built series (`raw_dgcis_annual`) is annual. §13's monthly
current-year section (D15, not yet built) is a distinct table
(`analytics_monthly_current_year`) with its own month-by-month row
discipline and does not go through this gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.warehouse.schema import analytics_coverage_summary, normalized_trade_flows

# §9: "if qty_missing_pct > 0.30 ... refuse" - strictly greater-than, not
# greater-or-equal, matching §10's own documented `<`/`>=` boundary
# discipline for mismatch severity bands (never guess which side a
# boundary value falls on).
_QTY_MISSING_GATE_THRESHOLD = Decimal("0.30")


@dataclass(frozen=True)
class CoverageGateResult:
    hs6: str
    flow: str
    window_start: date
    window_end: date
    expected_cells: int
    present_cells: int
    not_yet_published_cells: int
    suppressed_cells: int
    fetch_failed_cells: int
    qty_missing_cells: int
    qty_missing_pct: Decimal | None
    gate_passed: bool
    degraded: bool


async def evaluate_coverage(
    engine: AsyncEngine,
    *,
    hs6: str,
    flow: str,
    window_start: date,
    window_end: date,
    tracked_partners: int,
    source: str = "dgcis",
) -> CoverageGateResult:
    """Real status counts from `normalized_trade_flows` for `(hs6, flow,
    source)` within `[window_start, window_end]`, evaluated against §9's
    30% `QTY_MISSING` threshold. `expected_cells=0` (a zero-width window
    or `tracked_partners=0`) makes `qty_missing_pct` undefined — the gate
    then fails closed (`gate_passed=False`), never divides by zero."""
    periods_expected = window_end.year - window_start.year + 1
    expected_cells = periods_expected * tracked_partners

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(normalized_trade_flows.c.status, func.count())
                .where(
                    normalized_trade_flows.c.hs6 == hs6,
                    normalized_trade_flows.c.flow == flow,
                    normalized_trade_flows.c.source == source,
                    normalized_trade_flows.c.period_month >= window_start,
                    normalized_trade_flows.c.period_month <= window_end,
                )
                .group_by(normalized_trade_flows.c.status)
            )
        ).all()
    counts_by_status: dict[str, int] = {}
    for status, count in rows:
        counts_by_status[status] = count

    present_cells = counts_by_status.get("OK", 0) + counts_by_status.get("ZERO", 0)
    not_yet_published_cells = counts_by_status.get("NOT_YET_PUBLISHED", 0)
    suppressed_cells = counts_by_status.get("SUPPRESSED", 0)
    fetch_failed_cells = counts_by_status.get("FETCH_FAILED", 0)
    qty_missing_cells = counts_by_status.get("QTY_MISSING", 0)

    if expected_cells == 0:
        qty_missing_pct = None
        gate_passed = False
    else:
        qty_missing_pct = (Decimal(qty_missing_cells) / Decimal(expected_cells)) * 100
        gate_passed = qty_missing_pct <= _QTY_MISSING_GATE_THRESHOLD * 100

    return CoverageGateResult(
        hs6=hs6,
        flow=flow,
        window_start=window_start,
        window_end=window_end,
        expected_cells=expected_cells,
        present_cells=present_cells,
        not_yet_published_cells=not_yet_published_cells,
        suppressed_cells=suppressed_cells,
        fetch_failed_cells=fetch_failed_cells,
        qty_missing_cells=qty_missing_cells,
        qty_missing_pct=qty_missing_pct,
        gate_passed=gate_passed,
        # A real fetch failure in-window is a distinct, stronger signal
        # than "the gate's threshold happened to pass" - surfaced
        # separately so a caller can't miss it by only checking
        # gate_passed. No other definition of "degraded" is stated in
        # §9; flagged as a judgment call, not an unstated spec detail.
        degraded=fetch_failed_cells > 0,
    )


async def upsert_coverage_summary(engine: AsyncEngine, result: CoverageGateResult) -> None:
    async with engine.begin() as conn:
        stmt = insert(analytics_coverage_summary).values(
            hs6=result.hs6,
            flow=result.flow,
            window_start=result.window_start,
            window_end=result.window_end,
            expected_cells=result.expected_cells,
            present_cells=result.present_cells,
            not_yet_published_cells=result.not_yet_published_cells,
            suppressed_cells=result.suppressed_cells,
            fetch_failed_cells=result.fetch_failed_cells,
            gate_passed=result.gate_passed,
            degraded=result.degraded,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["hs6", "flow", "window_start", "window_end"],
            set_={
                "expected_cells": stmt.excluded.expected_cells,
                "present_cells": stmt.excluded.present_cells,
                "not_yet_published_cells": stmt.excluded.not_yet_published_cells,
                "suppressed_cells": stmt.excluded.suppressed_cells,
                "fetch_failed_cells": stmt.excluded.fetch_failed_cells,
                "gate_passed": stmt.excluded.gate_passed,
                "degraded": stmt.excluded.degraded,
            },
        )
        await conn.execute(stmt)
