"""D15 current-year month-wise section (`docs/PLAN.md` §13) —
`analytics_monthly_current_year` writer. Reads `raw_dgcis_monthly`
directly, never `normalized_trade_flows` — this section is explicitly its
own, separate analytics table per §13 ("written by the DGCIS ingestion
job as part of its normal monthly run"), not part of the D9 mismatch-
check pipeline's normalized-layer join.

**Always writes exactly 12 rows** (Jan-Dec of the requested year), even
for a month with no raw row present at all — a real, stated limitation:
at this read-only layer, a month the ingestion job hasn't fetched yet is
indistinguishable from a month DGCIS genuinely hasn't published yet, so
both read as `NOT_YET_PUBLISHED` here. Real fetch failures are tracked
separately via `dead_letter_ingestion`, never conflated with this status.

`mom_change_pct`/`yoy_same_month_pct` are computed deterministically here
— never trusting DGCIS's own displayed "%Growth" figure, matching this
pipeline's "we compute derived numbers ourselves, never an external
source's own" discipline (D8). Reads the *prior* year's raw data too:
January's month-over-month comparison needs December of the previous
year, and every month's year-over-year comparison needs the same month
one year prior.

A real HS6 with more than one active HS8 line for the same month sums
their `value_inr_paise` (money is always addable) but leaves
`quantity_kg` as `None` (`QTY_MISSING`) rather than summing quantities
across lines without confirming they share a unit — D10's own concern,
not re-litigated here, just never silently assumed safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.warehouse.schema import analytics_monthly_current_year, raw_dgcis_monthly

_MARKER_NOT_YET_PUBLISHED = "A"
_MARKER_PROVISIONAL = "F"

_HUNDRED = Decimal(100)


@dataclass(frozen=True)
class _RawCell:
    value_inr_paise: int | None
    quantity_kg: Decimal | None
    marker: str


@dataclass(frozen=True)
class MonthlyCurrentYearRow:
    hs6: str
    flow: str
    month: date
    value_inr_paise: int | None
    status: str
    status_detail: str | None
    is_provisional: bool
    mom_change_pct: Decimal | None
    yoy_same_month_pct: Decimal | None


def _pct_change(*, previous: int, current: int) -> Decimal | None:
    if previous == 0:
        return None
    return (Decimal(current - previous) / Decimal(previous)) * _HUNDRED


async def _fetch_raw_cells(
    engine: AsyncEngine, *, hs6: str, flow: str, years: list[int]
) -> dict[date, _RawCell]:
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_dgcis_monthly).where(
                        raw_dgcis_monthly.c.hs8.startswith(hs6),
                        raw_dgcis_monthly.c.flow == flow,
                    )
                )
            )
            .mappings()
            .all()
        )

    by_month: dict[date, list[_RawCell]] = {}
    for row in rows:
        month = row["calendar_month"]
        if month.year not in years:
            continue
        raw_payload = row["raw_payload"]
        marker = str(raw_payload.get("marker", "")) if isinstance(raw_payload, dict) else ""
        value = row["value_inr_paise"]
        assert value is None or isinstance(value, int)
        quantity = row["quantity"]
        assert quantity is None or isinstance(quantity, Decimal)
        by_month.setdefault(month, []).append(
            _RawCell(value_inr_paise=value, quantity_kg=quantity, marker=marker)
        )

    cells: dict[date, _RawCell] = {}
    for month, month_cells in by_month.items():
        if len(month_cells) == 1:
            cells[month] = month_cells[0]
            continue
        values = [c.value_inr_paise for c in month_cells]
        total_value = (
            sum(v for v in values if v is not None) if all(v is not None for v in values) else None
        )
        markers = {c.marker for c in month_cells}
        cells[month] = _RawCell(
            value_inr_paise=total_value,
            quantity_kg=None,  # never guessed across multiple real HS8 lines
            marker=next(iter(markers)) if len(markers) == 1 else "",
        )
    return cells


def _status_for_cell(cell: _RawCell | None) -> tuple[str, str | None, bool]:
    """`(status, status_detail, is_provisional)` — `is_provisional` is a
    real, orthogonal signal from `status` itself: e.g. a month can
    legitimately be `status='ZERO'` (real, reported zero trade) while
    still `is_provisional=True` (that zero hasn't been finalized yet)."""
    if cell is None or cell.marker == _MARKER_NOT_YET_PUBLISHED:
        return "NOT_YET_PUBLISHED", "DGCIS has not published this month yet", False
    if cell.value_inr_paise is None:
        return "NOT_REPORTED", None, False
    is_provisional = cell.marker == _MARKER_PROVISIONAL
    if cell.value_inr_paise == 0:
        return "ZERO", None, is_provisional
    if cell.quantity_kg is None:
        return "QTY_MISSING", None, is_provisional
    if is_provisional:
        return "PROVISIONAL", None, True
    return "OK", None, False


async def compute_monthly_current_year(
    engine: AsyncEngine, *, hs6: str, flow: str, year: int
) -> list[MonthlyCurrentYearRow]:
    cells = await _fetch_raw_cells(engine, hs6=hs6, flow=flow, years=[year - 1, year])

    rows = []
    for month_num in range(1, 13):
        month = date(year, month_num, 1)
        cell = cells.get(month)
        status, status_detail, is_provisional = _status_for_cell(cell)
        value = cell.value_inr_paise if cell is not None else None

        prev_month = date(year, month_num - 1, 1) if month_num > 1 else date(year - 1, 12, 1)
        prev_cell = cells.get(prev_month)
        mom = None
        if value is not None and prev_cell is not None and prev_cell.value_inr_paise is not None:
            mom = _pct_change(previous=prev_cell.value_inr_paise, current=value)

        prior_year_cell = cells.get(date(year - 1, month_num, 1))
        yoy = None
        if (
            value is not None
            and prior_year_cell is not None
            and prior_year_cell.value_inr_paise is not None
        ):
            yoy = _pct_change(previous=prior_year_cell.value_inr_paise, current=value)

        rows.append(
            MonthlyCurrentYearRow(
                hs6=hs6,
                flow=flow,
                month=month,
                value_inr_paise=value,
                status=status,
                status_detail=status_detail,
                is_provisional=is_provisional,
                mom_change_pct=mom,
                yoy_same_month_pct=yoy,
            )
        )
    return rows


async def upsert_monthly_current_year(
    engine: AsyncEngine, rows: list[MonthlyCurrentYearRow], *, data_as_of: datetime
) -> int:
    if not rows:
        return 0
    db_rows = [
        {
            "hs6": r.hs6,
            "flow": r.flow,
            "month": r.month,
            "value_inr_paise": r.value_inr_paise,
            "status": r.status,
            "status_detail": r.status_detail,
            "is_provisional": r.is_provisional,
            "mom_change_pct": r.mom_change_pct,
            "yoy_same_month_pct": r.yoy_same_month_pct,
            "data_as_of": data_as_of,
        }
        for r in rows
    ]
    async with engine.begin() as conn:
        stmt = insert(analytics_monthly_current_year).values(db_rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["hs6", "flow", "month"],
            set_={
                "value_inr_paise": stmt.excluded.value_inr_paise,
                "status": stmt.excluded.status,
                "status_detail": stmt.excluded.status_detail,
                "is_provisional": stmt.excluded.is_provisional,
                "mom_change_pct": stmt.excluded.mom_change_pct,
                "yoy_same_month_pct": stmt.excluded.yoy_same_month_pct,
                "data_as_of": stmt.excluded.data_as_of,
            },
        )
        await conn.execute(stmt)
    return len(db_rows)
