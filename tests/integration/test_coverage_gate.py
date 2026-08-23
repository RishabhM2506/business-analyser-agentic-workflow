"""Integration tests for `app.warehouse.coverage_gate` against a real
Postgres — §9's 30% `QTY_MISSING` threshold, boundary-tested, plus the
zero-`expected_cells` fail-closed path and `degraded` detection.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.warehouse.coverage_gate import evaluate_coverage, upsert_coverage_summary
from app.warehouse.schema import analytics_coverage_summary, normalized_trade_flows

pytestmark = pytest.mark.integration

_TEST_HS6 = "999996"  # never a real HS6 - test-only, deleted after every test


def _row(*, year: int, partner: str, status: str) -> dict[str, object]:
    return {
        "source": "dgcis",
        "hs6": _TEST_HS6,
        "hs8": _TEST_HS6 + "00",
        "hs_revision": "ITC-HS",
        "flow": "import",
        "period_month": date(year, 1, 1),
        "calendar": "FY",
        "partner_country_code": partner,
        "basis": "CIF",
        "currency": "INR",
        "universe": "india-customs",
        "dataset_version": "dgcis-annual-v1",
        "is_provisional": False,
        "status": status,
        "status_detail": None,
        "value_inr_paise": 1_000_000_000 if status in ("OK", "QTY_MISSING") else None,
        "value_original_currency_paise": 1_000_000_000 if status in ("OK", "QTY_MISSING") else None,
        "fx_rate_used": None,
        "fx_rate_date": None,
        "quantity_kg": None,
    }


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(normalized_trade_flows).where(normalized_trade_flows.c.hs6 == _TEST_HS6)
        )
        await conn.execute(
            delete(analytics_coverage_summary).where(analytics_coverage_summary.c.hs6 == _TEST_HS6)
        )


async def test_gate_passes_when_qty_missing_pct_is_below_30_percent(
    warehouse_engine: AsyncEngine,
) -> None:
    # 1 year x 10 tracked partners = 10 expected cells; 2 QTY_MISSING = 20% < 30%.
    async with warehouse_engine.begin() as conn:
        rows = [_row(year=2023, partner=str(i), status="QTY_MISSING") for i in range(2)]
        rows += [_row(year=2023, partner=str(i), status="OK") for i in range(2, 10)]
        await conn.execute(insert(normalized_trade_flows).values(rows))

    result = await evaluate_coverage(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        tracked_partners=10,
    )

    assert result.expected_cells == 10
    assert result.qty_missing_cells == 2
    assert result.qty_missing_pct == 20
    assert result.gate_passed is True


async def test_gate_fails_at_exactly_30_point_1_percent(warehouse_engine: AsyncEngine) -> None:
    # 1000 expected cells, 301 QTY_MISSING = 30.1% > 30%.
    async with warehouse_engine.begin() as conn:
        rows = [_row(year=2023, partner=str(i), status="QTY_MISSING") for i in range(301)]
        await conn.execute(insert(normalized_trade_flows).values(rows))

    result = await evaluate_coverage(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        tracked_partners=1000,
    )

    assert result.qty_missing_pct == Decimal("30.1")
    assert result.gate_passed is False


async def test_gate_passes_at_exactly_30_percent_the_boundary_is_not_off_by_one(
    warehouse_engine: AsyncEngine,
) -> None:
    """§9: "if qty_missing_pct > 0.30" - strictly greater-than, so exactly
    30% still passes."""
    async with warehouse_engine.begin() as conn:
        rows = [_row(year=2023, partner=str(i), status="QTY_MISSING") for i in range(30)]
        rows += [_row(year=2023, partner=str(i), status="OK") for i in range(30, 100)]
        await conn.execute(insert(normalized_trade_flows).values(rows))

    result = await evaluate_coverage(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        tracked_partners=100,
    )

    assert result.qty_missing_pct == 30
    assert result.gate_passed is True


async def test_gate_fails_closed_when_expected_cells_is_zero(warehouse_engine: AsyncEngine) -> None:
    result = await evaluate_coverage(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        tracked_partners=0,
    )

    assert result.expected_cells == 0
    assert result.qty_missing_pct is None  # undefined, never fabricated as 0% or 100%
    assert result.gate_passed is False


async def test_degraded_is_true_when_any_fetch_failed_cell_is_present(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                [
                    _row(year=2023, partner="1", status="OK"),
                    _row(year=2023, partner="2", status="FETCH_FAILED"),
                ]
            )
        )

    result = await evaluate_coverage(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        tracked_partners=2,
    )

    assert result.fetch_failed_cells == 1
    assert result.degraded is True


async def test_upsert_coverage_summary_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    result = await evaluate_coverage(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        tracked_partners=0,
    )

    await upsert_coverage_summary(warehouse_engine, result)
    await upsert_coverage_summary(warehouse_engine, result)

    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(analytics_coverage_summary).where(
                        analytics_coverage_summary.c.hs6 == _TEST_HS6
                    )
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1  # same real unique key, not a duplicate
    assert rows[0]["gate_passed"] is False
