"""Integration tests for `app.report.unit_value` against a real Postgres —
real unit-value computation, the calendar-adjacency rule for deltas (a gap
year must not silently be skipped over), and upsert idempotency.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.normalize import COMTRADE_DATASET_VERSION_REPORTER_ROLE
from app.report.unit_value import compute_unit_value_series, upsert_unit_value_series
from app.warehouse.schema import analytics_unit_value_series, normalized_trade_flows

pytestmark = pytest.mark.integration

_TEST_HS6 = "999994"  # never a real HS6 - test-only, deleted after every test


def _row(*, year: int, value_inr_paise, value_usd_paise, fx_rate, quantity_kg) -> dict[str, object]:
    return {
        "source": "comtrade",
        "hs6": _TEST_HS6,
        "hs8": None,
        "hs_revision": "H6",
        "flow": "import",
        "period_month": date(year, 1, 1),
        "calendar": "CY",
        "partner_country_code": "0",
        "basis": "CIF",
        "currency": "USD",
        "universe": "un-comtrade-mirror",
        "dataset_version": COMTRADE_DATASET_VERSION_REPORTER_ROLE,
        "is_provisional": False,
        "status": "OK" if quantity_kg is not None else "QTY_MISSING",
        "status_detail": None,
        "value_inr_paise": value_inr_paise,
        "value_original_currency_paise": value_usd_paise,
        "fx_rate_used": fx_rate,
        "fx_rate_date": date(year, 6, 15) if fx_rate is not None else None,
        "quantity_kg": quantity_kg,
    }


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(normalized_trade_flows).where(normalized_trade_flows.c.hs6 == _TEST_HS6)
        )
        await conn.execute(
            delete(analytics_unit_value_series).where(
                analytics_unit_value_series.c.hs6 == _TEST_HS6
            )
        )


async def test_compute_unit_value_series_computes_real_unit_value_when_data_present(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                _row(
                    year=2023,
                    value_inr_paise=8_300_000,
                    value_usd_paise=100_000,
                    fx_rate=Decimal("83.000000"),
                    quantity_kg=Decimal("10"),
                )
            )
        )

    results = await compute_unit_value_series(
        warehouse_engine, hs6=_TEST_HS6, flow="import", years=[2023]
    )

    assert len(results) == 1
    assert results[0].coverage_gate_passed is True
    assert results[0].unit_value_inr_paise_per_kg == Decimal("830000")  # 8_300_000 / 10
    assert results[0].delta_value_pct is None  # no prior year to compare


async def test_compute_unit_value_series_fails_gate_when_year_has_no_data(
    warehouse_engine: AsyncEngine,
) -> None:
    results = await compute_unit_value_series(
        warehouse_engine, hs6=_TEST_HS6, flow="import", years=[2023]
    )

    assert len(results) == 1
    assert results[0].coverage_gate_passed is False
    assert results[0].unit_value_inr_paise_per_kg is None


async def test_compute_unit_value_series_computes_decomposition_for_adjacent_years(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                [
                    _row(
                        year=2022,
                        value_inr_paise=8_300_000,
                        value_usd_paise=100_000,
                        fx_rate=Decimal("83.000000"),
                        quantity_kg=Decimal("10"),
                    ),
                    _row(
                        year=2023,
                        value_inr_paise=16_600_000,
                        value_usd_paise=200_000,
                        fx_rate=Decimal("83.000000"),
                        quantity_kg=Decimal("20"),
                    ),
                ]
            )
        )

    results = await compute_unit_value_series(
        warehouse_engine, hs6=_TEST_HS6, flow="import", years=[2022, 2023]
    )

    by_year = {r.year: r for r in results}
    assert by_year[2022].delta_value_pct is None  # no 2021 to compare against
    # 2023: qty doubled, price/kg and fx unchanged -> delta_value_pct ~100%,
    # delta_from_qty_pct ~100%, price/fx deltas ~0%.
    assert by_year[2023].delta_value_pct == Decimal("100")
    assert by_year[2023].delta_from_price_pct == Decimal("0")
    assert by_year[2023].delta_from_fx_pct == Decimal("0")
    assert by_year[2023].delta_from_qty_pct > Decimal("69")  # ln(2)*100 ~= 69.3


async def test_compute_unit_value_series_does_not_skip_over_a_gap_year(
    warehouse_engine: AsyncEngine,
) -> None:
    """A missing 2022 must never let 2023's delta silently compare against
    2021 instead - that would misreport a two-year change as one year."""
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                [
                    _row(
                        year=2021,
                        value_inr_paise=8_300_000,
                        value_usd_paise=100_000,
                        fx_rate=Decimal("83.000000"),
                        quantity_kg=Decimal("10"),
                    ),
                    # 2022 deliberately absent - a real gap
                    _row(
                        year=2023,
                        value_inr_paise=16_600_000,
                        value_usd_paise=200_000,
                        fx_rate=Decimal("83.000000"),
                        quantity_kg=Decimal("20"),
                    ),
                ]
            )
        )

    results = await compute_unit_value_series(
        warehouse_engine, hs6=_TEST_HS6, flow="import", years=[2021, 2022, 2023]
    )

    by_year = {r.year: r for r in results}
    assert by_year[2022].coverage_gate_passed is False
    assert by_year[2023].delta_value_pct is None  # not silently compared to 2021


async def test_upsert_unit_value_series_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                _row(
                    year=2023,
                    value_inr_paise=8_300_000,
                    value_usd_paise=100_000,
                    fx_rate=Decimal("83.000000"),
                    quantity_kg=Decimal("10"),
                )
            )
        )

    results = await compute_unit_value_series(
        warehouse_engine, hs6=_TEST_HS6, flow="import", years=[2023]
    )
    first = await upsert_unit_value_series(warehouse_engine, results)
    second = await upsert_unit_value_series(warehouse_engine, results)

    assert first == second == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(analytics_unit_value_series).where(
                        analytics_unit_value_series.c.hs6 == _TEST_HS6
                    )
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1
