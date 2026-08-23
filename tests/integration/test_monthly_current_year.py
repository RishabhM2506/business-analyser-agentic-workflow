"""Integration tests for `app.report.monthly_current_year` against a real
Postgres — the always-12-rows contract, real MoM/YoY computation
(including the January-needs-prior-December cross-year case), the
multi-HS8 value-sum-but-never-quantity-sum rule, and upsert idempotency.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.report.monthly_current_year import (
    compute_monthly_current_year,
    upsert_monthly_current_year,
)
from app.warehouse.schema import analytics_monthly_current_year, raw_dgcis_monthly

pytestmark = pytest.mark.integration

_TEST_HS6 = "999990"  # never a real HS6 - test-only, deleted after every test
_TEST_HS8 = _TEST_HS6 + "01"


def _raw_row(
    *, hs8: str = _TEST_HS8, month: date, value: int | None, quantity=None, marker: str = "R"
) -> dict[str, object]:
    return {
        "scraped_at": datetime.now(UTC),
        "fiscal_year": "2022 - 2023",
        "calendar_month": month,
        "hs8": hs8,
        "flow": "import",
        "partner_country": "ALL_PARTNERS",
        "value_inr_paise": value,
        "quantity": quantity,
        "unit": "KGS" if quantity is not None else None,
        "raw_payload": {"marker": marker},
    }


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(raw_dgcis_monthly).where(raw_dgcis_monthly.c.hs8.startswith(_TEST_HS6))
        )
        await conn.execute(
            delete(analytics_monthly_current_year).where(
                analytics_monthly_current_year.c.hs6 == _TEST_HS6
            )
        )


async def test_compute_always_returns_exactly_12_rows(warehouse_engine: AsyncEngine) -> None:
    results = await compute_monthly_current_year(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )

    assert len(results) == 12
    assert [r.month for r in results] == [date(2023, m, 1) for m in range(1, 13)]
    assert all(r.status == "NOT_YET_PUBLISHED" for r in results)  # no raw data seeded at all


async def test_compute_real_ok_month_and_month_over_month_delta(
    warehouse_engine: AsyncEngine,
) -> None:
    from decimal import Decimal

    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_dgcis_monthly).values(
                [
                    _raw_row(
                        month=date(2023, 5, 1),
                        value=100_000_000_000,
                        quantity=Decimal("1000"),
                    ),
                    _raw_row(
                        month=date(2023, 6, 1),
                        value=110_000_000_000,
                        quantity=Decimal("1100"),
                    ),
                ]
            )
        )

    results = await compute_monthly_current_year(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )

    by_month = {r.month: r for r in results}
    june = by_month[date(2023, 6, 1)]
    assert june.status == "OK"
    assert june.value_inr_paise == 110_000_000_000
    assert june.mom_change_pct == 10  # (110-100)/100 * 100


async def test_compute_january_uses_prior_decembers_value_for_month_over_month(
    warehouse_engine: AsyncEngine,
) -> None:
    from decimal import Decimal

    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_dgcis_monthly).values(
                [
                    _raw_row(
                        month=date(2022, 12, 1),
                        value=50_000_000_000,
                        quantity=Decimal("500"),
                    ),
                    _raw_row(
                        month=date(2023, 1, 1),
                        value=60_000_000_000,
                        quantity=Decimal("600"),
                    ),
                ]
            )
        )

    results = await compute_monthly_current_year(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )

    january = next(r for r in results if r.month == date(2023, 1, 1))
    assert january.mom_change_pct == 20  # (60-50)/50 * 100, vs Dec 2022


async def test_compute_year_over_year_uses_the_same_month_the_prior_year(
    warehouse_engine: AsyncEngine,
) -> None:
    from decimal import Decimal

    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_dgcis_monthly).values(
                [
                    _raw_row(
                        month=date(2022, 6, 1),
                        value=100_000_000_000,
                        quantity=Decimal("1000"),
                    ),
                    _raw_row(
                        month=date(2023, 6, 1),
                        value=150_000_000_000,
                        quantity=Decimal("1500"),
                    ),
                ]
            )
        )

    results = await compute_monthly_current_year(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )

    june = next(r for r in results if r.month == date(2023, 6, 1))
    assert june.yoy_same_month_pct == 50  # (150-100)/100 * 100


async def test_compute_sums_value_but_never_quantity_across_multiple_hs8_lines(
    warehouse_engine: AsyncEngine,
) -> None:
    from decimal import Decimal

    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_dgcis_monthly).values(
                [
                    _raw_row(
                        hs8=_TEST_HS6 + "01",
                        month=date(2023, 6, 1),
                        value=100_000_000_000,
                        quantity=Decimal("1000"),
                    ),
                    _raw_row(
                        hs8=_TEST_HS6 + "02",
                        month=date(2023, 6, 1),
                        value=50_000_000_000,
                        quantity=Decimal("500"),
                    ),
                ]
            )
        )

    results = await compute_monthly_current_year(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )

    june = next(r for r in results if r.month == date(2023, 6, 1))
    assert june.value_inr_paise == 150_000_000_000  # value sums correctly
    assert june.status == "QTY_MISSING"  # quantity never guessed across lines


async def test_upsert_writes_all_12_months_and_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    results = await compute_monthly_current_year(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )
    data_as_of = datetime.now(UTC)

    first = await upsert_monthly_current_year(warehouse_engine, results, data_as_of=data_as_of)
    second = await upsert_monthly_current_year(warehouse_engine, results, data_as_of=data_as_of)

    assert first == second == 12
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(analytics_monthly_current_year).where(
                        analytics_monthly_current_year.c.hs6 == _TEST_HS6
                    )
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 12  # same real unique key per month, not duplicated


async def test_upsert_with_no_rows_writes_nothing(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_monthly_current_year(warehouse_engine, [], data_as_of=datetime.now(UTC))

    assert written == 0
