"""Integration tests for `app.pipeline.dgcis.upsert_monthly_records`
against a real Postgres — idempotency, the `'ALL_PARTNERS'` sentinel, and
real fiscal-year-label derivation from the calendar month.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.dgcis import DgcisMonthlyRecord, upsert_monthly_records
from app.warehouse.schema import raw_dgcis_monthly

pytestmark = pytest.mark.integration

_TEST_HS8 = "99999903"  # never a real HS8 line - test-only, deleted after every test


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(delete(raw_dgcis_monthly).where(raw_dgcis_monthly.c.hs8 == _TEST_HS8))


def _record(*, month: date = date(2023, 6, 1), marker: str = "R") -> DgcisMonthlyRecord:
    return DgcisMonthlyRecord(
        hs8=_TEST_HS8,
        flow="import",
        calendar_month=month,
        value_inr_paise=166_500_000_000,
        quantity_kg=Decimal("6347970"),
        unit="KGS",
        marker=marker,
    )


async def test_upsert_writes_a_real_row_with_the_all_partners_sentinel(
    warehouse_engine: AsyncEngine,
) -> None:
    written = await upsert_monthly_records(warehouse_engine, [_record()])

    assert written == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_dgcis_monthly).where(raw_dgcis_monthly.c.hs8 == _TEST_HS8)
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1
    row = rows[0]
    assert row["partner_country"] == "ALL_PARTNERS"
    assert row["fiscal_year"] == "2023 - 2024"  # June -> FY starting that same April
    assert row["value_inr_paise"] == 166_500_000_000
    assert row["quantity"] == Decimal("6347970")
    assert row["unit"] == "KGS"
    assert row["raw_payload"] == {"marker": "R"}


async def test_upsert_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    record = _record()

    first = await upsert_monthly_records(warehouse_engine, [record])
    second = await upsert_monthly_records(warehouse_engine, [record])

    assert first == second == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_dgcis_monthly).where(raw_dgcis_monthly.c.hs8 == _TEST_HS8)
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1  # same real unique key, not a duplicate


async def test_upsert_a_january_month_uses_the_previous_aprils_fiscal_year(
    warehouse_engine: AsyncEngine,
) -> None:
    await upsert_monthly_records(warehouse_engine, [_record(month=date(2024, 1, 1))])

    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_dgcis_monthly).where(raw_dgcis_monthly.c.hs8 == _TEST_HS8)
                )
            )
            .mappings()
            .all()
        )
    assert rows[0]["fiscal_year"] == "2023 - 2024"


async def test_upsert_with_no_records_writes_nothing(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_monthly_records(warehouse_engine, [])

    assert written == 0
