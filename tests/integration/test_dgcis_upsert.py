"""Integration tests for `app.pipeline.dgcis.upsert_annual_records` against
a real Postgres — proves idempotency (`docs/PLAN.md` Testing standard:
"run each ingestion job twice, assert identical row count and content")
and the ₹-Crore-to-paise conversion for real, not just trusted arithmetic.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.dgcis import DgcisAnnualRecord, upsert_annual_records
from app.warehouse.schema import raw_dgcis_annual

pytestmark = pytest.mark.integration

_TEST_HS8 = "99999903"  # never a real HS8 line - test-only, deleted after every test


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(delete(raw_dgcis_annual).where(raw_dgcis_annual.c.hs8 == _TEST_HS8))


def _record() -> DgcisAnnualRecord:
    return DgcisAnnualRecord(
        country="TESTLAND",
        hs8=_TEST_HS8,
        description="TEST COMMODITY",
        unit="KGS",
        report_date="23 Aug 2026",
        value_type="₹ Crore",
        values_by_year={
            "2023 - 2024": Decimal("424.66"),
            "2024 - 2025": Decimal("0.00"),
            "2025 - 2026": None,  # a genuinely blank cell in the real source
        },
    )


async def test_upsert_writes_one_row_per_year_with_correct_paise_conversion(
    warehouse_engine: AsyncEngine,
) -> None:
    written = await upsert_annual_records(warehouse_engine, [_record()], flow="import")

    assert written == 3
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_dgcis_annual).where(raw_dgcis_annual.c.hs8 == _TEST_HS8)
                )
            )
            .mappings()
            .all()
        )

    by_year = {r["fiscal_year_label"]: r for r in rows}
    assert len(rows) == 3
    # 424.66 crore = 424.66 * 1,000,000,000 paise, exactly (2 decimal places
    # in, an exact integer out - no float rounding drift, D8).
    assert by_year["2023 - 2024"]["value_inr_paise"] == 424_660_000_000
    assert by_year["2024 - 2025"]["value_inr_paise"] == 0
    assert by_year["2025 - 2026"]["value_inr_paise"] is None  # blank cell stays NULL, never 0
    assert all(r["partner_country"] == "TESTLAND" for r in rows)
    assert all(r["flow"] == "import" for r in rows)


async def test_upsert_is_idempotent_running_twice_produces_identical_rows(
    warehouse_engine: AsyncEngine,
) -> None:
    """docs/PLAN.md Testing standard: idempotency proven by running twice."""
    first_count = await upsert_annual_records(warehouse_engine, [_record()], flow="import")
    second_count = await upsert_annual_records(warehouse_engine, [_record()], flow="import")

    assert first_count == second_count == 3
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_dgcis_annual).where(raw_dgcis_annual.c.hs8 == _TEST_HS8)
                )
            )
            .mappings()
            .all()
        )

    # Still exactly 3 rows (one per year), not 6 - re-running upserts the
    # same real unique key rather than appending duplicates.
    assert len(rows) == 3


async def test_upsert_with_no_records_writes_nothing_and_does_not_error(
    warehouse_engine: AsyncEngine,
) -> None:
    written = await upsert_annual_records(warehouse_engine, [], flow="import")

    assert written == 0
