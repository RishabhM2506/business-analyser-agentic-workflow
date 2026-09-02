"""Integration test for `app.pipeline.faostat.upsert_faostat_records`
against a real Postgres — idempotency, per this project's Testing
standard.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.faostat import FaostatRecord, upsert_faostat_records
from app.warehouse.schema import raw_faostat_records

pytestmark = pytest.mark.integration

_TEST_ITEM_CODE = "___test-item-code___"  # never a real item - test-only, deleted after every test


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(raw_faostat_records).where(raw_faostat_records.c.item_code == _TEST_ITEM_CODE)
        )


def _record(*, value: Decimal | None = Decimal("100"), flag: str | None = "A") -> FaostatRecord:
    return FaostatRecord(
        area_code="356",
        area="India",
        item_code=_TEST_ITEM_CODE,
        item="Test Item",
        element="Production",
        unit="t",
        year=2023,
        value=value,
        flag=flag,
    )


async def test_upsert_writes_a_real_row(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_faostat_records(warehouse_engine, [_record()])

    assert written == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_faostat_records).where(
                        raw_faostat_records.c.item_code == _TEST_ITEM_CODE
                    )
                )
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1
    assert rows[0]["area"] == "India"
    assert rows[0]["value"] == Decimal("100")
    assert rows[0]["flag"] == "A"


async def test_upsert_preserves_a_missing_value_as_null_not_zero(
    warehouse_engine: AsyncEngine,
) -> None:
    await upsert_faostat_records(warehouse_engine, [_record(value=None, flag="M")])

    async with warehouse_engine.connect() as conn:
        row = (
            await conn.execute(
                select(raw_faostat_records.c.value, raw_faostat_records.c.flag).where(
                    raw_faostat_records.c.item_code == _TEST_ITEM_CODE
                )
            )
        ).one()

    assert row.value is None  # never coerced to 0
    assert row.flag == "M"


async def test_upsert_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    record = _record()

    first = await upsert_faostat_records(warehouse_engine, [record])
    second = await upsert_faostat_records(warehouse_engine, [record])

    assert first == second == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_faostat_records).where(
                        raw_faostat_records.c.item_code == _TEST_ITEM_CODE
                    )
                )
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1  # not 2 - same real unique key, not a duplicate


async def test_upsert_with_no_records_writes_nothing(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_faostat_records(warehouse_engine, [])

    assert written == 0


async def test_upsert_batches_past_asyncpgs_real_parameter_limit(
    warehouse_engine: AsyncEngine,
) -> None:
    """Regression test for a real bug found live, 2026-08-25: a single
    real item ("Poppy seed", every country, every year) produced 5,760
    rows in one batch, which asyncpg rejected with
    `InterfaceError: the number of query arguments cannot exceed 32767`
    (11 columns x 5,760 rows = 63,360 params). Uses more rows than one
    real asyncpg batch (`_UPSERT_BATCH_SIZE` = 1000) allows, to prove the
    upsert function itself batches internally rather than requiring every
    caller to remember to."""
    records = [
        FaostatRecord(
            area_code=str(i),
            area=f"Area {i}",
            item_code=_TEST_ITEM_CODE,
            item="Test Item",
            element="Production",
            unit="t",
            year=2023,
            value=Decimal("1"),
            flag="A",
        )
        for i in range(1500)
    ]

    written = await upsert_faostat_records(warehouse_engine, records)

    assert written == 1500
    async with warehouse_engine.connect() as conn:
        count = (
            await conn.execute(
                select(raw_faostat_records.c.id).where(
                    raw_faostat_records.c.item_code == _TEST_ITEM_CODE
                )
            )
        ).rowcount
    assert count == 1500
