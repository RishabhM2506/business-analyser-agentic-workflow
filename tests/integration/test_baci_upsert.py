"""Integration test for `app.pipeline.baci.upsert_baci_records` against a
real Postgres — idempotency, per this project's Testing standard.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.baci import BaciRecord, upsert_baci_records
from app.warehouse.schema import raw_baci_records

pytestmark = pytest.mark.integration

_TEST_HS6 = "999992"  # never a real HS6 - test-only, deleted after every test
_TEST_VINTAGE = "TEST_V000000"


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(delete(raw_baci_records).where(raw_baci_records.c.hs6 == _TEST_HS6))


def _record(*, importer: str = "36") -> BaciRecord:
    return BaciRecord(
        vintage=_TEST_VINTAGE,
        hs_revision="22",
        year=2022,
        exporter_code="699",
        importer_code=importer,
        hs6=_TEST_HS6,
        value_fob_usd=Decimal("1234.56"),
        quantity_kg=Decimal("100.500"),
    )


async def test_upsert_writes_a_real_row(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_baci_records(warehouse_engine, [_record()])

    assert written == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_baci_records).where(raw_baci_records.c.hs6 == _TEST_HS6)
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1
    assert rows[0]["exporter_code"] == "699"
    assert rows[0]["value_fob_usd"] == Decimal("1234.56")


async def test_upsert_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    record = _record()

    first = await upsert_baci_records(warehouse_engine, [record])
    second = await upsert_baci_records(warehouse_engine, [record])

    assert first == second == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_baci_records).where(raw_baci_records.c.hs6 == _TEST_HS6)
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1  # same real unique key, not a duplicate


async def test_upsert_keeps_distinct_importers_as_distinct_rows(
    warehouse_engine: AsyncEngine,
) -> None:
    written = await upsert_baci_records(
        warehouse_engine, [_record(importer="36"), _record(importer="124")]
    )

    assert written == 2
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_baci_records).where(raw_baci_records.c.hs6 == _TEST_HS6)
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 2


async def test_upsert_with_no_records_writes_nothing(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_baci_records(warehouse_engine, [])

    assert written == 0
