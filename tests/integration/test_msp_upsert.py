"""Integration test for `app.pipeline.msp.upsert_msp_records` against a
real Postgres — idempotency, per this project's Testing standard.
"""

from __future__ import annotations

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.msp import MspRecord, upsert_msp_records
from app.warehouse.schema import raw_msp_records

pytestmark = pytest.mark.integration

_TEST_COMMODITY = (
    "___test-commodity___"  # never a real commodity - test-only, deleted after every test
)


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(raw_msp_records).where(raw_msp_records.c.commodity == _TEST_COMMODITY)
        )


def _record(*, year_label: str = "2017-18", msp: int | None = 155_000) -> MspRecord:
    return MspRecord(
        crops="Test Crops",
        commodity=_TEST_COMMODITY,
        year_label=year_label,
        cost_inr_paise_per_qtl=111_700,
        msp_inr_paise_per_qtl=msp,
        raw_payload={"commodity": _TEST_COMMODITY},
    )


async def test_upsert_writes_a_real_row(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_msp_records(warehouse_engine, [_record()])

    assert written == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_msp_records).where(raw_msp_records.c.commodity == _TEST_COMMODITY)
                )
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1
    assert rows[0]["year_label"] == "2017-18"
    assert rows[0]["msp_inr_paise_per_qtl"] == 155_000


async def test_upsert_preserves_a_missing_msp_as_null_not_zero(
    warehouse_engine: AsyncEngine,
) -> None:
    await upsert_msp_records(warehouse_engine, [_record(msp=None)])

    async with warehouse_engine.connect() as conn:
        row = (
            await conn.execute(
                select(raw_msp_records.c.msp_inr_paise_per_qtl).where(
                    raw_msp_records.c.commodity == _TEST_COMMODITY
                )
            )
        ).scalar_one()

    assert row is None  # never coerced to 0


async def test_upsert_keeps_both_year_pairs_as_separate_rows(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_msp_records(
        warehouse_engine,
        [_record(year_label="2017-18"), _record(year_label="2022-23", msp=204_000)],
    )

    assert written == 2
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_msp_records).where(raw_msp_records.c.commodity == _TEST_COMMODITY)
                )
            )
            .mappings()
            .all()
        )

    assert {r["year_label"] for r in rows} == {"2017-18", "2022-23"}


async def test_upsert_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    record = _record()

    first = await upsert_msp_records(warehouse_engine, [record])
    second = await upsert_msp_records(warehouse_engine, [record])

    assert first == second == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_msp_records).where(raw_msp_records.c.commodity == _TEST_COMMODITY)
                )
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1  # not 2 - same real unique key, not a duplicate


async def test_upsert_with_no_records_writes_nothing(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_msp_records(warehouse_engine, [])

    assert written == 0
