"""Integration test for `app.pipeline.agmarknet.upsert_agmarknet_records`
against a real Postgres — idempotency, per this project's Testing
standard.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.agmarknet import AgmarknetRecord, upsert_agmarknet_records
from app.warehouse.schema import raw_agmarknet_prices

pytestmark = pytest.mark.integration

_TEST_COMMODITY = (
    "___test-commodity___"  # never a real commodity - test-only, deleted after every test
)


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(raw_agmarknet_prices).where(raw_agmarknet_prices.c.commodity == _TEST_COMMODITY)
        )


def _record(*, market: str = "Test Market", modal_price: int | None = 250_000) -> AgmarknetRecord:
    return AgmarknetRecord(
        price_date=date(2023, 2, 21),
        commodity=_TEST_COMMODITY,
        market=market,
        state="Test State",
        modal_price_inr_paise_per_qtl=modal_price,
        raw_payload={"Commodity": _TEST_COMMODITY, "Market": market},
    )


async def test_upsert_writes_a_real_row(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_agmarknet_records(warehouse_engine, [_record()])

    assert written == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_agmarknet_prices).where(
                        raw_agmarknet_prices.c.commodity == _TEST_COMMODITY
                    )
                )
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1
    assert rows[0]["market"] == "Test Market"
    assert rows[0]["modal_price_inr_paise_per_qtl"] == 250_000


async def test_upsert_preserves_a_missing_modal_price_as_null_not_zero(
    warehouse_engine: AsyncEngine,
) -> None:
    await upsert_agmarknet_records(warehouse_engine, [_record(modal_price=None)])

    async with warehouse_engine.connect() as conn:
        row = (
            await conn.execute(
                select(raw_agmarknet_prices.c.modal_price_inr_paise_per_qtl).where(
                    raw_agmarknet_prices.c.commodity == _TEST_COMMODITY
                )
            )
        ).scalar_one()

    assert row is None  # never coerced to 0


async def test_upsert_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    record = _record()

    first = await upsert_agmarknet_records(warehouse_engine, [record])
    second = await upsert_agmarknet_records(warehouse_engine, [record])

    assert first == second == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_agmarknet_prices).where(
                        raw_agmarknet_prices.c.commodity == _TEST_COMMODITY
                    )
                )
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1  # not 2 - same real unique key, not a duplicate


async def test_upsert_with_no_records_writes_nothing(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_agmarknet_records(warehouse_engine, [])

    assert written == 0
