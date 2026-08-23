"""Integration test for `app.pipeline.comtrade_mirror.upsert_comtrade_records`
against a real Postgres — idempotency, per this project's Testing standard.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.comtrade_mirror import upsert_comtrade_records
from app.warehouse.schema import raw_comtrade_records

pytestmark = pytest.mark.integration

_TEST_CMD_CODE = "999999"  # never a real HS6 - test-only, deleted after every test


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(raw_comtrade_records).where(raw_comtrade_records.c.cmd_code == _TEST_CMD_CODE)
        )


def _raw_row(*, reporter: str, partner: str, period: str = "2023") -> dict[str, object]:
    return {
        "period": period,
        "reporterCode": reporter,
        "partnerCode": partner,
        "flowCode": "M",
        "cmdCode": _TEST_CMD_CODE,
        "primaryValue": "1234.56",
        "netWgt": "100.5",
        "isReported": True,
    }


async def test_upsert_writes_real_rows(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_comtrade_records(
        warehouse_engine, [_raw_row(reporter="699", partner="0")]
    )

    assert written == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_comtrade_records).where(
                        raw_comtrade_records.c.cmd_code == _TEST_CMD_CODE
                    )
                )
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1
    assert rows[0]["reporter_code"] == "699"
    assert rows[0]["partner_code"] == "0"
    assert rows[0]["primary_value_usd"] == Decimal("1234.56")


async def test_upsert_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    record = _raw_row(reporter="699", partner="0")

    first = await upsert_comtrade_records(warehouse_engine, [record])
    second = await upsert_comtrade_records(warehouse_engine, [record])

    assert first == second == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(raw_comtrade_records).where(
                        raw_comtrade_records.c.cmd_code == _TEST_CMD_CODE
                    )
                )
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1  # not 2 - same real unique key, not a duplicate


async def test_upsert_with_no_records_writes_nothing(warehouse_engine: AsyncEngine) -> None:
    written = await upsert_comtrade_records(warehouse_engine, [])

    assert written == 0
