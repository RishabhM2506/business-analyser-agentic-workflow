"""Integration tests for `app.pipeline.normalize.normalize_baci_rows`
against a real Postgres — flow derivation from the exporter/importer
pair, the FOB-always basis, real FX conversion, and upsert idempotency.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.normalize import normalize_baci_rows
from app.warehouse.schema import normalized_trade_flows, raw_baci_records

pytestmark = pytest.mark.integration

_TEST_HS6 = "999991"  # never a real HS6 - test-only, deleted after every test
_TEST_VINTAGE = "TEST_202601"


def _row(
    *, exporter: str, importer: str, year: int = 2023, value_usd, quantity_kg=None
) -> dict[str, object]:
    return {
        "loaded_at": datetime.now(UTC),
        "vintage": _TEST_VINTAGE,
        "hs_revision": "22",
        "year": year,
        "exporter_code": exporter,
        "importer_code": importer,
        "hs6": _TEST_HS6,
        "value_fob_usd": value_usd,
        "quantity_kg": quantity_kg,
    }


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(delete(raw_baci_records).where(raw_baci_records.c.hs6 == _TEST_HS6))
        await conn.execute(
            delete(normalized_trade_flows).where(normalized_trade_flows.c.hs6 == _TEST_HS6)
        )


async def test_normalize_baci_rows_derives_import_flow_when_india_is_importer(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_baci_records).values(
                _row(
                    exporter="156",
                    importer="699",
                    value_usd=Decimal("1000.00"),
                    quantity_kg=Decimal("50"),
                )
            )
        )

    written = await normalize_baci_rows(
        warehouse_engine, hs6=_TEST_HS6, fx_rates={2023: (Decimal("83.000000"), date(2023, 6, 15))}
    )

    assert written == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(normalized_trade_flows).where(normalized_trade_flows.c.hs6 == _TEST_HS6)
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "baci"
    assert row["flow"] == "import"
    assert row["partner_country_code"] == "156"
    assert (
        row["basis"] == "FOB"
    )  # always FOB, even for import - BACI's own CEPII-adjusted convention
    assert row["currency"] == "USD"
    assert row["calendar"] == "CY"
    assert row["hs8"] is None
    assert row["value_original_currency_paise"] == 100_000
    assert row["value_inr_paise"] == 8_300_000
    assert row["status"] == "OK"


async def test_normalize_baci_rows_derives_export_flow_when_india_is_exporter(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_baci_records).values(
                _row(
                    exporter="699",
                    importer="124",
                    value_usd=Decimal("500.00"),
                    quantity_kg=Decimal("25"),
                )
            )
        )

    written = await normalize_baci_rows(
        warehouse_engine, hs6=_TEST_HS6, fx_rates={2023: (Decimal("83.000000"), date(2023, 6, 15))}
    )

    assert written == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(normalized_trade_flows).where(normalized_trade_flows.c.hs6 == _TEST_HS6)
                )
            )
            .mappings()
            .all()
        )
    assert rows[0]["flow"] == "export"
    assert rows[0]["partner_country_code"] == "124"
    assert rows[0]["basis"] == "FOB"


async def test_normalize_baci_rows_sets_qty_missing_when_quantity_is_null(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_baci_records).values(
                _row(exporter="156", importer="699", value_usd=Decimal("1000.00"), quantity_kg=None)
            )
        )

    await normalize_baci_rows(
        warehouse_engine, hs6=_TEST_HS6, fx_rates={2023: (Decimal("83.000000"), date(2023, 6, 15))}
    )

    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(normalized_trade_flows).where(normalized_trade_flows.c.hs6 == _TEST_HS6)
                )
            )
            .mappings()
            .all()
        )
    assert rows[0]["status"] == "QTY_MISSING"


async def test_normalize_baci_rows_skips_a_year_with_no_fx_rate(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_baci_records).values(
                _row(exporter="156", importer="699", year=2023, value_usd=Decimal("1000.00"))
            )
        )

    written = await normalize_baci_rows(warehouse_engine, hs6=_TEST_HS6, fx_rates={})

    assert written == 0


async def test_normalize_baci_rows_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_baci_records).values(
                _row(exporter="156", importer="699", value_usd=Decimal("1000.00"))
            )
        )

    fx_rates = {2023: (Decimal("83.000000"), date(2023, 6, 15))}
    first = await normalize_baci_rows(warehouse_engine, hs6=_TEST_HS6, fx_rates=fx_rates)
    second = await normalize_baci_rows(warehouse_engine, hs6=_TEST_HS6, fx_rates=fx_rates)

    assert first == second == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(normalized_trade_flows).where(normalized_trade_flows.c.hs6 == _TEST_HS6)
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1  # same real unique key, not a duplicate
