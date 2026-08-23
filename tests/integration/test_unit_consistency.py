"""Integration tests for `app.report.unit_consistency` against a real
Postgres — no pure logic to unit-test in isolation here (the module is a
thin query + conditional update over real tables), so this is the only
test file for it, per this project's established split.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.report.unit_consistency import check_unit_consistency, mark_unit_mismatch
from app.warehouse.schema import normalized_trade_flows, raw_dgcis_annual

pytestmark = pytest.mark.integration

_TEST_HS6 = "999997"  # never a real HS6 - test-only, deleted after every test
_TEST_HS8_A = "99999701"
_TEST_HS8_B = "99999702"


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(raw_dgcis_annual).where(raw_dgcis_annual.c.hs8.startswith(_TEST_HS6))
        )
        await conn.execute(
            delete(normalized_trade_flows).where(normalized_trade_flows.c.hs6 == _TEST_HS6)
        )


def _raw_row(
    *, hs8: str, unit: str | None, fiscal_year_label: str = "2023 - 2024"
) -> dict[str, object]:
    return {
        "scraped_at": datetime.now(UTC),
        "fiscal_year_label": fiscal_year_label,
        "hs8": hs8,
        "flow": "import",
        "partner_country": "TESTLAND",
        "description": "TEST COMMODITY",
        "unit": unit,
        "value_inr_paise": 1_000_000_000,
        "raw_payload": {},
    }


def _normalized_row(*, status: str = "OK") -> dict[str, object]:
    return {
        "source": "dgcis",
        "hs6": _TEST_HS6,
        "hs8": _TEST_HS8_A,
        "hs_revision": "ITC-HS",
        "flow": "import",
        "period_month": date(2023, 1, 1),
        "calendar": "FY",
        "partner_country_code": "999",
        "basis": "CIF",
        "currency": "INR",
        "universe": "india-customs",
        "dataset_version": "dgcis-annual-v1",
        "is_provisional": False,
        "status": status,
        "status_detail": None,
        "value_inr_paise": 1_000_000_000,
        "value_original_currency_paise": 1_000_000_000,
        "fx_rate_used": None,
        "fx_rate_date": None,
        "quantity_kg": None,
    }


async def test_check_unit_consistency_is_consistent_when_all_siblings_agree(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_dgcis_annual).values(
                [
                    _raw_row(hs8=_TEST_HS8_A, unit="KGS"),
                    _raw_row(hs8=_TEST_HS8_B, unit="KGS"),
                ]
            )
        )

    result = await check_unit_consistency(warehouse_engine, hs6=_TEST_HS6)

    assert result.is_consistent is True
    assert result.units_by_hs8 == {_TEST_HS8_A: {"KGS"}, _TEST_HS8_B: {"KGS"}}


async def test_check_unit_consistency_detects_disagreeing_siblings(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_dgcis_annual).values(
                [
                    _raw_row(hs8=_TEST_HS8_A, unit="KGS"),
                    _raw_row(hs8=_TEST_HS8_B, unit="TON"),
                ]
            )
        )

    result = await check_unit_consistency(warehouse_engine, hs6=_TEST_HS6)

    assert result.is_consistent is False
    assert result.units_by_hs8[_TEST_HS8_A] == {"KGS"}
    assert result.units_by_hs8[_TEST_HS8_B] == {"TON"}


async def test_check_unit_consistency_ignores_null_units(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_dgcis_annual).values(
                [
                    _raw_row(hs8=_TEST_HS8_A, unit="KGS"),
                    _raw_row(hs8=_TEST_HS8_B, unit=None),
                ]
            )
        )

    result = await check_unit_consistency(warehouse_engine, hs6=_TEST_HS6)

    assert result.is_consistent is True  # NULL is not evidence of a mismatch either way
    assert _TEST_HS8_B not in result.units_by_hs8


async def test_mark_unit_mismatch_updates_eligible_rows_only(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_dgcis_annual).values(
                [
                    _raw_row(hs8=_TEST_HS8_A, unit="KGS"),
                    _raw_row(hs8=_TEST_HS8_B, unit="TON"),
                ]
            )
        )
        await conn.execute(
            insert(normalized_trade_flows).values(
                [
                    _normalized_row(status="OK"),
                    {**_normalized_row(status="FETCH_FAILED"), "partner_country_code": "998"},
                ]
            )
        )

    updated = await mark_unit_mismatch(warehouse_engine, hs6=_TEST_HS6, flow="import")

    assert updated == 1  # only the OK row - FETCH_FAILED carries a stronger, unrelated signal
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
    by_partner = {r["partner_country_code"]: r for r in rows}
    assert by_partner["999"]["status"] == "UNIT_MISMATCH"
    assert "KGS" in by_partner["999"]["status_detail"]
    assert "TON" in by_partner["999"]["status_detail"]
    assert by_partner["998"]["status"] == "FETCH_FAILED"  # untouched


async def test_mark_unit_mismatch_is_a_noop_when_consistent(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_dgcis_annual).values(
                [
                    _raw_row(hs8=_TEST_HS8_A, unit="KGS"),
                    _raw_row(hs8=_TEST_HS8_B, unit="KGS"),
                ]
            )
        )
        await conn.execute(insert(normalized_trade_flows).values(_normalized_row(status="OK")))

    updated = await mark_unit_mismatch(warehouse_engine, hs6=_TEST_HS6, flow="import")

    assert updated == 0
