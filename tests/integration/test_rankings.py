"""Integration tests for `app.report.rankings` against a real Postgres —
real DGCIS-sourced ranking, rank-vs-NULL split, tie-breaking, and
idempotent upsert.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.report.rankings import compute_partner_rankings, upsert_partner_rankings
from app.warehouse.schema import analytics_partner_rankings, normalized_trade_flows

pytestmark = pytest.mark.integration

_TEST_HS6 = "999995"  # never a real HS6 - test-only, deleted after every test


def _row(*, partner: str, status: str, value: int | None) -> dict[str, object]:
    return {
        "source": "dgcis",
        "hs6": _TEST_HS6,
        "hs8": _TEST_HS6 + "00",
        "hs_revision": "ITC-HS",
        "flow": "import",
        "period_month": date(2023, 1, 1),
        "calendar": "FY",
        "partner_country_code": partner,
        "basis": "CIF",
        "currency": "INR",
        "universe": "india-customs",
        "dataset_version": "dgcis-annual-v1",
        "is_provisional": False,
        "status": status,
        "status_detail": None,
        "value_inr_paise": value,
        "value_original_currency_paise": value,
        "fx_rate_used": None,
        "fx_rate_date": None,
        "quantity_kg": None,
    }


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(normalized_trade_flows).where(normalized_trade_flows.c.hs6 == _TEST_HS6)
        )
        await conn.execute(
            delete(analytics_partner_rankings).where(analytics_partner_rankings.c.hs6 == _TEST_HS6)
        )


async def test_compute_partner_rankings_ranks_by_value_descending(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                [
                    _row(partner="792", status="OK", value=500),
                    _row(partner="156", status="OK", value=1000),
                    _row(partner="4", status="NOT_REPORTED", value=None),
                ]
            )
        )

    results = await compute_partner_rankings(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )

    by_partner = {r.partner_country_code: r for r in results}
    assert by_partner["156"].rank == 1
    assert by_partner["792"].rank == 2
    assert by_partner["4"].rank is None
    assert by_partner["4"].value_inr_paise is None


async def test_compute_partner_rankings_breaks_ties_by_partner_code(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                [
                    _row(partner="792", status="OK", value=500),
                    _row(partner="156", status="OK", value=500),
                ]
            )
        )

    results = await compute_partner_rankings(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )

    by_partner = {r.partner_country_code: r for r in results}
    assert by_partner["156"].rank == 1  # "156" < "792" lexicographically
    assert by_partner["792"].rank == 2


async def test_compute_partner_rankings_zero_and_qty_missing_are_rankable(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                [
                    _row(partner="792", status="QTY_MISSING", value=1000),
                    _row(partner="156", status="ZERO", value=0),
                ]
            )
        )

    results = await compute_partner_rankings(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )

    by_partner = {r.partner_country_code: r for r in results}
    assert by_partner["792"].rank == 1
    assert by_partner["156"].rank == 2


async def test_upsert_partner_rankings_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(_row(partner="792", status="OK", value=500))
        )

    results = await compute_partner_rankings(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )
    first = await upsert_partner_rankings(warehouse_engine, results)
    second = await upsert_partner_rankings(warehouse_engine, results)

    assert first == second == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(analytics_partner_rankings).where(
                        analytics_partner_rankings.c.hs6 == _TEST_HS6
                    )
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1
    assert rows[0]["rank"] == 1


async def test_upsert_partner_rankings_handles_a_rank_reshuffle(
    warehouse_engine: AsyncEngine,
) -> None:
    """Regression test for a real bug found live: re-running rankings
    after a newly-ingested partner outranks the previously-#1 partner
    used to raise a real UniqueViolationError on
    ix_apr_rank_where_present, since a plain ON CONFLICT (hs6, flow,
    year, partner_country_code) only suppresses conflicts on that one
    named constraint - not the separate partial unique index on rank -
    and the fresh batch transiently collided with the old row still
    sitting at its stale rank."""
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(_row(partner="792", status="OK", value=500))
        )
    first_pass = await compute_partner_rankings(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )
    await upsert_partner_rankings(warehouse_engine, first_pass)  # "792" is rank 1

    async with warehouse_engine.begin() as conn:
        # A new partner outranks "792" - the exact real-world shape of the
        # bug (a newly-ingested country reshuffling relative rank order).
        await conn.execute(
            insert(normalized_trade_flows).values(_row(partner="156", status="OK", value=1000))
        )
    second_pass = await compute_partner_rankings(
        warehouse_engine, hs6=_TEST_HS6, flow="import", year=2023
    )
    written = await upsert_partner_rankings(warehouse_engine, second_pass)  # must not raise

    assert written == 2
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(analytics_partner_rankings).where(
                        analytics_partner_rankings.c.hs6 == _TEST_HS6
                    )
                )
            )
            .mappings()
            .all()
        )
    by_partner = {r["partner_country_code"]: r for r in rows}
    assert by_partner["156"]["rank"] == 1
    assert by_partner["792"]["rank"] == 2
