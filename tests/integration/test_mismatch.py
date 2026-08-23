"""Integration tests for `app.report.mismatch` against a real Postgres —
seeds `normalized_trade_flows` rows directly (the layer this module reads,
per §10's single-join-point design) and asserts real check A/B computation
and upsert idempotency.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.normalize import (
    COMTRADE_DATASET_VERSION_PARTNER_ROLE,
    COMTRADE_DATASET_VERSION_REPORTER_ROLE,
    DGCIS_DATASET_VERSION,
)
from app.report.mismatch import (
    ALL_PARTNERS,
    CHECK_A,
    CHECK_B,
    WORLD_AGGREGATE_PARTNER_CODE,
    compute_check_a,
    compute_check_b,
    upsert_mismatch_checks,
)
from app.warehouse.schema import analytics_mismatch_checks, normalized_trade_flows

pytestmark = pytest.mark.integration

_TEST_HS6 = "999998"  # never a real HS6 - test-only, deleted after every test
_TEST_PARTNER = "999"  # never a real country code


def _row(
    *,
    source: str,
    dataset_version: str,
    year: int,
    partner_country_code: str,
    value_inr_paise: int | None,
    flow: str = "import",
) -> dict[str, object]:
    return {
        "source": source,
        "hs6": _TEST_HS6,
        "hs8": None,
        "hs_revision": "test",
        "flow": flow,
        "period_month": date(year, 1, 1),
        "calendar": "FY" if source == "dgcis" else "CY",
        "partner_country_code": partner_country_code,
        "basis": "CIF",
        "currency": "INR" if source == "dgcis" else "USD",
        "universe": "test",
        "dataset_version": dataset_version,
        "is_provisional": False,
        "status": "NOT_REPORTED" if value_inr_paise is None else "OK",
        "status_detail": None,
        "value_inr_paise": value_inr_paise,
        "value_original_currency_paise": value_inr_paise,
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
            delete(analytics_mismatch_checks).where(analytics_mismatch_checks.c.hs6 == _TEST_HS6)
        )


async def test_compute_check_a_compares_dgcis_total_to_comtrade_world_aggregate(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                [
                    _row(
                        source="dgcis",
                        dataset_version=DGCIS_DATASET_VERSION,
                        year=2023,
                        partner_country_code=_TEST_PARTNER,
                        value_inr_paise=100_000_000_000,
                    ),
                    _row(
                        source="comtrade",
                        dataset_version=COMTRADE_DATASET_VERSION_REPORTER_ROLE,
                        year=2023,
                        partner_country_code=WORLD_AGGREGATE_PARTNER_CODE,
                        value_inr_paise=105_000_000_000,
                    ),
                ]
            )
        )

    results, skipped = await compute_check_a(
        warehouse_engine, hs6=_TEST_HS6, flow="import", years=[2023]
    )

    assert skipped == []
    assert len(results) == 1
    assert results[0].check_name == CHECK_A
    assert results[0].partner_country_code == ALL_PARTNERS
    assert results[0].gap_pct == 5
    assert results[0].severity == "quiet"


async def test_compute_check_a_skips_a_year_with_no_comtrade_world_row(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                _row(
                    source="dgcis",
                    dataset_version=DGCIS_DATASET_VERSION,
                    year=2023,
                    partner_country_code=_TEST_PARTNER,
                    value_inr_paise=100_000_000_000,
                )
            )
        )

    results, skipped = await compute_check_a(
        warehouse_engine, hs6=_TEST_HS6, flow="import", years=[2023]
    )

    assert results == []
    assert len(skipped) == 1
    assert "NOT_REPORTED" in skipped[0].reason


async def test_compute_check_b_compares_dgcis_partner_to_that_partners_own_comtrade_export(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                [
                    _row(
                        source="dgcis",
                        dataset_version=DGCIS_DATASET_VERSION,
                        year=2023,
                        partner_country_code=_TEST_PARTNER,
                        value_inr_paise=100_000_000_000,
                    ),
                    _row(
                        source="comtrade",
                        dataset_version=COMTRADE_DATASET_VERSION_PARTNER_ROLE,
                        year=2023,
                        partner_country_code=_TEST_PARTNER,
                        value_inr_paise=108_000_000_000,
                        flow="export",
                    ),
                ]
            )
        )

    results, skipped = await compute_check_b(
        warehouse_engine, hs6=_TEST_HS6, flow="import", years=[2023]
    )

    assert skipped == []
    assert len(results) == 1
    assert results[0].check_name == CHECK_B
    assert results[0].partner_country_code == _TEST_PARTNER
    assert results[0].gap_pct == 8
    assert results[0].severity == "quiet"


async def test_compute_check_b_excludes_unmapped_partners(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                _row(
                    source="dgcis",
                    dataset_version=DGCIS_DATASET_VERSION,
                    year=2023,
                    partner_country_code="UNMAPPED",
                    value_inr_paise=50_000_000_000,
                )
            )
        )

    results, skipped = await compute_check_b(
        warehouse_engine, hs6=_TEST_HS6, flow="import", years=[2023]
    )

    assert results == []
    assert skipped == []  # not even attempted - excluded before evaluation, §10


async def test_upsert_mismatch_checks_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                [
                    _row(
                        source="dgcis",
                        dataset_version=DGCIS_DATASET_VERSION,
                        year=2023,
                        partner_country_code=_TEST_PARTNER,
                        value_inr_paise=100_000_000_000,
                    ),
                    _row(
                        source="comtrade",
                        dataset_version=COMTRADE_DATASET_VERSION_REPORTER_ROLE,
                        year=2023,
                        partner_country_code=WORLD_AGGREGATE_PARTNER_CODE,
                        value_inr_paise=150_000_000_000,
                    ),
                ]
            )
        )

    results, _ = await compute_check_a(warehouse_engine, hs6=_TEST_HS6, flow="import", years=[2023])
    first = await upsert_mismatch_checks(warehouse_engine, results)
    second = await upsert_mismatch_checks(warehouse_engine, results)

    assert first == second == 1
    async with warehouse_engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(analytics_mismatch_checks).where(
                        analytics_mismatch_checks.c.hs6 == _TEST_HS6
                    )
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1  # same real unique key, not a duplicate
    assert rows[0]["severity"] == "warning"  # 50% gap


async def test_upsert_mismatch_checks_with_no_results_writes_nothing(
    warehouse_engine: AsyncEngine,
) -> None:
    written = await upsert_mismatch_checks(warehouse_engine, [])

    assert written == 0
