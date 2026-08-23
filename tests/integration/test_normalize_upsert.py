"""Integration tests for `app.pipeline.normalize` against a real Postgres —
raw_dgcis_annual/raw_comtrade_records rows in, normalized_trade_flows rows
out, idempotency proven by running twice (this project's Testing
standard).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.normalize import (
    CountryCrosswalk,
    normalize_comtrade_rows,
    normalize_dgcis_annual_rows,
)
from app.warehouse.schema import (
    normalized_trade_flows,
    raw_comtrade_records,
    raw_dgcis_annual,
    ref_country_crosswalk,
)

pytestmark = pytest.mark.integration

_TEST_HS6 = "999999"  # never a real HS6 - test-only, deleted after every test
_TEST_HS8 = "99999903"


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(delete(raw_dgcis_annual).where(raw_dgcis_annual.c.hs8 == _TEST_HS8))
        await conn.execute(
            delete(raw_comtrade_records).where(raw_comtrade_records.c.cmd_code == _TEST_HS6)
        )
        await conn.execute(
            delete(normalized_trade_flows).where(normalized_trade_flows.c.hs6 == _TEST_HS6)
        )
        await conn.execute(
            delete(ref_country_crosswalk).where(
                ref_country_crosswalk.c.dgcis_country_name == "TESTLAND"
            )
        )


async def test_normalize_dgcis_annual_rows_resolves_crosswalk_and_status(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(ref_country_crosswalk).values(dgcis_country_name="TESTLAND", country_code="792")
        )
        await conn.execute(
            insert(raw_dgcis_annual).values(
                [
                    {
                        "scraped_at": datetime.now(UTC),
                        "fiscal_year_label": "2023 - 2024",
                        "hs8": _TEST_HS8,
                        "flow": "import",
                        "partner_country": "TESTLAND",
                        "description": "TEST COMMODITY",
                        "unit": "KGS",
                        "value_inr_paise": 424_660_000_000,
                        "raw_payload": {},
                    },
                    {
                        "scraped_at": datetime.now(UTC),
                        "fiscal_year_label": "2024 - 2025",
                        "hs8": _TEST_HS8,
                        "flow": "import",
                        "partner_country": "TESTLAND",
                        "description": "TEST COMMODITY",
                        "unit": "KGS",
                        "value_inr_paise": 0,
                        "raw_payload": {},
                    },
                    {
                        "scraped_at": datetime.now(UTC),
                        "fiscal_year_label": "2025 - 2026",
                        "hs8": _TEST_HS8,
                        "flow": "import",
                        "partner_country": "UNKNOWNLAND",
                        "description": "TEST COMMODITY",
                        "unit": "KGS",
                        "value_inr_paise": None,
                        "raw_payload": {},
                    },
                ]
            )
        )

    crosswalk = await _load_test_crosswalk(warehouse_engine)
    written = await normalize_dgcis_annual_rows(
        warehouse_engine, hs6=_TEST_HS6, crosswalk=crosswalk
    )

    assert written == 3
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

    by_year = {r["period_month"].year: r for r in rows}
    # QTY_MISSING, not OK - raw_dgcis_annual has no quantity column at
    # all (this report never returns one), so every real nonzero-value
    # DGCIS row is QTY_MISSING per §5's status table.
    assert by_year[2023]["status"] == "QTY_MISSING"
    assert by_year[2023]["value_inr_paise"] == 424_660_000_000
    assert by_year[2023]["partner_country_code"] == "792"
    assert by_year[2023]["calendar"] == "FY"
    assert by_year[2023]["basis"] == "CIF"
    assert by_year[2023]["currency"] == "INR"
    assert by_year[2023]["fx_rate_used"] is None  # never round-tripped through USD, D8

    assert by_year[2024]["status"] == "ZERO"
    assert by_year[2024]["value_inr_paise"] == 0

    assert by_year[2025]["status"] == "NOT_REPORTED"
    assert by_year[2025]["value_inr_paise"] is None
    # unmapped name embedded in the code, never dropped and never
    # collapsed onto a shared bare 'UNMAPPED' constant (a real bug found
    # live: two distinct unmapped countries in the same batch collided on
    # normalized_trade_flows' unique key under a flat sentinel).
    assert by_year[2025]["partner_country_code"] == "UNMAPPED:UNKNOWNLAND"


async def test_normalize_dgcis_annual_rows_does_not_collide_on_two_distinct_unmapped_countries(
    warehouse_engine: AsyncEngine,
) -> None:
    """Regression test for a real CardinalityViolationError found live
    during the first ~250-country DGCIS run: many distinct countries were
    genuinely unmapped at once, and a bare shared 'UNMAPPED' sentinel made
    every one of them propose the same normalized_trade_flows row within
    a single bulk upsert statement."""
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_dgcis_annual).values(
                [
                    {
                        "scraped_at": datetime.now(UTC),
                        "fiscal_year_label": "2023 - 2024",
                        "hs8": _TEST_HS8,
                        "flow": "import",
                        "partner_country": "RURITANIA",
                        "description": "TEST COMMODITY",
                        "unit": "KGS",
                        "value_inr_paise": 100,
                        "raw_payload": {},
                    },
                    {
                        "scraped_at": datetime.now(UTC),
                        "fiscal_year_label": "2023 - 2024",
                        "hs8": _TEST_HS8,
                        "flow": "import",
                        "partner_country": "FREEDONIA",
                        "description": "TEST COMMODITY",
                        "unit": "KGS",
                        "value_inr_paise": 200,
                        "raw_payload": {},
                    },
                ]
            )
        )

    crosswalk = await _load_test_crosswalk(warehouse_engine)
    written = await normalize_dgcis_annual_rows(
        warehouse_engine, hs6=_TEST_HS6, crosswalk=crosswalk
    )  # must not raise CardinalityViolationError

    assert written == 2
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
    codes = {r["partner_country_code"] for r in rows}
    assert codes == {"UNMAPPED:RURITANIA", "UNMAPPED:FREEDONIA"}


async def test_normalize_dgcis_annual_rows_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(ref_country_crosswalk).values(dgcis_country_name="TESTLAND", country_code="792")
        )
        await conn.execute(
            insert(raw_dgcis_annual).values(
                scraped_at=datetime.now(UTC),
                fiscal_year_label="2023 - 2024",
                hs8=_TEST_HS8,
                flow="import",
                partner_country="TESTLAND",
                description="TEST COMMODITY",
                unit="KGS",
                value_inr_paise=100_000_000_000,
                raw_payload={},
            )
        )

    crosswalk = await _load_test_crosswalk(warehouse_engine)
    first = await normalize_dgcis_annual_rows(warehouse_engine, hs6=_TEST_HS6, crosswalk=crosswalk)
    second = await normalize_dgcis_annual_rows(warehouse_engine, hs6=_TEST_HS6, crosswalk=crosswalk)

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


async def test_normalize_comtrade_rows_converts_fx_and_skips_unresolved_years(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_comtrade_records).values(
                [
                    {
                        "fetched_at": datetime.now(UTC),
                        "period": 2023,
                        "reporter_code": "699",
                        "partner_code": "792",
                        "flow_code": "M",
                        "cmd_code": _TEST_HS6,
                        "primary_value_usd": Decimal("1000.00"),
                        "net_weight_kg": Decimal("50.000"),
                        "is_reported": True,
                        "raw_payload": {},
                    },
                    {
                        "fetched_at": datetime.now(UTC),
                        "period": 2022,
                        "reporter_code": "699",
                        "partner_code": "792",
                        "flow_code": "M",
                        "cmd_code": _TEST_HS6,
                        "primary_value_usd": Decimal("500.00"),
                        "net_weight_kg": Decimal("25.000"),
                        "is_reported": True,
                        "raw_payload": {},
                    },
                ]
            )
        )

    written = await normalize_comtrade_rows(
        warehouse_engine,
        hs6=_TEST_HS6,
        fx_rates={2023: (Decimal("83.000000"), date(2023, 1, 1))},  # 2022 deliberately absent
    )

    assert written == 1  # 2022 skipped - no fx rate resolved, never guessed at 1:1
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
    assert row["period_month"] == date(2023, 1, 1)
    assert row["calendar"] == "CY"
    assert row["flow"] == "import"
    assert row["basis"] == "CIF"
    assert row["currency"] == "USD"
    assert row["fx_rate_used"] == Decimal("83.000000")
    assert row["value_original_currency_paise"] == 100_000
    assert row["value_inr_paise"] == 8_300_000
    assert row["status"] == "OK"


async def test_normalize_comtrade_rows_is_idempotent(warehouse_engine: AsyncEngine) -> None:
    """Regression test for a real bug found live: normalized_trade_flows'
    unique constraint didn't have NULLS NOT DISTINCT, and every Comtrade
    row has hs8=NULL (Comtrade is HS6-level only) - standard SQL treats
    NULL as distinct from NULL even in a unique constraint, so ON
    CONFLICT never matched two otherwise-identical Comtrade rows and every
    re-run silently duplicated the whole slice (confirmed live: 363 real
    rows had grown to 726 after one extra run, undetected until this test
    was written). Fixed with a migration adding
    postgresql_nulls_not_distinct=True to the constraint."""
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_comtrade_records).values(
                fetched_at=datetime.now(UTC),
                period=2023,
                reporter_code="699",
                partner_code="792",
                flow_code="M",
                cmd_code=_TEST_HS6,
                primary_value_usd=Decimal("1000.00"),
                net_weight_kg=Decimal("50.000"),
                is_reported=True,
                raw_payload={},
            )
        )

    fx_rates = {2023: (Decimal("83.000000"), date(2023, 1, 1))}
    first = await normalize_comtrade_rows(warehouse_engine, hs6=_TEST_HS6, fx_rates=fx_rates)
    second = await normalize_comtrade_rows(warehouse_engine, hs6=_TEST_HS6, fx_rates=fx_rates)

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
    assert len(rows) == 1  # not 2 - same real unique key (NULL hs8 included), not a duplicate


async def test_normalize_comtrade_rows_sets_qty_missing_when_value_present_but_no_weight(
    warehouse_engine: AsyncEngine,
) -> None:
    """§5: 'value present, quantity null' -> QTY_MISSING, not OK - real
    Comtrade rows can have a value with no net_weight_kg."""
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_comtrade_records).values(
                fetched_at=datetime.now(UTC),
                period=2023,
                reporter_code="699",
                partner_code="792",
                flow_code="M",
                cmd_code=_TEST_HS6,
                primary_value_usd=Decimal("1000.00"),
                net_weight_kg=None,
                is_reported=True,
                raw_payload={},
            )
        )

    await normalize_comtrade_rows(
        warehouse_engine,
        hs6=_TEST_HS6,
        fx_rates={2023: (Decimal("83.000000"), date(2023, 1, 1))},
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
    assert len(rows) == 1
    assert rows[0]["status"] == "QTY_MISSING"


async def test_normalize_comtrade_rows_resolves_partner_country_from_reporter_code(
    warehouse_engine: AsyncEngine,
) -> None:
    """Regression test for a real bug found before mismatch.py could be
    built: a partner-role query (build_query_params(role="partner")) fixes
    partner_code=699 (India) and varies reporter_code instead - naively
    using raw partner_code as partner_country_code would store India as
    its own trade partner and silently break check_B (which specifically
    needs the foreign reporter's own export figure)."""
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_comtrade_records).values(
                fetched_at=datetime.now(UTC),
                period=2023,
                reporter_code="792",  # Turkey reporting its own export to India
                partner_code="699",  # India, fixed by the partner-role query shape
                flow_code="X",
                cmd_code=_TEST_HS6,
                primary_value_usd=Decimal("2000.00"),
                net_weight_kg=Decimal("80.000"),
                is_reported=True,
                raw_payload={},
            )
        )

    written = await normalize_comtrade_rows(
        warehouse_engine,
        hs6=_TEST_HS6,
        fx_rates={2023: (Decimal("83.000000"), date(2023, 1, 1))},
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
    assert rows[0]["partner_country_code"] == "792"  # not '699' - India is never its own partner
    assert rows[0]["flow"] == "export"
    assert rows[0]["dataset_version"] == "comtrade-mirror-partner-v1"


async def test_normalize_comtrade_rows_keeps_reporter_and_partner_role_rows_separate(
    warehouse_engine: AsyncEngine,
) -> None:
    """Regression test for a second real bug found in the same
    investigation: a reporter-role row (India self-reporting its own
    export to Turkey: reporter=699/partner=792/flow=X) and a partner-role
    row (Turkey self-reporting its own export to India, the mirror of
    India's *import*: reporter=792/partner=699/flow=X) both resolve to
    partner_country_code=792/flow='export' - without a dataset_version
    discriminator they'd collide on normalized_trade_flows' unique key and
    one would silently overwrite the other via ON CONFLICT DO UPDATE."""
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_comtrade_records).values(
                [
                    {
                        "fetched_at": datetime.now(UTC),
                        "period": 2023,
                        "reporter_code": "699",  # India self-reporting its own export
                        "partner_code": "792",
                        "flow_code": "X",
                        "cmd_code": _TEST_HS6,
                        "primary_value_usd": Decimal("1000.00"),
                        "net_weight_kg": Decimal("50.000"),
                        "is_reported": True,
                        "raw_payload": {},
                    },
                    {
                        "fetched_at": datetime.now(UTC),
                        "period": 2023,
                        "reporter_code": "792",  # Turkey self-reporting its own export
                        "partner_code": "699",
                        "flow_code": "X",
                        "cmd_code": _TEST_HS6,
                        "primary_value_usd": Decimal("2000.00"),
                        "net_weight_kg": Decimal("80.000"),
                        "is_reported": True,
                        "raw_payload": {},
                    },
                ]
            )
        )

    written = await normalize_comtrade_rows(
        warehouse_engine,
        hs6=_TEST_HS6,
        fx_rates={2023: (Decimal("83.000000"), date(2023, 1, 1))},
    )

    assert written == 2  # not 1 - neither silently overwrote the other
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

    assert len(rows) == 2
    by_dataset_version = {r["dataset_version"]: r for r in rows}
    assert (
        by_dataset_version["comtrade-mirror-reporter-v1"]["value_original_currency_paise"]
        == 100_000
    )
    assert (
        by_dataset_version["comtrade-mirror-partner-v1"]["value_original_currency_paise"] == 200_000
    )


async def _load_test_crosswalk(warehouse_engine: AsyncEngine) -> CountryCrosswalk:
    async with warehouse_engine.connect() as conn:
        rows = (await conn.execute(select(ref_country_crosswalk))).mappings().all()
    return CountryCrosswalk(
        by_dgcis_name={r["dgcis_country_name"]: r["country_code"] for r in rows}
    )
