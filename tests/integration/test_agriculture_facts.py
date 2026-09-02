"""Integration tests for `app.report.facts`'s three domain-specific
agriculture sections (`_fetch_mandi_price`, `_fetch_msp`,
`_fetch_international_production`) against a real Postgres, plus
`assemble_facts`'s own `include_agriculture_sources` wiring.

The two FAOSTAT poppy-seed tests below used to read ambient, already-
committed real data from a one-off `app.pipeline.faostat` curator run
made earlier in the same session this file was written in ("no inserts/
deletes here, purely reading what's already real and committed" - the
prior version of this docstring). That's real against a long-lived local
Postgres a curator script has actually been run against, but not
self-contained: CI spins up a fresh, migrations-only Postgres with no
such ambient data, and this file had never actually run in CI before
(confirmed: it doesn't exist on `main` at all yet). Fixed to seed its own
FAOSTAT fixture rows, matching every other test in this file's own
established pattern (`raw_agmarknet_prices`/`raw_msp_records` below) -
still tests the real matching logic against the real HS 120791 taxonomy
description and item name ("Poppy seed"), just no longer depends on
out-of-band ingestion having already happened against whatever database
the test happens to run on.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import delete, insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.report.facts import (
    _fetch_international_production,
    _fetch_mandi_price,
    _fetch_msp,
    assemble_facts,
)
from app.warehouse.schema import raw_agmarknet_prices, raw_faostat_records, raw_msp_records

pytestmark = pytest.mark.integration

_TEST_COMMODITY = "___test-widget-commodity___"  # never a real commodity - test-only
_TEST_FAOSTAT_ITEM = "Poppy seed"  # the real FAOSTAT item name for HS 120791 - deliberately real
# text (so the matching logic under test is genuine), paired with sentinel
# area/item codes and a year (9999) no real FAOSTAT vintage will ever reach -
# guarantees no collision with any real data a long-lived local database
# might also hold.
_TEST_FAOSTAT_AREA_CODE_WORLD = "TEST-WORLD"
_TEST_FAOSTAT_AREA_CODE_INDIA = "TEST-INDIA"
_TEST_FAOSTAT_ITEM_CODE = "TEST-POPPY"
_TEST_FAOSTAT_YEAR = 9999


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(raw_agmarknet_prices).where(raw_agmarknet_prices.c.commodity == _TEST_COMMODITY)
        )
        await conn.execute(
            delete(raw_msp_records).where(raw_msp_records.c.commodity == _TEST_COMMODITY)
        )
        await conn.execute(
            delete(raw_faostat_records).where(
                raw_faostat_records.c.item_code == _TEST_FAOSTAT_ITEM_CODE
            )
        )


async def _seed_faostat_poppy_seed_fixture(
    engine: AsyncEngine, *, world_value: str, india_value: str | None
) -> None:
    """A real-shaped FAOSTAT `Production` pair (World + India, same item/
    year) - `india_value=None` reproduces FAOSTAT's own real `M` (missing)
    flag shape for a genuinely absent figure, never a fabricated zero."""
    rows: list[dict[str, object]] = [
        {
            "fetched_at": datetime.now(UTC),
            "area_code": _TEST_FAOSTAT_AREA_CODE_WORLD,
            "area": "World",
            "item_code": _TEST_FAOSTAT_ITEM_CODE,
            "item": _TEST_FAOSTAT_ITEM,
            "element": "Production",
            "unit": "t",
            "year": _TEST_FAOSTAT_YEAR,
            "value": world_value,
            "flag": "A",
            "raw_payload": {},
        },
        {
            "fetched_at": datetime.now(UTC),
            "area_code": _TEST_FAOSTAT_AREA_CODE_INDIA,
            "area": "India",
            "item_code": _TEST_FAOSTAT_ITEM_CODE,
            "item": _TEST_FAOSTAT_ITEM,
            "element": "Production",
            "unit": "t",
            "year": _TEST_FAOSTAT_YEAR,
            "value": india_value,
            "flag": "M" if india_value is None else "A",
            "raw_payload": {},
        },
    ]
    async with engine.begin() as conn:
        await conn.execute(insert(raw_faostat_records).values(rows))


async def test_fetch_mandi_price_matches_a_real_row(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_agmarknet_prices).values(
                fetched_at=datetime.now(UTC),
                price_date=date(2024, 1, 15),
                commodity=_TEST_COMMODITY,
                market="Test Market",
                state="Test State",
                modal_price_inr_paise_per_qtl=250_000,
                raw_payload={},
            )
        )

    fact = await _fetch_mandi_price(warehouse_engine, taxonomy_description=_TEST_COMMODITY)

    assert fact.status == "OK"
    assert fact.matched_commodity == _TEST_COMMODITY
    assert fact.modal_price_inr_paise_per_qtl == 250_000
    assert fact.market == "Test Market"


async def test_fetch_mandi_price_not_found_when_nothing_matches(
    warehouse_engine: AsyncEngine,
) -> None:
    fact = await _fetch_mandi_price(
        warehouse_engine, taxonomy_description="___a commodity nothing will ever match___"
    )

    assert fact.status == "NOT_FOUND"
    assert fact.matched_commodity is None
    assert fact.modal_price_inr_paise_per_qtl is None


async def test_fetch_msp_matches_a_real_row(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(raw_msp_records).values(
                fetched_at=datetime.now(UTC),
                crops="Test Crops",
                commodity=_TEST_COMMODITY,
                year_label="2022-23",
                cost_inr_paise_per_qtl=100_000,
                msp_inr_paise_per_qtl=150_000,
                raw_payload={},
            )
        )

    fact = await _fetch_msp(warehouse_engine, taxonomy_description=_TEST_COMMODITY)

    assert fact.status == "OK"
    assert fact.matched_commodity == _TEST_COMMODITY
    assert fact.msp_inr_paise_per_qtl == 150_000
    assert fact.cost_inr_paise_per_qtl == 100_000


async def test_fetch_msp_not_found_when_nothing_matches(warehouse_engine: AsyncEngine) -> None:
    fact = await _fetch_msp(
        warehouse_engine, taxonomy_description="___a commodity nothing will ever match___"
    )

    assert fact.status == "NOT_FOUND"


async def test_fetch_international_production_finds_the_real_poppy_seed_data(
    warehouse_engine: AsyncEngine,
) -> None:
    """Self-contained regression for the real, live-confirmed FAOSTAT shape
    (`app.pipeline.faostat`'s own real run for "Poppy seed"): India reports
    a genuine `M` (missing) flag some years, never a fabricated zero, while
    the world total is a real, positive figure - reproduced here as a
    seeded fixture rather than depending on ambient, already-ingested data
    (see this module's own docstring for why)."""
    await _seed_faostat_poppy_seed_fixture(
        warehouse_engine, world_value="853910.500", india_value=None
    )

    fact = await _fetch_international_production(
        warehouse_engine,
        taxonomy_description="Oil seeds; poppy seeds, whether or not broken",
    )

    assert fact.status == "OK"
    assert fact.matched_item == "Poppy seed"
    assert fact.india_status == "NOT_FOUND"  # never a fabricated zero
    assert fact.india_production_tonnes is None
    assert fact.world_production_tonnes == Decimal("853910.500")


async def test_fetch_international_production_not_found_when_nothing_matches(
    warehouse_engine: AsyncEngine,
) -> None:
    fact = await _fetch_international_production(
        warehouse_engine, taxonomy_description="___a commodity nothing will ever match___"
    )

    assert fact.status == "NOT_FOUND"
    assert fact.matched_item is None


async def test_assemble_facts_marks_all_three_sections_not_applicable_when_excluded(
    warehouse_engine: AsyncEngine,
) -> None:
    facts = await assemble_facts(
        warehouse_engine,
        hs6="999994",  # never a real HS6 - test-only
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        top_n=10,
        as_of=date(2023, 12, 31),
        include_agriculture_sources=False,
    )

    assert facts.mandi_price.status == "NOT_APPLICABLE"
    assert facts.msp.status == "NOT_APPLICABLE"
    assert facts.international_production.status == "NOT_APPLICABLE"


async def test_assemble_facts_real_end_to_end_for_poppy_seeds(
    warehouse_engine: AsyncEngine,
) -> None:
    """The real product-intelligence scenario end to end: HS6 120791
    (poppy seeds), agriculture-relevant, a seeded FAOSTAT match (see this
    module's own docstring), and real Agmarknet/MSP gaps - this pipeline
    genuinely has no Agmarknet/MSP rows for poppy seeds at all, so those
    two stay NOT_FOUND with zero fixture data needed, exactly like
    production."""
    await _seed_faostat_poppy_seed_fixture(
        warehouse_engine, world_value="853910.500", india_value=None
    )

    facts = await assemble_facts(
        warehouse_engine,
        hs6="120791",
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        top_n=10,
        as_of=date(2023, 12, 31),
        include_agriculture_sources=True,
    )

    assert facts.product_label == "Oil seeds; poppy seeds, whether or not broken"
    assert facts.mandi_price.status == "NOT_FOUND"  # real, confirmed Agmarknet gap
    assert facts.msp.status == "NOT_FOUND"  # poppy seeds are not one of the 22 MSP crops
    assert facts.international_production.status == "OK"
    assert facts.international_production.india_status == "NOT_FOUND"
    assert facts.international_production.world_production_tonnes == Decimal("853910.500")
