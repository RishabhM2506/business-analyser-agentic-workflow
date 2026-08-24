"""Integration tests for `app.report.facts`'s three domain-specific
agriculture sections (`_fetch_mandi_price`, `_fetch_msp`,
`_fetch_international_production`) against a real Postgres, plus
`assemble_facts`'s own `include_agriculture_sources` wiring — including a
real end-to-end check against the actual FAOSTAT poppy-seed data already
loaded this session (`app.pipeline.faostat`'s live-verified real run).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import delete, insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.report.facts import (
    _fetch_international_production,
    _fetch_mandi_price,
    _fetch_msp,
    assemble_facts,
)
from app.warehouse.schema import raw_agmarknet_prices, raw_msp_records

pytestmark = pytest.mark.integration

_TEST_COMMODITY = "___test-widget-commodity___"  # never a real commodity - test-only


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
    """Real, end-to-end check against the actual FAOSTAT data loaded this
    session (`app.pipeline.faostat`'s live run for "Poppy seed") - no
    inserts/deletes here, purely reading what's already real and
    committed. India must read NOT_FOUND (FAOSTAT's own real `M` flag for
    every year 2015-2024, never a fabricated zero); the world total must
    be a real, positive figure."""
    fact = await _fetch_international_production(
        warehouse_engine,
        taxonomy_description="Oil seeds; poppy seeds, whether or not broken",
    )

    assert fact.status == "OK"
    assert fact.matched_item == "Poppy seed"
    assert fact.india_status == "NOT_FOUND"  # never a fabricated zero
    assert fact.india_production_tonnes is None
    assert fact.world_production_tonnes is not None
    assert fact.world_production_tonnes > 0


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
    (poppy seeds), agriculture-relevant, real FAOSTAT coverage but real
    Agmarknet/MSP gaps - every one of these is a genuine finding from
    real ingested data, not a fabricated placeholder."""
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
    assert facts.international_production.world_production_tonnes is not None
