"""Integration test for `app.report.facts.assemble_facts` against a real
Postgres — seeds every analytics/ref table it reads and asserts the
assembled JSON's real shape, including the flagged deviations from §14's
literal example (empty month_wise_current_year, None coverage/landed_cost
when no precomputed row exists for the exact window).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import delete, insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.report.facts import assemble_facts
from app.warehouse.schema import (
    analytics_coverage_summary,
    analytics_mismatch_checks,
    analytics_partner_rankings,
    analytics_unit_value_series,
    ref_hs6_hs8_crosswalk,
    ref_llm_datapoints,
    ref_regulatory_notes,
)

pytestmark = pytest.mark.integration

_TEST_HS6 = "999993"  # never a real HS6 - test-only, deleted after every test
_TEST_HS8 = _TEST_HS6 + "00"


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(analytics_partner_rankings).where(analytics_partner_rankings.c.hs6 == _TEST_HS6)
        )
        await conn.execute(
            delete(analytics_unit_value_series).where(
                analytics_unit_value_series.c.hs6 == _TEST_HS6
            )
        )
        await conn.execute(
            delete(analytics_mismatch_checks).where(analytics_mismatch_checks.c.hs6 == _TEST_HS6)
        )
        await conn.execute(
            delete(analytics_coverage_summary).where(analytics_coverage_summary.c.hs6 == _TEST_HS6)
        )
        await conn.execute(
            delete(ref_hs6_hs8_crosswalk).where(ref_hs6_hs8_crosswalk.c.hs6 == _TEST_HS6)
        )
        await conn.execute(
            delete(ref_regulatory_notes).where(ref_regulatory_notes.c.hs6 == _TEST_HS6)
        )
        await conn.execute(delete(ref_llm_datapoints).where(ref_llm_datapoints.c.hs6 == _TEST_HS6))


async def test_assemble_facts_real_shape(warehouse_engine: AsyncEngine) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(analytics_partner_rankings).values(
                [
                    {
                        "hs6": _TEST_HS6,
                        "flow": "import",
                        "year": 2023,
                        "partner_country_code": "792",
                        "rank": 1,
                        "value_inr_paise": 900,
                        "status": "OK",
                    },
                    {
                        "hs6": _TEST_HS6,
                        "flow": "import",
                        "year": 2023,
                        "partner_country_code": "4",
                        "rank": 2,
                        "value_inr_paise": 100,
                        "status": "OK",
                    },
                    {
                        "hs6": _TEST_HS6,
                        "flow": "import",
                        "year": 2023,
                        "partner_country_code": "156",
                        "rank": None,
                        "value_inr_paise": None,
                        "status": "NOT_REPORTED",
                    },
                ]
            )
        )
        await conn.execute(
            insert(analytics_unit_value_series).values(
                hs6=_TEST_HS6,
                flow="import",
                year=2023,
                unit_value_inr_paise_per_kg=Decimal("500"),
                delta_value_pct=Decimal("10"),
                delta_from_qty_pct=Decimal("5"),
                delta_from_price_pct=Decimal("3"),
                delta_from_fx_pct=Decimal("2"),
                coverage_gate_passed=True,
            )
        )
        await conn.execute(
            insert(analytics_mismatch_checks).values(
                hs6=_TEST_HS6,
                flow="import",
                year=2023,
                check_name="B_dgcis_vs_partner_comtrade",
                partner_country_code="792",
                gap_pct=Decimal("9.1"),
                severity="quiet",
                direction_flip_yoy=False,
            )
        )
        await conn.execute(
            insert(analytics_coverage_summary).values(
                hs6=_TEST_HS6,
                flow="import",
                window_start=date(2023, 1, 1),
                window_end=date(2023, 12, 31),
                expected_cells=3,
                present_cells=2,
                not_yet_published_cells=0,
                suppressed_cells=0,
                fetch_failed_cells=0,
                gate_passed=True,
                degraded=False,
            )
        )
        await conn.execute(
            insert(ref_hs6_hs8_crosswalk).values(
                hs6=_TEST_HS6, hs8=_TEST_HS8, first_seen_at=date(2023, 1, 1)
            )
        )

    facts = await assemble_facts(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        top_n=10,
        as_of=date(2023, 12, 31),
        include_agriculture_sources=False,
    )

    assert facts.hs6 == _TEST_HS6
    assert facts.product_label == _TEST_HS6  # not a real taxonomy code - falls back, never guessed
    assert facts.window.years == 1
    assert facts.window.start_year == facts.window.end_year == 2023

    year_2023 = facts.annual_series[0]
    assert year_2023.total_inr_paise == 1000  # 900 + 100, NOT_REPORTED contributes 0
    assert year_2023.status == "NOT_REPORTED"  # worst status across all 3 partners
    assert [p.country for p in year_2023.partners] == ["Türkiye", "Afghanistan"]
    assert year_2023.all_other_partners.value_inr_paise == 0
    assert year_2023.all_other_partners.status == "NOT_REPORTED"

    assert facts.unit_value_trend[0].inr_paise_per_kg == Decimal("500")
    assert facts.hhi_by_year[0].hhi == pytest.approx(Decimal("0.82"), abs=Decimal("0.001"))

    assert facts.mismatch_checks[0].partner == "Türkiye"
    assert facts.mismatch_checks[0].gap_pct == Decimal("9.1")

    assert facts.coverage is not None
    assert facts.coverage.expected_cells == 3

    assert (
        facts.month_wise_current_year == []
    )  # no monthly ingestion job yet - honest, not fabricated

    assert facts.hs8_split_note.startswith(_TEST_HS8)
    assert facts.landed_cost is not None
    assert facts.landed_cost.is_complete is False  # no real duty rate recorded for this test hs8
    assert facts.landed_cost_as_of_period == "2023"

    assert facts.regulatory_note is None
    # top-1 share = 900/1000 = 90% > 60%, no ref_regulatory_notes row -> warning fires
    assert facts.regulatory_note_missing_warning is True


async def test_assemble_facts_coverage_is_none_when_no_row_exists_for_the_window(
    warehouse_engine: AsyncEngine,
) -> None:
    facts = await assemble_facts(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        top_n=10,
        as_of=date(2023, 12, 31),
        include_agriculture_sources=False,
    )

    assert facts.coverage is None
    assert facts.landed_cost is None  # no ref_hs6_hs8_crosswalk row either
    assert facts.landed_cost_as_of_period is None
    assert facts.annual_series[0].status == "NOT_REPORTED"
    assert facts.annual_series[0].partners == []


async def test_assemble_facts_no_warning_when_a_regulatory_note_exists(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(analytics_partner_rankings).values(
                hs6=_TEST_HS6,
                flow="import",
                year=2023,
                partner_country_code="792",
                rank=1,
                value_inr_paise=900,
                status="OK",
            )
        )
        await conn.execute(
            insert(ref_regulatory_notes).values(
                hs6=_TEST_HS6, note="Test regulatory note.", updated_by="test"
            )
        )

    facts = await assemble_facts(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        top_n=10,
        as_of=date(2023, 12, 31),
        include_agriculture_sources=False,
    )

    assert facts.regulatory_note == "Test regulatory note."
    assert facts.regulatory_note_missing_warning is False  # a note exists, so no warning fires


# --- ref_llm_datapoints / llm_datapoints (2026-09-02, Step 4 hardening) -------


async def test_assemble_facts_reads_an_active_llm_datapoint_and_filters_by_field(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(ref_llm_datapoints).values(
                hs6=_TEST_HS6,
                field_name="mandi_price",
                effective_period="2026-08",
                value_json={"modal_price_inr_paise_per_qtl": 850000, "market": "Nashik"},
                source_authority="Test Authority",
                source_reference="Test market bulletin",
                source_url="https://example.test/bulletin",
                verified_date=date(2026, 9, 2),
            )
        )

    facts = await assemble_facts(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        top_n=10,
        as_of=date(2023, 12, 31),
        include_agriculture_sources=True,
    )

    assert len(facts.llm_datapoints) == 1
    entry = facts.llm_datapoints[0]
    assert entry.field_name == "mandi_price"
    assert entry.value == {"modal_price_inr_paise_per_qtl": 850000, "market": "Nashik"}
    assert entry.source_authority == "Test Authority"
    assert entry.source_url == "https://example.test/bulletin"

    # Filtered correctly into the matching field's slice, not the others.
    assert facts.mandi_price_llm_datapoints == [entry]
    assert facts.msp_llm_datapoints == []
    assert facts.international_production_llm_datapoints == []

    # Never blended into the verified field it sits beside - the honest
    # NOT_FOUND status (no real Agmarknet row for this fake test hs6)
    # is completely unaffected by the cited datapoint existing.
    assert facts.mandi_price.status == "NOT_FOUND"


async def test_assemble_facts_excludes_a_retracted_llm_datapoint(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(ref_llm_datapoints).values(
                [
                    {
                        "hs6": _TEST_HS6,
                        "field_name": "mandi_price",
                        "effective_period": "2026-08",
                        "value_json": {"modal_price_inr_paise_per_qtl": 850000},
                        "source_authority": "Test Authority",
                        "source_reference": "Still active",
                        "source_url": None,
                        "verified_date": date(2026, 9, 2),
                        "status": "ACTIVE",
                        # Explicit, matching the second row's key set - a
                        # multi-row `insert().values([...])` compiles its
                        # column list from the *first* dict only, silently
                        # dropping any key absent there even if a later row
                        # includes it (a real SQLAlchemy Core gotcha hit
                        # while writing this test, not a schema issue).
                        "retracted_reason": None,
                    },
                    {
                        "hs6": _TEST_HS6,
                        "field_name": "mandi_price",
                        "effective_period": "2026-07",
                        "value_json": {"modal_price_inr_paise_per_qtl": 1},
                        "source_authority": "Test Authority",
                        "source_reference": "Later found to be wrong",
                        "source_url": None,
                        "verified_date": date(2026, 8, 1),
                        "status": "RETRACTED",
                        "retracted_reason": "Source page was actually about a different commodity",
                    },
                ]
            )
        )

    facts = await assemble_facts(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        top_n=10,
        as_of=date(2023, 12, 31),
        include_agriculture_sources=True,
    )

    assert len(facts.llm_datapoints) == 1
    assert facts.llm_datapoints[0].source_reference == "Still active"


async def test_assemble_facts_llm_datapoints_empty_when_agriculture_sources_excluded(
    warehouse_engine: AsyncEngine,
) -> None:
    """Same gate as the verified mandi_price/msp/international_production
    fetches: a cited search result for a categorically-inapplicable field
    (e.g. cement's "mandi price") is exactly as meaningless as that field's
    own NOT_APPLICABLE status - never fetched, regardless of whether a row
    happens to exist."""
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(ref_llm_datapoints).values(
                hs6=_TEST_HS6,
                field_name="mandi_price",
                effective_period="2026-08",
                value_json={"modal_price_inr_paise_per_qtl": 850000},
                source_authority="Test Authority",
                source_reference="Test",
                source_url=None,
                verified_date=date(2026, 9, 2),
            )
        )

    facts = await assemble_facts(
        warehouse_engine,
        hs6=_TEST_HS6,
        flow="import",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
        top_n=10,
        as_of=date(2023, 12, 31),
        include_agriculture_sources=False,
    )

    assert facts.llm_datapoints == []
    assert facts.mandi_price_llm_datapoints == []
    assert facts.mandi_price.status == "NOT_APPLICABLE"


async def test_ref_llm_datapoints_rejects_a_row_for_a_field_not_in_the_allowlist(
    warehouse_engine: AsyncEngine,
) -> None:
    """The check constraint, not just application code, is what stops a
    typo'd or unsupported field_name from ever landing in this table."""
    with pytest.raises(IntegrityError, match="ck_rld_field_name"):
        async with warehouse_engine.begin() as conn:
            await conn.execute(
                insert(ref_llm_datapoints).values(
                    hs6=_TEST_HS6,
                    field_name="not_a_real_field",
                    effective_period="2026-08",
                    value_json={},
                    source_authority="Test Authority",
                    source_reference="Test",
                    source_url=None,
                    verified_date=date(2026, 9, 2),
                )
            )


async def test_ref_llm_datapoints_requires_a_reason_to_retract_a_row(
    warehouse_engine: AsyncEngine,
) -> None:
    """A row can't be silently marked RETRACTED with no explanation - the
    check constraint enforces that a retraction always carries its own
    reason, the same way ref_duty_components's status/value_pct pairing is
    enforced in the database, not just trusted from application code."""
    with pytest.raises(IntegrityError, match="ck_rld_retracted_reason_matches_status"):
        async with warehouse_engine.begin() as conn:
            await conn.execute(
                insert(ref_llm_datapoints).values(
                    hs6=_TEST_HS6,
                    field_name="mandi_price",
                    effective_period="2026-08",
                    value_json={},
                    source_authority="Test Authority",
                    source_reference="Test",
                    source_url=None,
                    verified_date=date(2026, 9, 2),
                    status="RETRACTED",
                    retracted_reason=None,
                )
            )
