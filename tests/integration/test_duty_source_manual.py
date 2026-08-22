"""Integration tests for `app.pipeline.duty_source.ManualDutySource` against
a real Postgres — proves the evidence-first contract holds at the actual
database layer, not just in Pydantic validators. Uses `warehouse_engine`
(`tests/integration/conftest.py`); skips if no real Postgres is configured.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import delete, insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.duty_source import ManualDutySource
from app.warehouse.schema import ref_duty_component_conflicts, ref_duty_components

pytestmark = pytest.mark.integration

_TEST_HS8 = "99999901"  # never a real HS8 line - test-only, deleted after every test


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(ref_duty_component_conflicts).where(
                ref_duty_component_conflicts.c.hs8 == _TEST_HS8
            )
        )
        await conn.execute(
            delete(ref_duty_components).where(ref_duty_components.c.hs8 == _TEST_HS8)
        )


async def test_missing_component_is_not_verified_by_construction(
    warehouse_engine: AsyncEngine,
) -> None:
    """No row at all for a component -> NOT_VERIFIED, never a guess."""
    source = ManualDutySource(engine=warehouse_engine)

    evidence = await source.get_duty_evidence(_TEST_HS8, as_of=date(2026, 8, 23))

    for component in ("BCD", "AIDC", "SWS", "IGST"):
        component_evidence = evidence.components[component]
        assert component_evidence.verification_status == "NOT_VERIFIED"
        assert component_evidence.value_pct is None
        assert component_evidence.notes == "Not verified from an authoritative official source."


async def test_verified_row_is_returned_with_its_full_citation(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(ref_duty_components).values(
                hs8=_TEST_HS8,
                component="BCD",
                effective_from=date(2026, 1, 1),
                effective_to=None,
                verification_status="VERIFIED",
                value_pct=Decimal("20.000"),
                source_authority="ICEGATE Trade Guide on Imports",
                source_reference="test-citation-001",
                source_url="https://www.icegate.gov.in/example",
                verified_date=date(2026, 8, 23),
                notes=None,
            )
        )

    source = ManualDutySource(engine=warehouse_engine)
    evidence = await source.get_duty_evidence(_TEST_HS8, as_of=date(2026, 8, 23))

    bcd = evidence.components["BCD"]
    assert bcd.verification_status == "VERIFIED"
    assert bcd.value_pct == Decimal("20.000")
    assert bcd.source_authority == "ICEGATE Trade Guide on Imports"
    assert bcd.source_reference == "test-citation-001"
    assert bcd.verified_date == date(2026, 8, 23)


async def test_conflicting_component_returns_every_candidate_without_picking_one(
    warehouse_engine: AsyncEngine,
) -> None:
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(ref_duty_components).values(
                hs8=_TEST_HS8,
                component="IGST",
                effective_from=date(2026, 1, 1),
                effective_to=None,
                verification_status="CONFLICTING",
                value_pct=None,
                source_authority="CBIC Tax Information Portal",
                source_reference="see conflicting_candidates",
                verified_date=date(2026, 8, 23),
                notes="Two official sources disagree.",
            )
        )
        await conn.execute(
            insert(ref_duty_component_conflicts).values(
                [
                    {
                        "hs8": _TEST_HS8,
                        "component": "IGST",
                        "effective_from": date(2026, 1, 1),
                        "candidate_value_pct": Decimal("5.000"),
                        "source_authority": "CBIC Tax Information Portal",
                        "source_reference": "citation-A",
                    },
                    {
                        "hs8": _TEST_HS8,
                        "component": "IGST",
                        "effective_from": date(2026, 1, 1),
                        "candidate_value_pct": Decimal("12.000"),
                        "source_authority": "CBIC Tax Information Portal",
                        "source_reference": "citation-B",
                    },
                ]
            )
        )

    source = ManualDutySource(engine=warehouse_engine)
    evidence = await source.get_duty_evidence(_TEST_HS8, as_of=date(2026, 8, 23))

    igst = evidence.components["IGST"]
    assert igst.verification_status == "CONFLICTING"
    assert igst.value_pct is None  # never auto-picked
    assert igst.conflicting_candidates is not None
    candidate_values = {c.value_pct for c in igst.conflicting_candidates}
    assert candidate_values == {Decimal("5.000"), Decimal("12.000")}


async def test_expired_row_is_only_visible_for_its_own_historical_window(
    warehouse_engine: AsyncEngine,
) -> None:
    """An EXPIRED row (superseded by a later VERIFIED one) must never be
    returned as the *current* rate, but must still be findable for a
    historical `as_of` date within its own validity window."""
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(ref_duty_components).values(
                [
                    {
                        "hs8": _TEST_HS8,
                        "component": "AIDC",
                        "effective_from": date(2025, 1, 1),
                        "effective_to": date(2026, 1, 1),
                        "verification_status": "EXPIRED",
                        "value_pct": Decimal("5.000"),
                        "source_authority": "ICEGATE Trade Guide on Imports",
                        "source_reference": "old-citation",
                        "verified_date": date(2025, 6, 1),
                        "notes": None,
                    },
                    {
                        "hs8": _TEST_HS8,
                        "component": "AIDC",
                        "effective_from": date(2026, 1, 1),
                        "effective_to": None,
                        "verification_status": "VERIFIED",
                        "value_pct": Decimal("0.000"),
                        "source_authority": "ICEGATE Trade Guide on Imports",
                        "source_reference": "current-citation",
                        "verified_date": date(2026, 8, 23),
                        "notes": None,
                    },
                ]
            )
        )

    source = ManualDutySource(engine=warehouse_engine)

    current = await source.get_duty_evidence(_TEST_HS8, as_of=date(2026, 8, 23))
    assert current.components["AIDC"].verification_status == "VERIFIED"
    assert current.components["AIDC"].value_pct == Decimal("0.000")

    historical = await source.get_duty_evidence(_TEST_HS8, as_of=date(2025, 6, 15))
    assert historical.components["AIDC"].verification_status == "EXPIRED"
    assert historical.components["AIDC"].value_pct == Decimal("5.000")
