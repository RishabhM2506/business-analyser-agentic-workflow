"""Unit tests for `app.report.landed_cost.compute_landed_cost` — the core
regression tests for the evidence-first duty rule: never a complete total
when any component isn't VERIFIED, never a silent 0%-as-missing."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.pipeline.duty_source import DutyComponentEvidence, DutyEvidence
from app.report.landed_cost import compute_landed_cost

pytestmark = pytest.mark.unit

_TODAY = date(2026, 8, 23)
_CIF_PAISE_PER_KG = 100_000_00  # 100,000.00 INR/kg in paise


def _verified(component: str, pct: str) -> DutyComponentEvidence:
    return DutyComponentEvidence(
        component=component,
        verification_status="VERIFIED",
        value_pct=Decimal(pct),
        source_authority="ICEGATE Trade Guide on Imports",
        source_reference="citation",
        verified_date=_TODAY,
    )


def _not_verified(component: str) -> DutyComponentEvidence:
    return DutyComponentEvidence(
        component=component,
        verification_status="NOT_VERIFIED",
        notes="Not verified from an authoritative official source.",
    )


def _all_verified_evidence() -> DutyEvidence:
    return DutyEvidence(
        hs8="12079100",
        as_of=_TODAY,
        components={
            "BCD": _verified("BCD", "20"),
            "AIDC": _verified("AIDC", "0"),
            "SWS": _verified("SWS", "10"),
            "IGST": _verified("IGST", "5"),
        },
    )


def test_all_verified_components_produce_a_complete_landed_cost() -> None:
    result = compute_landed_cost(_CIF_PAISE_PER_KG, _all_verified_evidence())

    assert result.is_complete is True
    assert result.excluded_components == []
    assert result.landed_cost_inr_paise_per_kg is not None
    # cif * (1 + 0.20 + 0 + 0.10) * (1 + 0.05) = cif * 1.30 * 1.05 = cif * 1.365
    expected = int(Decimal(_CIF_PAISE_PER_KG) * Decimal("1.365"))
    assert result.landed_cost_inr_paise_per_kg == expected
    assert result.partial_landed_cost_inr_paise_per_kg == expected


def test_a_single_not_verified_component_makes_the_result_incomplete() -> None:
    evidence = _all_verified_evidence()
    evidence.components["IGST"] = _not_verified("IGST")

    result = compute_landed_cost(_CIF_PAISE_PER_KG, evidence)

    assert result.is_complete is False
    assert result.landed_cost_inr_paise_per_kg is None  # never a complete total
    assert result.excluded_components == ["IGST"]
    # partial is still computed, from BCD+AIDC+SWS only (no IGST factor)
    expected_partial = int(Decimal(_CIF_PAISE_PER_KG) * Decimal("1.30"))
    assert result.partial_landed_cost_inr_paise_per_kg == expected_partial


def test_a_conflicting_component_also_makes_the_result_incomplete() -> None:
    evidence = _all_verified_evidence()
    evidence.components["BCD"] = DutyComponentEvidence(
        component="BCD",
        verification_status="CONFLICTING",
        conflicting_candidates=[
            {"value_pct": Decimal(15), "source_authority": "a", "source_reference": "ref-a"},
            {"value_pct": Decimal(25), "source_authority": "b", "source_reference": "ref-b"},
        ],
    )

    result = compute_landed_cost(_CIF_PAISE_PER_KG, evidence)

    assert result.is_complete is False
    assert "BCD" in result.excluded_components


def test_an_expired_component_also_makes_the_result_incomplete() -> None:
    evidence = _all_verified_evidence()
    evidence.components["SWS"] = DutyComponentEvidence(
        component="SWS", verification_status="EXPIRED", value_pct=Decimal("10.000")
    )

    result = compute_landed_cost(_CIF_PAISE_PER_KG, evidence)

    assert result.is_complete is False
    assert "SWS" in result.excluded_components


def test_every_component_evidence_is_always_returned_regardless_of_completeness() -> None:
    """The full evidence must always be available for the report to render
    each component's citation next to its number — even the excluded ones."""
    evidence = _all_verified_evidence()
    evidence.components["IGST"] = _not_verified("IGST")

    result = compute_landed_cost(_CIF_PAISE_PER_KG, evidence)

    assert set(result.components) == {"BCD", "AIDC", "SWS", "IGST"}
    assert result.components["IGST"].verification_status == "NOT_VERIFIED"
