"""Unit tests for `app.pipeline.duty_source`'s Pydantic evidence models —
the "NULL must never be interpreted as 0%" rule enforced at the model
layer, independent of the database's own check constraint."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.pipeline.duty_source import ConflictCandidate, DutyComponentEvidence, DutyEvidence

pytestmark = pytest.mark.unit

_TODAY = date(2026, 8, 23)


def test_verified_requires_a_value() -> None:
    with pytest.raises(ValidationError):
        DutyComponentEvidence(component="BCD", verification_status="VERIFIED", value_pct=None)


def test_not_verified_forbids_a_value() -> None:
    with pytest.raises(ValidationError):
        DutyComponentEvidence(
            component="BCD", verification_status="NOT_VERIFIED", value_pct=Decimal(20)
        )


def test_verified_with_a_value_is_valid() -> None:
    evidence = DutyComponentEvidence(
        component="BCD",
        verification_status="VERIFIED",
        value_pct=Decimal("20.000"),
        source_authority="ICEGATE Trade Guide on Imports",
        source_reference="citation",
        verified_date=_TODAY,
    )
    assert evidence.value_pct == Decimal("20.000")


def test_conflicting_requires_candidates() -> None:
    with pytest.raises(ValidationError):
        DutyComponentEvidence(
            component="IGST", verification_status="CONFLICTING", conflicting_candidates=None
        )


def test_conflicting_with_candidates_is_valid() -> None:
    evidence = DutyComponentEvidence(
        component="IGST",
        verification_status="CONFLICTING",
        conflicting_candidates=[
            ConflictCandidate(value_pct=Decimal(5), source_authority="a", source_reference="ref-a"),
            ConflictCandidate(
                value_pct=Decimal(12), source_authority="b", source_reference="ref-b"
            ),
        ],
    )
    assert evidence.value_pct is None  # never auto-picked, even though candidates exist


def _verified(component: str) -> DutyComponentEvidence:
    return DutyComponentEvidence(
        component=component,
        verification_status="VERIFIED",
        value_pct=Decimal("1.000"),
        source_authority="a",
        source_reference="b",
        verified_date=_TODAY,
    )


def test_duty_evidence_requires_all_four_components() -> None:
    with pytest.raises(ValidationError):
        DutyEvidence(
            hs8="12079100",
            as_of=_TODAY,
            components={"BCD": _verified("BCD"), "AIDC": _verified("AIDC")},  # missing SWS, IGST
        )


def test_duty_evidence_with_all_four_components_is_valid() -> None:
    evidence = DutyEvidence(
        hs8="12079100",
        as_of=_TODAY,
        components={c: _verified(c) for c in ("BCD", "AIDC", "SWS", "IGST")},
    )
    assert set(evidence.components) == {"BCD", "AIDC", "SWS", "IGST"}
