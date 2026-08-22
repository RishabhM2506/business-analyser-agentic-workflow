"""Evidence-first customs duty data (`docs/PLAN.md` §4a, user-directed
2026-08-23 revision).

Primary sources only: ICEGATE's Trade Guide on Imports / Customs Duty
Calculator, and the CBIC Tax Information Portal. Both verified to be real
government sites but neither exposes a public API — both are human-facing
search/calculator tools, and a live fetch attempt against both hit TLS
certificate-verification failures from this environment. **No scraper for
v1**: duty verification is a deliberate manual-curation workflow (a human
looks up the current rate, records it with a citation via
`scripts/record_duty_rate.py`), not an ingestion job.

`DutySource` is the seam that makes this swappable later — a real
`CbicApiDutySource` could be added if CBIC/ICEGATE ever exposes an API,
without `app.report.landed_cost` ever needing to change.

Critical rule, enforced twice (in the database via `ref_duty_components`'s
check constraint, and in `DutyComponentEvidence`'s own validator below):
`value_pct` is present if and only if `verification_status='VERIFIED'`.
NULL is never interpreted as 0%.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql import or_

from app.warehouse.schema import ref_duty_component_conflicts, ref_duty_components

DutyComponent = Literal["BCD", "AIDC", "SWS", "IGST"]
DutyVerificationStatus = Literal["VERIFIED", "NOT_VERIFIED", "CONFLICTING", "EXPIRED"]

# Typed for iteration (schema.py's own DUTY_COMPONENTS is a plain
# tuple[str, ...], correct for building the DB CHECK constraint but not
# narrow enough for mypy to accept as a DutyComponent when iterated here).
DUTY_COMPONENT_VALUES: tuple[DutyComponent, ...] = ("BCD", "AIDC", "SWS", "IGST")

_NOT_VERIFIED_NOTE = "Not verified from an authoritative official source."


class ConflictCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value_pct: Decimal
    source_authority: str
    source_reference: str
    source_url: str | None = None


class DutyComponentEvidence(BaseModel):
    """One duty component's full evidence — always present for every one
    of `DUTY_COMPONENTS`, regardless of verification status (a missing
    component is `NOT_VERIFIED`, never simply absent from the result)."""

    model_config = ConfigDict(extra="forbid")

    component: DutyComponent
    verification_status: DutyVerificationStatus
    value_pct: Decimal | None = None
    source_authority: str | None = None
    source_reference: str | None = None
    source_url: str | None = None
    verified_date: date | None = None
    notes: str | None = None
    conflicting_candidates: list[ConflictCandidate] | None = None

    @model_validator(mode="after")
    def _value_matches_status(self) -> DutyComponentEvidence:
        # EXPIRED keeps its value_pct too — the user's spec: "preserve it
        # for historical analysis" requires the actual historical number,
        # not just the fact that one once existed. landed_cost.py still
        # excludes EXPIRED from any *current*/complete calculation
        # regardless (it checks verification_status == 'VERIFIED', not
        # "does a value exist") — this validator only guards NULL-as-0%.
        has_value = self.value_pct is not None
        should_have_value = self.verification_status in ("VERIFIED", "EXPIRED")
        if has_value != should_have_value:
            raise ValueError(
                f"{self.component}: value_pct is {'present' if has_value else 'absent'} but "
                f"verification_status is {self.verification_status!r} — value_pct must be set "
                f"if and only if verification_status is 'VERIFIED' or 'EXPIRED' (never NULL-as-0%, "
                f"never a value without a verified citation)."
            )
        if self.verification_status == "CONFLICTING" and not self.conflicting_candidates:
            raise ValueError(
                f"{self.component}: verification_status is 'CONFLICTING' but no "
                f"conflicting_candidates were supplied."
            )
        return self


class DutyEvidence(BaseModel):
    """Every one of `DUTY_COMPONENTS`, for one HS8 line as of one date."""

    model_config = ConfigDict(extra="forbid")

    hs8: str
    as_of: date
    components: dict[DutyComponent, DutyComponentEvidence]

    @model_validator(mode="after")
    def _every_component_present(self) -> DutyEvidence:
        missing = set(DUTY_COMPONENT_VALUES) - set(self.components)
        if missing:
            raise ValueError(
                f"DutyEvidence for {self.hs8} is missing components: {sorted(missing)}"
            )
        return self


class DutySource(Protocol):
    """Minimal interface for fetching one HS8 line's duty evidence — narrow
    so a future real-API adapter or a test double never needs to know
    about SQL, matching this repo's established `ModelClient`/`FxClient`/
    `ProductSearchProvider` Protocol pattern."""

    async def get_duty_evidence(self, hs8: str, *, as_of: date) -> DutyEvidence: ...


def _not_verified(component: DutyComponent) -> DutyComponentEvidence:
    return DutyComponentEvidence(
        component=component, verification_status="NOT_VERIFIED", notes=_NOT_VERIFIED_NOTE
    )


class ManualDutySource:
    """v1, only implementation: reads `ref_duty_components`/
    `ref_duty_component_conflicts` — rows written exclusively by a human
    curator via `scripts/record_duty_rate.py`, never inferred or guessed
    here. A component with no matching row for `as_of` is `NOT_VERIFIED`
    by construction (the absence of a row, not a stored value)."""

    def __init__(self, *, engine: AsyncEngine) -> None:
        self._engine = engine

    async def get_duty_evidence(self, hs8: str, *, as_of: date) -> DutyEvidence:
        components: dict[DutyComponent, DutyComponentEvidence] = {}
        async with self._engine.connect() as conn:
            for component in DUTY_COMPONENT_VALUES:
                row_stmt = select(ref_duty_components).where(
                    ref_duty_components.c.hs8 == hs8,
                    ref_duty_components.c.component == component,
                    ref_duty_components.c.effective_from <= as_of,
                    or_(
                        ref_duty_components.c.effective_to.is_(None),
                        ref_duty_components.c.effective_to > as_of,
                    ),
                )
                row = (await conn.execute(row_stmt)).mappings().one_or_none()
                if row is None:
                    components[component] = _not_verified(component)
                    continue

                conflicting_candidates = None
                if row["verification_status"] == "CONFLICTING":
                    conflicts_stmt = select(ref_duty_component_conflicts).where(
                        ref_duty_component_conflicts.c.hs8 == hs8,
                        ref_duty_component_conflicts.c.component == component,
                        ref_duty_component_conflicts.c.effective_from == row["effective_from"],
                    )
                    conflict_rows = (await conn.execute(conflicts_stmt)).mappings().all()
                    conflicting_candidates = [
                        ConflictCandidate(
                            value_pct=c["candidate_value_pct"],
                            source_authority=c["source_authority"],
                            source_reference=c["source_reference"],
                            source_url=c["source_url"],
                        )
                        for c in conflict_rows
                    ]

                components[component] = DutyComponentEvidence(
                    component=component,
                    verification_status=row["verification_status"],
                    value_pct=row["value_pct"],
                    source_authority=row["source_authority"],
                    source_reference=row["source_reference"],
                    source_url=row["source_url"],
                    verified_date=row["verified_date"],
                    notes=row["notes"],
                    conflicting_candidates=conflicting_candidates,
                )

        return DutyEvidence(hs8=hs8, as_of=as_of, components=components)
