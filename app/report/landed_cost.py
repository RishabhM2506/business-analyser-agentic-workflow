"""Landed cost from CIF + evidence-aware duty data (`docs/PLAN.md` §4a, §11).

`compute_landed_cost` consumes a `DutyEvidence` object
(`app.pipeline.duty_source`), never raw percentages — this is the whole
point of the evidence model: the calculation layer cannot "forget" to
check verification status, because it has no other way to get a number in
the first place.

Formula: `landed_cost = cif x (1 + bcd% + aidc% + sws%) x (1 + igst%)` —
duty on a duty-inclusive base, IGST applied last (`docs/PLAN.md` §11,
flagged there for confirmation against a real CBIC worked example).

**Never presents a complete total when any component isn't `VERIFIED`.**
An unverified/conflicting/expired component is excluded from the
calculation entirely (mathematically equivalent, for this multiplicative
formula, to a 0% contribution from that stage — there is no other
sensible way to "exclude" a compounding percentage) — the honesty
guarantee is in the labeling, not the arithmetic: the result is always
`is_complete=False` with the real total in `partial_landed_cost_inr_paise_per_kg`,
never silently presented as `landed_cost_inr_paise_per_kg`.

Money is `Decimal` throughout, rounded to integer paise exactly once at
the end (D8: "rounding happens once, at render time, never mid-calculation").
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, ConfigDict

from app.pipeline.duty_source import (
    DUTY_COMPONENT_VALUES,
    DutyComponent,
    DutyComponentEvidence,
    DutyEvidence,
)

_HUNDRED = Decimal(100)
_ONE = Decimal(1)


class LandedCostResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_complete: bool
    landed_cost_inr_paise_per_kg: int | None
    partial_landed_cost_inr_paise_per_kg: int
    excluded_components: list[DutyComponent]
    components: dict[DutyComponent, DutyComponentEvidence]


def _fraction(evidence: DutyEvidence, component: DutyComponent) -> Decimal:
    component_evidence = evidence.components[component]
    if component_evidence.verification_status != "VERIFIED" or component_evidence.value_pct is None:
        return Decimal(0)
    return component_evidence.value_pct / _HUNDRED


def compute_landed_cost(cif_inr_paise_per_kg: int, evidence: DutyEvidence) -> LandedCostResult:
    excluded = sorted(
        component
        for component in DUTY_COMPONENT_VALUES
        if evidence.components[component].verification_status != "VERIFIED"
    )

    cif = Decimal(cif_inr_paise_per_kg)
    pre_igst = cif * (
        _ONE + _fraction(evidence, "BCD") + _fraction(evidence, "AIDC") + _fraction(evidence, "SWS")
    )
    total = pre_igst * (_ONE + _fraction(evidence, "IGST"))
    total_paise = int(total.to_integral_value(rounding=ROUND_HALF_UP))

    is_complete = not excluded
    return LandedCostResult(
        is_complete=is_complete,
        landed_cost_inr_paise_per_kg=total_paise if is_complete else None,
        partial_landed_cost_inr_paise_per_kg=total_paise,
        excluded_components=excluded,
        components=evidence.components,
    )
