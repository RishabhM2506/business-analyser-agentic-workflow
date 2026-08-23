"""D9 mismatch checks (`docs/PLAN.md` §10) — check A (DGCIS's own national
total vs. India's own Comtrade submission) and check B (DGCIS's per-partner
figure vs. that partner's own Comtrade submission — real "mirror trade
statistics" reconciliation). Check C (vs. BACI) is not implemented yet;
BACI ingestion (build sequence item 6) hasn't been built.

Owned by this module alone, run as part of the same ingestion pass that
writes `analytics_partner_rankings` (§10: "always precomputed, never
request-time"). The join is always `normalized_trade_flows`, filtered to
`hs6, flow, period_month within year` — the single join point §10's PM-1
fix required, not re-implemented per check.

Both checks compare two independently-sourced totals for what should be
the same real shipment(s) — this is the whole point: neither side is
ever treated as "ground truth" to silently prefer, and a check is never
computed by guessing a missing side as zero (that would fabricate
agreement or disagreement that was never actually observed, exactly what
the user's evidence-first directive rules out). A side that is `None`
(one/both totals `NOT_REPORTED`, or a `dgcis_total`/`dgcis_import`
denominator of exactly zero) makes the check un-computable for that
`(hs6, flow, year[, partner])` — no row is written, and the caller is told
why via `SkippedCheck.reason`.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.normalize import (
    COMTRADE_DATASET_VERSION_PARTNER_ROLE,
    COMTRADE_DATASET_VERSION_REPORTER_ROLE,
    DGCIS_DATASET_VERSION,
    UNMAPPED_PREFIX,
)
from app.warehouse.schema import analytics_mismatch_checks, normalized_trade_flows

ALL_PARTNERS = "ALL_PARTNERS"
WORLD_AGGREGATE_PARTNER_CODE = "0"  # Comtrade's own "all partners" code, §8

CHECK_A = "A_dgcis_vs_comtrade_india"
CHECK_B = "B_dgcis_vs_partner_comtrade"

# §10: gap < 15% quiet, 15% <= gap < 40% flag, gap >= 40% warning — bands
# are `<`/`>=`, not off-by-one (boundary-tested at 14.9/15.1/39.9/40.1%).
_QUIET_CEILING_PCT = Decimal("15")
_FLAG_CEILING_PCT = Decimal("40")

_HUNDRED = Decimal("100")


@dataclass(frozen=True)
class MismatchResult:
    check_name: str
    hs6: str
    flow: str
    year: int
    partner_country_code: str
    gap_pct: Decimal
    severity: str
    direction_flip_yoy: bool


@dataclass(frozen=True)
class SkippedCheck:
    check_name: str
    partner_country_code: str
    year: int
    reason: str


def _severity(gap_pct: Decimal, *, direction_flip: bool) -> str:
    """`direction_flip` overrides the magnitude bands entirely (§10: "sign
    flips year-on-year -> ... untrustworthy (independent of gap size)")."""
    if direction_flip:
        return "untrustworthy"
    if gap_pct < _QUIET_CEILING_PCT:
        return "quiet"
    if gap_pct < _FLAG_CEILING_PCT:
        return "flag"
    return "warning"


def _signed_gap_pct(*, dgcis_value: int, other_value: int) -> Decimal:
    """`(other - dgcis) / dgcis`, as a percentage. Positive means the
    other source reports higher than DGCIS; the sign (not the magnitude)
    is what year-on-year flip detection watches."""
    return (Decimal(other_value - dgcis_value) / Decimal(dgcis_value)) * _HUNDRED


def _evaluate(
    *,
    check_name: str,
    hs6: str,
    flow: str,
    year: int,
    partner_country_code: str,
    dgcis_value: int | None,
    other_value: int | None,
    previous_signed_gap_pct: Decimal | None,
) -> MismatchResult | SkippedCheck:
    if dgcis_value is None or other_value is None:
        return SkippedCheck(
            check_name=check_name,
            partner_country_code=partner_country_code,
            year=year,
            reason="one or both sources have status NOT_REPORTED for this year — "
            "never treated as zero (D2), so no gap can be computed",
        )
    if dgcis_value == 0:
        return SkippedCheck(
            check_name=check_name,
            partner_country_code=partner_country_code,
            year=year,
            reason="dgcis value is ZERO — a percentage gap against a zero denominator "
            "is undefined, not computed as infinite or skipped-as-agreement",
        )

    signed_gap_pct = _signed_gap_pct(dgcis_value=dgcis_value, other_value=other_value)
    direction_flip = (
        previous_signed_gap_pct is not None
        and previous_signed_gap_pct != 0
        and signed_gap_pct != 0
        and (previous_signed_gap_pct > 0) != (signed_gap_pct > 0)
    )
    gap_pct = abs(signed_gap_pct)
    return MismatchResult(
        check_name=check_name,
        hs6=hs6,
        flow=flow,
        year=year,
        partner_country_code=partner_country_code,
        gap_pct=gap_pct,
        severity=_severity(gap_pct, direction_flip=direction_flip),
        direction_flip_yoy=direction_flip,
    )


async def _fetch_values(
    engine: AsyncEngine,
    *,
    hs6: str,
    flow: str,
    source: str,
    dataset_version: str,
    partner_country_code: str,
) -> dict[int, int | None]:
    """`{year: value_inr_paise}` for every `(hs6, flow, source,
    dataset_version, partner_country_code)` row present, keyed by the
    calendar year of `period_month`. A row with `value_inr_paise IS NULL`
    (status `NOT_REPORTED`) is kept as `None`, not dropped — the caller
    must see the year exists but has no value, distinct from the year
    simply never having been ingested."""
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(
                        normalized_trade_flows.c.period_month,
                        normalized_trade_flows.c.value_inr_paise,
                    ).where(
                        normalized_trade_flows.c.hs6 == hs6,
                        normalized_trade_flows.c.flow == flow,
                        normalized_trade_flows.c.source == source,
                        normalized_trade_flows.c.dataset_version == dataset_version,
                        normalized_trade_flows.c.partner_country_code == partner_country_code,
                    )
                )
            )
            .mappings()
            .all()
        )
    return {row["period_month"].year: row["value_inr_paise"] for row in rows}


async def _fetch_dgcis_partner_totals(
    engine: AsyncEngine, *, hs6: str, flow: str
) -> dict[str, dict[int, int | None]]:
    """`{partner_country_code: {year: value_inr_paise}}` — every DGCIS
    partner tracked for this `(hs6, flow)`, `'UNMAPPED'` included (it's
    excluded from check B individually per §10, but the caller does that
    exclusion explicitly rather than this fetch silently dropping it)."""
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(
                        normalized_trade_flows.c.partner_country_code,
                        normalized_trade_flows.c.period_month,
                        normalized_trade_flows.c.value_inr_paise,
                    ).where(
                        normalized_trade_flows.c.hs6 == hs6,
                        normalized_trade_flows.c.flow == flow,
                        normalized_trade_flows.c.source == "dgcis",
                        normalized_trade_flows.c.dataset_version == DGCIS_DATASET_VERSION,
                    )
                )
            )
            .mappings()
            .all()
        )
    by_partner: dict[str, dict[int, int | None]] = {}
    for row in rows:
        by_partner.setdefault(row["partner_country_code"], {})[row["period_month"].year] = row[
            "value_inr_paise"
        ]
    return by_partner


async def compute_check_a(
    engine: AsyncEngine, *, hs6: str, flow: str, years: list[int]
) -> tuple[list[MismatchResult], list[SkippedCheck]]:
    """DGCIS's national total (summed across every partner DGCIS tracks,
    `'UNMAPPED'` included — §10: unmapped partners "fold into check A/C's
    aggregate totals only") vs. India's own Comtrade submission
    (`partner_country_code='0'`, the World aggregate row from the
    reporter-role query)."""
    dgcis_by_partner = await _fetch_dgcis_partner_totals(engine, hs6=hs6, flow=flow)
    dgcis_totals: dict[int, int | None] = {}
    for year in years:
        year_values = [
            partner_years[year]
            for partner_years in dgcis_by_partner.values()
            if year in partner_years
        ]
        if not year_values or any(v is None for v in year_values):
            dgcis_totals[year] = None
        else:
            dgcis_totals[year] = sum(v for v in year_values if v is not None)

    comtrade_india_totals = await _fetch_values(
        engine,
        hs6=hs6,
        flow=flow,
        source="comtrade",
        dataset_version=COMTRADE_DATASET_VERSION_REPORTER_ROLE,
        partner_country_code=WORLD_AGGREGATE_PARTNER_CODE,
    )

    results: list[MismatchResult] = []
    skipped: list[SkippedCheck] = []
    previous_signed_gap: Decimal | None = None
    for year in sorted(years):
        dgcis_value = dgcis_totals.get(year)
        other_value = comtrade_india_totals.get(year)
        outcome = _evaluate(
            check_name=CHECK_A,
            hs6=hs6,
            flow=flow,
            year=year,
            partner_country_code=ALL_PARTNERS,
            dgcis_value=dgcis_value,
            other_value=other_value,
            previous_signed_gap_pct=previous_signed_gap,
        )
        if isinstance(outcome, MismatchResult):
            results.append(outcome)
            assert dgcis_value is not None and other_value is not None  # else _evaluate skips
            # gap_pct on the result is already abs() - re-derive the signed
            # value for next year's flip comparison rather than losing it.
            previous_signed_gap = _signed_gap_pct(dgcis_value=dgcis_value, other_value=other_value)
        else:
            skipped.append(outcome)
    return results, skipped


async def compute_check_b(
    engine: AsyncEngine, *, hs6: str, flow: str, years: list[int]
) -> tuple[list[MismatchResult], list[SkippedCheck]]:
    """DGCIS's per-partner figure vs. that partner's own Comtrade
    submission (partner-role query, that partner as reporter). `'UNMAPPED'`
    partners are excluded individually — an unmapped country can't be
    blamed for a specific partner-level gap it can't be identified for
    (§10).

    Mirror pairing, not matching flow labels: `flow` is DGCIS's own label
    (India's customs record). The partner's own Comtrade submission
    records the *same physical shipment* under the opposite flow from
    their own perspective — DGCIS `import` (India receiving) pairs with
    the partner's own `export` (partner sending), and vice versa. Fetching
    the partner's data under DGCIS's own flow label would silently query
    the wrong, unrelated direction (a real bug found by this module's own
    integration test)."""
    dgcis_by_partner = await _fetch_dgcis_partner_totals(engine, hs6=hs6, flow=flow)
    partner_mirror_flow = "export" if flow == "import" else "import"

    results: list[MismatchResult] = []
    skipped: list[SkippedCheck] = []
    for partner_country_code, dgcis_years in sorted(dgcis_by_partner.items()):
        if partner_country_code.startswith(UNMAPPED_PREFIX):
            continue
        partner_comtrade_years = await _fetch_values(
            engine,
            hs6=hs6,
            flow=partner_mirror_flow,
            source="comtrade",
            dataset_version=COMTRADE_DATASET_VERSION_PARTNER_ROLE,
            partner_country_code=partner_country_code,
        )
        previous_signed_gap: Decimal | None = None
        for year in sorted(years):
            dgcis_value = dgcis_years.get(year)
            other_value = partner_comtrade_years.get(year)
            outcome = _evaluate(
                check_name=CHECK_B,
                hs6=hs6,
                flow=flow,
                year=year,
                partner_country_code=partner_country_code,
                dgcis_value=dgcis_value,
                other_value=other_value,
                previous_signed_gap_pct=previous_signed_gap,
            )
            if isinstance(outcome, MismatchResult):
                results.append(outcome)
                assert dgcis_value is not None and other_value is not None
                previous_signed_gap = _signed_gap_pct(
                    dgcis_value=dgcis_value, other_value=other_value
                )
            else:
                skipped.append(outcome)
    return results, skipped


async def upsert_mismatch_checks(engine: AsyncEngine, results: list[MismatchResult]) -> int:
    if not results:
        return 0
    rows = [
        {
            "hs6": r.hs6,
            "flow": r.flow,
            "year": r.year,
            "check_name": r.check_name,
            "partner_country_code": r.partner_country_code,
            "gap_pct": r.gap_pct,
            "severity": r.severity,
            "direction_flip_yoy": r.direction_flip_yoy,
        }
        for r in results
    ]
    async with engine.begin() as conn:
        stmt = insert(analytics_mismatch_checks).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["hs6", "flow", "year", "check_name", "partner_country_code"],
            set_={
                "gap_pct": stmt.excluded.gap_pct,
                "severity": stmt.excluded.severity,
                "direction_flip_yoy": stmt.excluded.direction_flip_yoy,
            },
        )
        await conn.execute(stmt)
    return len(rows)
