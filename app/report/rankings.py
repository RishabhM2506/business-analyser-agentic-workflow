"""D14 precompute: `analytics_partner_rankings` (`docs/PLAN.md` §4 DDL,
§12), plus HHI (§11: "computed once per (hs6, flow, year), independent of
the request's chosen top-N") since it's defined directly over this
module's own ranked output and has no dedicated storage table of its own
in §4's schema.

Rankings are always DGCIS-sourced. The master prompt names DGCIS Tradestat
as this pipeline's backbone data source; `comtrade_mirror.py`'s own
docstring calls Comtrade "mirror/benchmark only" — so the headline partner
list a report shows must come from the backbone, with Comtrade reserved
for `mismatch.py`'s cross-checks, never for ranking itself.

Only a row with a real, comparable value gets an integer rank
(`value_inr_paise` non-`NULL` — status `OK`/`ZERO`/`QTY_MISSING`/
`PROVISIONAL`/`UNIT_MISMATCH`, per §12's "group 1/3 vs group 2" split);
`NOT_REPORTED`/`SUPPRESSED`/`NOT_YET_PUBLISHED`/`FETCH_FAILED`/
`CODE_RETIRED` rows get `rank=NULL` — never approximated into the ranked
list. `'UNMAPPED'` partners are included and ranked here (they're a real,
if unidentified, trading relationship) — §10 excludes `'UNMAPPED'` only
from check B specifically, not from rankings in general. Ties (identical
`value_inr_paise`) are broken by `partner_country_code` ascending, since
`ix_apr_rank_where_present` requires every present rank to be unique
within `(hs6, flow, year)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.warehouse.schema import analytics_partner_rankings, normalized_trade_flows

# §12 group 1/3: a real, comparable value is present for exactly these
# statuses - everything else has no value to rank by.
_RANKABLE_STATUSES = ("OK", "ZERO", "QTY_MISSING", "PROVISIONAL", "UNIT_MISMATCH")


@dataclass(frozen=True)
class PartnerRanking:
    hs6: str
    flow: str
    year: int
    partner_country_code: str
    rank: int | None
    value_inr_paise: int | None
    status: str


async def compute_partner_rankings(
    engine: AsyncEngine, *, hs6: str, flow: str, year: int
) -> list[PartnerRanking]:
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(
                        normalized_trade_flows.c.partner_country_code,
                        normalized_trade_flows.c.value_inr_paise,
                        normalized_trade_flows.c.status,
                        normalized_trade_flows.c.period_month,
                    ).where(
                        normalized_trade_flows.c.hs6 == hs6,
                        normalized_trade_flows.c.flow == flow,
                        normalized_trade_flows.c.source == "dgcis",
                    )
                )
            )
            .mappings()
            .all()
        )
    year_rows = [r for r in rows if r["period_month"].year == year]

    rankable = [r for r in year_rows if r["status"] in _RANKABLE_STATUSES]
    unrankable = [r for r in year_rows if r["status"] not in _RANKABLE_STATUSES]
    rankable_sorted = sorted(
        rankable, key=lambda r: (-r["value_inr_paise"], r["partner_country_code"])
    )

    results = [
        PartnerRanking(
            hs6=hs6,
            flow=flow,
            year=year,
            partner_country_code=r["partner_country_code"],
            rank=rank,
            value_inr_paise=r["value_inr_paise"],
            status=r["status"],
        )
        for rank, r in enumerate(rankable_sorted, start=1)
    ]
    results.extend(
        PartnerRanking(
            hs6=hs6,
            flow=flow,
            year=year,
            partner_country_code=r["partner_country_code"],
            rank=None,
            value_inr_paise=None,
            status=r["status"],
        )
        for r in unrankable
    )
    return results


async def upsert_partner_rankings(engine: AsyncEngine, rankings: list[PartnerRanking]) -> int:
    """Delete-then-insert per `(hs6, flow, year)`, not a plain
    `ON CONFLICT` upsert — a real bug found live: when a re-run changes
    *relative* ranks (e.g. a newly-ingested country outranks the
    previously-#1 partner), the fresh batch can transiently collide with
    `ix_apr_rank_where_present` (the partial unique index on `rank`)
    against a row that's still sitting at its *old* rank value, even
    though `ON CONFLICT (hs6, flow, year, partner_country_code)` correctly
    targets the primary key — a plain `ON CONFLICT` clause only suppresses
    the one named constraint's violations, not a different unique index's.
    Clearing every row for the affected years first means the fresh batch
    is inserted into an empty slate, never colliding with stale rank
    assignments from a prior run."""
    if not rankings:
        return 0
    rows = [
        {
            "hs6": r.hs6,
            "flow": r.flow,
            "year": r.year,
            "partner_country_code": r.partner_country_code,
            "rank": r.rank,
            "value_inr_paise": r.value_inr_paise,
            "status": r.status,
        }
        for r in rankings
    ]
    hs6 = rankings[0].hs6
    flow = rankings[0].flow
    years = {r.year for r in rankings}
    async with engine.begin() as conn:
        await conn.execute(
            delete(analytics_partner_rankings).where(
                analytics_partner_rankings.c.hs6 == hs6,
                analytics_partner_rankings.c.flow == flow,
                analytics_partner_rankings.c.year.in_(years),
            )
        )
        await conn.execute(insert(analytics_partner_rankings).values(rows))
    return len(rows)


def compute_hhi(rankings: list[PartnerRanking]) -> Decimal | None:
    """Herfindahl-Hirschman Index — `sum(share_i ** 2)` over every ranked
    (i.e. valued) partner, `share_i = value_i / total`. `None` (never
    `0.0`) when there is no rankable total to compute a share against —
    an HHI of exactly `0.0` would falsely read as "perfectly
    unconcentrated," a real claim this function is in no position to make
    from zero data."""
    valued_amounts = [r.value_inr_paise for r in rankings if r.value_inr_paise is not None]
    total = sum(valued_amounts)
    if total <= 0:
        return None
    return sum(((Decimal(v) / Decimal(total)) ** 2 for v in valued_amounts), Decimal(0))
