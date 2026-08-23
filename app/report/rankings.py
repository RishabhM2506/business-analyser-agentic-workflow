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

from sqlalchemy import select
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
    async with engine.begin() as conn:
        stmt = insert(analytics_partner_rankings).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["hs6", "flow", "year", "partner_country_code"],
            set_={
                "rank": stmt.excluded.rank,
                "value_inr_paise": stmt.excluded.value_inr_paise,
                "status": stmt.excluded.status,
            },
        )
        await conn.execute(stmt)
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
