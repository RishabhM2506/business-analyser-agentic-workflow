"""Raw -> normalized layer transformation (`docs/PLAN.md` D7, §4) — the
module the build sequence's own layout didn't explicitly name (§3 lists
every `pipeline/*.py` ingestion job and every `report/*.py` consumer, but
not the raw->normalized step between them). Filling that real gap here
rather than skipping straight from raw tables to `report/mismatch.py`,
which the plan explicitly says reads `normalized_trade_flows`, not the
raw tables directly (§10: "Join is always normalized_trade_flows").

Two normalizers, one per source built so far:

- `normalize_dgcis_annual_rows`: `raw_dgcis_annual` -> `normalized_trade_flows`.
  DGCIS is natively INR (`fx_rate_used=NULL`, never round-tripped through
  USD, D8's explicit BLOCKER) and natively fiscal-year (`calendar='FY'`,
  `period_month` = 1 Jan of the fiscal year's first calendar year — the
  same "annual sources use Jan 1" convention `docs/PLAN.md` already
  documents for the normalized layer). `partner_country_code` resolves
  through `ref_country_crosswalk` — an unmapped DGCIS country name
  becomes `'UNMAPPED'` (§4's own documented policy), never a silent drop
  or a guessed code.

- `normalize_comtrade_rows`: `raw_comtrade_records` -> `normalized_trade_flows`.
  Comtrade is USD and calendar-year native (`calendar='CY'`) — real FX
  conversion via `app.fx` (already built, D8's exact cache contract) is
  required here, unlike DGCIS. `partner_country_code` is already the
  right coding scheme (Comtrade's own numeric codes) — no crosswalk
  needed for this source.

Both write `status='OK'` when a real value is present and `'ZERO'` when
the source explicitly reported a numeric `0` — never conflated (D2) — and
`'NOT_REPORTED'` when the source cell was genuinely blank/absent. Finer
status distinctions (`PROVISIONAL`, `QTY_MISSING`, ...) are not yet
populated by either normalizer — flagged as a real, deliberate gap, not
guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.comtrade_mirror import INDIA_CODE
from app.warehouse.schema import (
    normalized_trade_flows,
    raw_comtrade_records,
    raw_dgcis_annual,
    ref_country_crosswalk,
)

_UNMAPPED = "UNMAPPED"
DGCIS_DATASET_VERSION = "dgcis-annual-v1"
# Two distinct dataset_versions, not one - a real bug found before
# mismatch.py could be built: a Query 1 row (India self-reporting, e.g.
# reporter=699/partner=792/flow=export = "India's own claimed export to
# Turkey") and a Query 2 row (reporter=792/partner=699/flow=export =
# "Turkey's own claimed export to India", i.e. the mirror of India's
# *import*) both normalize to the same (partner_country_code=792,
# flow='export') pair - without a discriminator they collide on
# normalized_trade_flows' unique key and one silently overwrites the
# other via ON CONFLICT DO UPDATE. check_A needs only the reporter-shape
# rows (partner_country_code='0', §10); check_B needs only the
# partner-shape rows (the foreign country's own mirror figure) - keeping
# them in separate dataset_version buckets keeps both queryable.
COMTRADE_DATASET_VERSION_REPORTER_ROLE = "comtrade-mirror-reporter-v1"
COMTRADE_DATASET_VERSION_PARTNER_ROLE = "comtrade-mirror-partner-v1"


def _dgcis_fiscal_year_to_period_month(fiscal_year_label: str) -> tuple[int, date]:
    """`"2020 - 2021"` -> `(2020, date(2020, 1, 1))` — DGCIS's own label,
    verbatim, split on the first year. Raises on a genuinely malformed
    label rather than guessing a year — a normalizer silently inventing a
    date would be a worse failure than a loud one."""
    first_year_text = fiscal_year_label.split("-")[0].strip()
    year = int(first_year_text)
    return year, date(year, 1, 1)


@dataclass(frozen=True)
class CountryCrosswalk:
    """`dgcis_country_name -> country_code`, loaded once per normalizer
    call (not `lru_cache`d like the static reference CSVs — this table is
    live-curated, expected to change as new countries are mapped)."""

    by_dgcis_name: dict[str, str]

    def resolve(self, dgcis_name: str) -> str:
        return self.by_dgcis_name.get(dgcis_name, _UNMAPPED)


async def load_country_crosswalk(engine: AsyncEngine) -> CountryCrosswalk:
    async with engine.connect() as conn:
        rows = (await conn.execute(select(ref_country_crosswalk))).mappings().all()
    return CountryCrosswalk(
        by_dgcis_name={row["dgcis_country_name"]: row["country_code"] for row in rows}
    )


async def normalize_dgcis_annual_rows(
    engine: AsyncEngine, *, hs6: str, crosswalk: CountryCrosswalk
) -> int:
    """Normalize every `raw_dgcis_annual` row for `hs6` (matched by the
    row's `hs8` starting with `hs6` — DGCIS rows are HS8-level) into
    `normalized_trade_flows`. Idempotent: re-running upserts the same
    real unique key, never a duplicate row."""
    async with engine.connect() as conn:
        raw_rows = (
            (
                await conn.execute(
                    select(raw_dgcis_annual).where(raw_dgcis_annual.c.hs8.startswith(hs6))
                )
            )
            .mappings()
            .all()
        )

    if not raw_rows:
        return 0

    normalized_rows = []
    for raw in raw_rows:
        _year, period_month = _dgcis_fiscal_year_to_period_month(raw["fiscal_year_label"])
        value = raw["value_inr_paise"]
        status = "NOT_REPORTED" if value is None else ("ZERO" if value == 0 else "OK")
        normalized_rows.append(
            {
                "source": "dgcis",
                "hs6": raw["hs8"][:6],
                "hs8": raw["hs8"],
                "hs_revision": "ITC-HS",
                "flow": raw["flow"],
                "period_month": period_month,
                "calendar": "FY",
                "partner_country_code": crosswalk.resolve(raw["partner_country"]),
                "basis": "CIF" if raw["flow"] == "import" else "FOB",
                "currency": "INR",
                "universe": "india-customs",
                "dataset_version": DGCIS_DATASET_VERSION,
                "is_provisional": False,
                "status": status,
                "status_detail": None,
                "value_inr_paise": value,
                "value_original_currency_paise": value,
                "fx_rate_used": None,
                "fx_rate_date": None,
                "quantity_kg": None,
            }
        )

    async with engine.begin() as conn:
        stmt = insert(normalized_trade_flows).values(normalized_rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                "source",
                "hs6",
                "hs8",
                "flow",
                "period_month",
                "partner_country_code",
                "dataset_version",
            ],
            set_={
                "status": stmt.excluded.status,
                "value_inr_paise": stmt.excluded.value_inr_paise,
                "value_original_currency_paise": stmt.excluded.value_original_currency_paise,
            },
        )
        await conn.execute(stmt)
    return len(normalized_rows)


async def normalize_comtrade_rows(
    engine: AsyncEngine, *, hs6: str, fx_rates: dict[int, tuple[Decimal, date]]
) -> int:
    """Normalize every `raw_comtrade_records` row for `hs6` into
    `normalized_trade_flows`. `fx_rates` maps `period year -> (rate,
    rate_date)` — the caller resolves these via `app.fx.cache.FxCache`
    (already built, D8's exact cache contract) rather than this module
    reaching into Redis/HTTP itself, keeping this a pure transformation.
    A period with no entry in `fx_rates` is skipped (not silently
    converted at 1:1 or dropped as ZERO) — logged by the caller."""
    async with engine.connect() as conn:
        raw_rows = (
            (
                await conn.execute(
                    select(raw_comtrade_records).where(raw_comtrade_records.c.cmd_code == hs6)
                )
            )
            .mappings()
            .all()
        )

    if not raw_rows:
        return 0

    normalized_rows = []
    for raw in raw_rows:
        year = raw["period"]
        if year not in fx_rates:
            continue
        rate, rate_date = fx_rates[year]
        value_usd = raw["primary_value_usd"]
        value_original_paise = int(value_usd * 100) if value_usd is not None else None
        value_inr_paise = int(value_usd * rate * 100) if value_usd is not None else None
        status = "NOT_REPORTED" if value_usd is None else ("ZERO" if value_usd == 0 else "OK")
        flow = "import" if raw["flow_code"] == "M" else "export"
        # Query 1 (role="reporter") rows have reporter_code=699 (India) and
        # partner_code = the real foreign country (or '0' for the World
        # aggregate) - partner_code is already right. Query 2
        # (role="partner") rows have partner_code=699 (India) fixed and
        # reporter_code = the real foreign country instead - using
        # partner_code there would wrongly store India as its own trade
        # partner. Resolve whichever side isn't India (real bug found
        # before mismatch.py could be built: check_B needs Query 2's
        # foreign reporter identified correctly, not collapsed to '699').
        is_reporter_role_row = raw["reporter_code"] == INDIA_CODE
        partner_country_code = raw["partner_code"] if is_reporter_role_row else raw["reporter_code"]
        dataset_version = (
            COMTRADE_DATASET_VERSION_REPORTER_ROLE
            if is_reporter_role_row
            else COMTRADE_DATASET_VERSION_PARTNER_ROLE
        )
        normalized_rows.append(
            {
                "source": "comtrade",
                "hs6": raw["cmd_code"],
                "hs8": None,
                "hs_revision": "H6",
                "flow": flow,
                "period_month": date(year, 1, 1),
                "calendar": "CY",
                "partner_country_code": partner_country_code,
                "basis": "CIF" if flow == "import" else "FOB",
                "currency": "USD",
                "universe": "un-comtrade-mirror",
                "dataset_version": dataset_version,
                "is_provisional": False,
                "status": status,
                "status_detail": None,
                "value_inr_paise": value_inr_paise,
                "value_original_currency_paise": value_original_paise,
                "fx_rate_used": rate,
                "fx_rate_date": rate_date,
                "quantity_kg": raw["net_weight_kg"],
            }
        )

    if not normalized_rows:
        return 0

    async with engine.begin() as conn:
        stmt = insert(normalized_trade_flows).values(normalized_rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                "source",
                "hs6",
                "hs8",
                "flow",
                "period_month",
                "partner_country_code",
                "dataset_version",
            ],
            set_={
                "status": stmt.excluded.status,
                "value_inr_paise": stmt.excluded.value_inr_paise,
                "value_original_currency_paise": stmt.excluded.value_original_currency_paise,
                "fx_rate_used": stmt.excluded.fx_rate_used,
                "fx_rate_date": stmt.excluded.fx_rate_date,
            },
        )
        await conn.execute(stmt)
    return len(normalized_rows)
