"""Raw -> normalized layer transformation (`docs/PLAN.md` D7, §4) — the
module the build sequence's own layout didn't explicitly name (§3 lists
every `pipeline/*.py` ingestion job and every `report/*.py` consumer, but
not the raw->normalized step between them). Filling that real gap here
rather than skipping straight from raw tables to `report/mismatch.py`,
which the plan explicitly says reads `normalized_trade_flows`, not the
raw tables directly (§10: "Join is always normalized_trade_flows").

Three normalizers, one per source built so far:

- `normalize_dgcis_annual_rows`: `raw_dgcis_annual` -> `normalized_trade_flows`.
  DGCIS is natively INR (`fx_rate_used=NULL`, never round-tripped through
  USD, D8's explicit BLOCKER) and natively fiscal-year (`calendar='FY'`,
  `period_month` = 1 Jan of the fiscal year's first calendar year — the
  same "annual sources use Jan 1" convention `docs/PLAN.md` already
  documents for the normalized layer). `partner_country_code` resolves
  through `ref_country_crosswalk` — an unmapped DGCIS country name becomes
  `'UNMAPPED:<dgcis name>'` (§4's own documented policy, never a silent
  drop or a guessed code), the name embedded rather than a bare
  `'UNMAPPED'` constant: a real bug found live during the first
  ~250-country run with genuinely many distinct unmapped countries at
  once — a bare, shared sentinel collapsed every unmapped country in the
  same `(hs6, flow, period_month)` scope onto the *same*
  `normalized_trade_flows` unique key, raising a real
  `CardinalityViolationError` on the bulk upsert (two rows for two
  different real countries both proposing the same key in one statement).
  Embedding the name keeps every real, distinct country its own row.

- `normalize_comtrade_rows`: `raw_comtrade_records` -> `normalized_trade_flows`.
  Comtrade is USD and calendar-year native (`calendar='CY'`) — real FX
  conversion via `app.fx` (already built, D8's exact cache contract) is
  required here, unlike DGCIS. `partner_country_code` is already the
  right coding scheme (Comtrade's own numeric codes) — no crosswalk
  needed for this source.

- `normalize_baci_rows`: `raw_baci_records` -> `normalized_trade_flows`.
  BACI is USD and calendar-year native like Comtrade (same real FX
  conversion pattern), but has no explicit flow column — `flow` is
  derived from which side of the exporter/importer pair India is on.
  `partner_country_code` is already the right coding scheme (BACI's own
  numeric codes, verified live to be the *same* scheme Comtrade uses,
  including India's own `699` — no crosswalk needed here either).

All three derive status via the shared `_derive_status`: `'NOT_REPORTED'` when
the value cell is genuinely blank/absent, `'ZERO'` when the source
explicitly reported a numeric `0` — never conflated with `NOT_REPORTED`
(D2) — `'QTY_MISSING'` when a real nonzero value is present but the
source has no quantity for that cell (§5's table: "value present,
quantity null" — true for every `raw_dgcis_annual` row, since that report
never returns a quantity at all, and true for some real
`raw_comtrade_records` rows where `net_weight_kg` is null), and `'OK'`
only when both a real value and a real quantity are present. Finer status
distinctions (`PROVISIONAL`, ...) are not yet populated by either
normalizer — flagged as a real, deliberate gap, not guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.comtrade_mirror import INDIA_CODE
from app.warehouse.schema import (
    normalized_trade_flows,
    raw_baci_records,
    raw_comtrade_records,
    raw_dgcis_annual,
    ref_country_crosswalk,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

UNMAPPED_PREFIX = "UNMAPPED:"
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
# Fixed logic-version string, not a data-vintage encoding - matches how
# DGCIS/Comtrade's own dataset_version constants version the normalizer's
# logic, not the specific raw scrape/vintage that fed it. Re-normalizing
# after a newer real BACI vintage is loaded intentionally overwrites these
# rows in place (the normalized layer always reflects whichever vintage
# was most recently normalized) - raw_baci_records itself keeps every
# historical vintage forever (its own real unique key includes vintage),
# so nothing is lost, only the normalized layer's view moves forward.
BACI_DATASET_VERSION = "baci-v1"


def _to_paise(value: Decimal) -> int:
    """USD/INR-value `Decimal` -> integer paise, rounded half-up — not
    Python's `int()` (truncates toward zero, a systematic downward bias on
    every converted row). Matches `app.report.landed_cost`'s own rounding
    mode (architect-review finding, 2026-08-26: this module's own comment
    culture already states "rounding happens once, at render time, never
    mid-calculation" as a deliberate rule — this was a real gap against that
    rule, not a difference in per-row magnitude that would matter)."""
    return int((value * 100).to_integral_value(rounding=ROUND_HALF_UP))


def _derive_status(*, value: int | None, quantity: Decimal | None) -> str:
    """§5's status table, the value/quantity slice: `NOT_REPORTED` (no
    value at all) takes priority over everything else - a genuinely absent
    cell is never "zero" or "missing quantity". `ZERO` is checked before
    `QTY_MISSING`: a real zero-value trade flow trivially has no
    meaningful quantity to be missing, so a null quantity on a zero-value
    row isn't a data gap - only a real nonzero value with no quantity is
    (`QTY_MISSING`)."""
    if value is None:
        return "NOT_REPORTED"
    if value == 0:
        return "ZERO"
    if quantity is None:
        return "QTY_MISSING"
    return "OK"


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
        return self.by_dgcis_name.get(dgcis_name, UNMAPPED_PREFIX + dgcis_name)


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
        # Per-row fault isolation (architect-review finding, 2026-08-26):
        # `_dgcis_fiscal_year_to_period_month` deliberately raises on a
        # genuinely malformed fiscal_year_label rather than guessing a year
        # (its own docstring) — but before this fix, one bad label anywhere
        # among (potentially) ~250 countries' worth of real scraped rows
        # for this hs6 aborted normalization for *every* row, not just the
        # offending one, inconsistent with `app.pipeline.comtrade_mirror.
        # fetch_all_countries_annual`'s own documented "one bad country
        # never aborts the batch" discipline one layer upstream. Skip-and-
        # log the one bad row; every other row still normalizes.
        try:
            _year, period_month = _dgcis_fiscal_year_to_period_month(raw["fiscal_year_label"])
        except ValueError:
            logger.warning(
                "normalize_dgcis_annual_rows.malformed_fiscal_year_label",
                hs8=raw["hs8"],
                partner_country=raw["partner_country"],
                fiscal_year_label=raw["fiscal_year_label"],
            )
            continue
        value = raw["value_inr_paise"]
        # raw_dgcis_annual has no quantity column at all - this report
        # never returns one (verified live, §1) - so quantity is always
        # None here, meaning every real nonzero-value row is QTY_MISSING.
        status = _derive_status(value=value, quantity=None)
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

    if not normalized_rows:
        # Every row for this hs6 had a malformed fiscal_year_label (all
        # skipped above) — an empty `.values([])` upsert is invalid, and
        # there is nothing real to write anyway.
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
        value_original_paise = _to_paise(value_usd) if value_usd is not None else None
        value_inr_paise = _to_paise(value_usd * rate) if value_usd is not None else None
        status = _derive_status(value=value_inr_paise, quantity=raw["net_weight_kg"])
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


async def normalize_baci_rows(
    engine: AsyncEngine, *, hs6: str, fx_rates: dict[int, tuple[Decimal, date]]
) -> int:
    """Normalize every `raw_baci_records` row for `hs6` into
    `normalized_trade_flows`. Same `fx_rates` contract as
    `normalize_comtrade_rows` (caller-resolved via `app.fx`, a period with
    no entry is skipped, never guessed at 1:1).

    `flow` is derived from which side of the pair India is on (BACI has
    no explicit flow column, unlike Comtrade's `flowCode`): `importer_code
    == INDIA_CODE` means India importing, `exporter_code == INDIA_CODE`
    means India exporting. The raw-layer filter that produced these rows
    (`app.pipeline.baci.parse_baci_year_csv`) already guarantees India is
    on at least one side of every row; a row with India on *neither* side
    would be a real caller bug upstream, not a case this function guesses
    at — skipped defensively rather than silently mis-flowed.

    `basis='FOB'` for every row, deliberately never `'CIF'` even for
    `flow='import'` — unlike Comtrade/DGCIS's CIF-for-imports/FOB-for-
    exports convention, BACI reports **both** directions on a FOB basis by
    construction (CEPII's own stated methodology: "already CIF/FOB
    -adjusted", `docs/PLAN.md` §1) specifically so cross-country FOB
    comparison is valid either way — storing it as `'CIF'` here would
    misdescribe what the number actually is."""
    async with engine.connect() as conn:
        raw_rows = (
            (await conn.execute(select(raw_baci_records).where(raw_baci_records.c.hs6 == hs6)))
            .mappings()
            .all()
        )

    if not raw_rows:
        return 0

    normalized_rows = []
    for raw in raw_rows:
        year = raw["year"]
        if year not in fx_rates:
            continue
        if raw["importer_code"] == INDIA_CODE:
            flow = "import"
            partner_country_code = raw["exporter_code"]
        elif raw["exporter_code"] == INDIA_CODE:
            flow = "export"
            partner_country_code = raw["importer_code"]
        else:
            continue  # defensive - the raw-layer filter should prevent this

        rate, rate_date = fx_rates[year]
        value_usd = raw["value_fob_usd"]
        value_original_paise = _to_paise(value_usd) if value_usd is not None else None
        value_inr_paise = _to_paise(value_usd * rate) if value_usd is not None else None
        status = _derive_status(value=value_inr_paise, quantity=raw["quantity_kg"])
        normalized_rows.append(
            {
                "source": "baci",
                "hs6": raw["hs6"],
                "hs8": None,
                "hs_revision": raw["hs_revision"],
                "flow": flow,
                "period_month": date(year, 1, 1),
                "calendar": "CY",
                "partner_country_code": partner_country_code,
                "basis": "FOB",
                "currency": "USD",
                "universe": "baci-reconciled",
                "dataset_version": BACI_DATASET_VERSION,
                "is_provisional": False,
                "status": status,
                "status_detail": None,
                "value_inr_paise": value_inr_paise,
                "value_original_currency_paise": value_original_paise,
                "fx_rate_used": rate,
                "fx_rate_date": rate_date,
                "quantity_kg": raw["quantity_kg"],
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
