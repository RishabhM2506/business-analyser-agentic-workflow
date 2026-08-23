"""MSP + Cost of Production ingestion — Tier-1 agriculture-source
expansion, 2026-08-24 (user-directed, following the Agmarknet unit).
Powers a real "mandi price vs. government MSP" comparison per product,
the first slice of this expansion.

Real mechanics verified live against data.gov.in resource
`50012e24-85bc-4731-a6a9-2918caf5f0bf`, "Commodity-wise Minimum Support
Price (MSP) and Cost of Production of Mandated Agricultural Crops from
2017-18 and 2022-23" (Rajya Sabha, org sector "All"). Uses
`app.pipeline.data_gov_in`'s shared client, same platform/API as
Agmarknet.

**A small, static reference table**: 22 total real rows (one per
mandated crop), not a daily-updated feed — real fields
`{'Sl. No.', 'Crops', 'Commodity', '2017-18 - Cost', '2017-18 - MSP',
'2022-23 - Cost', '2022-23 - MSP'}`, every numeric field a real JSON
number (`"type": "double"`), unlike Agmarknet's string-typed price
fields — parsed here via `Decimal(str(...))` regardless, since D8's
"money never a float" rule applies to how *this pipeline* stores the
value, not to what shape the source happened to send it in.

**A real, live-confirmed silent page-size cap**: a `limit=25` request
returned exactly 10 records (`"limit": 10` echoed back in the response,
despite requesting 25) — this resource silently caps its own effective
page size regardless of what's asked for. `app.pipeline.data_gov_in.
fetch_all_pages` was fixed to page until a genuinely empty page (using
each page's own *actual* length to advance `offset`) specifically because
of this real finding — see that module's docstring for the full story.

**Wide-to-long normalization**: the source's own two year-pair columns
(`2017-18` and `2022-23`) are split here into one row per
`(commodity, year_label)`, matching this pipeline's "one fact per row"
raw-layer convention (`app.pipeline.baci`/`app.pipeline.dgcis` do the
same for their own wide source shapes) rather than mirroring the source's
wide columns verbatim into the warehouse.

**Unit inherited, not independently reconfirmed**: MSP/cost figures are
recorded here as Rs./Quintal (`*_inr_paise_per_qtl`), matching
`raw_agmarknet_prices`'s existing convention and MSP's well-established
public quotation convention in India — this API's own metadata does not
explicitly label the unit either (checked; same gap as Agmarknet's).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

import httpx
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.pipeline.data_gov_in import fetch_all_pages
from app.warehouse.schema import raw_msp_records

RESOURCE_PATH = "/resource/50012e24-85bc-4731-a6a9-2918caf5f0bf"
_DEFAULT_PAGE_SIZE = 25  # the real resource caps this to 10 regardless - see module docstring

# The real field's own two year pairs, as of this dataset's last real
# update (2023-11-20) - a real, flagged limitation: a future revision of
# this source could add a new year pair under a different field-name
# suffix, which this fixed list would silently miss. No pagination-style
# "discover all year columns automatically" mechanism exists for this
# shape; revisit if data.gov.in publishes a newer vintage.
_YEAR_LABELS = ("2017-18", "2022-23")


@dataclass(frozen=True)
class MspRecord:
    crops: str
    commodity: str
    year_label: str
    cost_inr_paise_per_qtl: int | None
    msp_inr_paise_per_qtl: int | None
    raw_payload: dict[str, object]


def _parse_paise(value: object) -> int | None:
    """`None` for anything that isn't a clean number - never guessed or
    coerced to 0, per D2."""
    if value is None:
        return None
    try:
        rupees = Decimal(str(value))
    except InvalidOperation:
        return None
    return int(rupees * 100)


def _records_from_raw(raw: dict[str, object]) -> list[MspRecord]:
    crops = raw.get("crops")
    commodity = raw.get("commodity")
    if not isinstance(crops, str) or not isinstance(commodity, str):
        return []

    records = []
    for year_label in _YEAR_LABELS:
        suffix = year_label.replace("-", "_")
        cost_key = f"_{suffix}___cost"
        msp_key = f"_{suffix}___msp"
        records.append(
            MspRecord(
                crops=crops,
                commodity=commodity,
                year_label=year_label,
                cost_inr_paise_per_qtl=_parse_paise(raw.get(cost_key)),
                msp_inr_paise_per_qtl=_parse_paise(raw.get(msp_key)),
                raw_payload=raw,
            )
        )
    return records


async def fetch_all_records(
    client: httpx.AsyncClient, *, api_key: str, page_size: int = _DEFAULT_PAGE_SIZE
) -> list[MspRecord]:
    """Fetches every real row from the MSP-and-cost-of-production
    resource and normalizes each into one `MspRecord` per year pair."""
    raw_rows = await fetch_all_pages(
        client, resource_path=RESOURCE_PATH, api_key=api_key, page_size=page_size
    )
    records: list[MspRecord] = []
    for raw in raw_rows:
        records.extend(_records_from_raw(raw))
    return records


async def upsert_msp_records(engine: AsyncEngine, records: list[MspRecord]) -> int:
    """Bulk-upsert into `raw_msp_records`, keyed on that table's real
    unique constraint `(commodity, year_label)` - idempotent by
    construction, same pattern as every other raw-layer upsert in this
    pipeline."""
    if not records:
        return 0
    fetched_at = datetime.now(UTC)
    rows = [
        {
            "fetched_at": fetched_at,
            "crops": r.crops,
            "commodity": r.commodity,
            "year_label": r.year_label,
            "cost_inr_paise_per_qtl": r.cost_inr_paise_per_qtl,
            "msp_inr_paise_per_qtl": r.msp_inr_paise_per_qtl,
            "raw_payload": r.raw_payload,
        }
        for r in records
    ]
    async with engine.begin() as conn:
        stmt = insert(raw_msp_records).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["commodity", "year_label"],
            set_={
                "fetched_at": stmt.excluded.fetched_at,
                "crops": stmt.excluded.crops,
                "cost_inr_paise_per_qtl": stmt.excluded.cost_inr_paise_per_qtl,
                "msp_inr_paise_per_qtl": stmt.excluded.msp_inr_paise_per_qtl,
                "raw_payload": stmt.excluded.raw_payload,
            },
        )
        await conn.execute(stmt)
    return len(rows)
