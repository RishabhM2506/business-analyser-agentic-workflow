"""FAOSTAT (UN Food and Agriculture Organization) bulk-file ingestion —
Tier-2 international-source expansion, 2026-08-25 (user-directed). Powers
cross-country production context for any tracked commodity (e.g. "how
does India's [missing] poppy-seed production compare to Türkiye's real
7,922 t?").

**FAOSTAT's own REST API is currently down** — real, live-confirmed
2026-08-24/25: `https://fenixservices.fao.org/faostat/api/v1/...`
consistently returns `HTTP 521` (Cloudflare's "origin server is down"),
reproduced on two separate real attempts. The Bulk Download service
(`https://bulks-faostat.fao.org/production/...`) *is* live and working —
no auth, no rate limit hit, a real, current file confirmed:
`Production_Crops_Livestock_E_All_Data.zip`, last modified 2025-12-31,
25,138,572 bytes.

**Real, verified format**: one CSV per file (`t,i,j,k,v,q`-style but for
production stats, not trade), wide across years —
`Area Code, Area Code (M49), Area, Item Code, Item Code (CPC), Item,
Element Code, Element, Unit, Y1961, Y1961F, Y1961N, Y1962, ...` through
`Y2024`. `Element` is one of `"Area harvested"` (ha), `"Yield"` (kg/ha),
`"Production"` (t) — confirmed live via the file's own
`Production_Crops_Livestock_E_Elements.csv` reference.

**A real, authoritative flag legend** (confirmed live via the file's own
`Production_Crops_Livestock_E_Flags.csv`): `A`=Official figure,
`E`=Estimated value, `I`=Value imputed by a receiving agency,
`M`="Missing value; data cannot exist", `X`=Figure from external
organization. **`M` means the raw cell is a genuinely empty string, not
a literal `0`** (confirmed live: India + Poppy seed + Production, every
year 2015-2024, is flag `M` with an empty value) — `_parse_value` treats
`M` as authoritative regardless of what's in the cell (never trusts a
value alongside an `M` flag, matching `app.pipeline.dgcis`'s identical
"the marker is authoritative regardless" precedent for its own `"(A)"`
advance marker), and D2's "missing != 0" discipline is preserved: `M`
(and any other unparseable/blank cell) becomes `None`, never `0`.

**This module is item-agnostic by design**: `parse_production_csv` takes
`item_names` as a caller-supplied filter (real FAOSTAT `Item` strings,
e.g. `{"Poppy seed"}`), never a hardcoded commodity — the same "caller
decides what to track" contract as `app.pipeline.baci`'s `hs6_codes`
parameter. There is no automatic HS6<->FAOSTAT-item crosswalk (FAOSTAT
uses CPC codes, a different classification); a caller must supply the
real FAOSTAT item name(s) it cares about, verified against the file's own
`*_ItemCodes.csv`, the same "no automatic guessing across code systems"
discipline already established for BACI's country codes.

**Every country/region row is kept, not just India** — unlike this
pipeline's other sources, FAOSTAT's value here is cross-country context,
so a caller filtering by item alone (no country filter) is the normal
case, not an oversight.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import IO

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.warehouse.schema import raw_faostat_records

PRODUCTION_ZIP_URL = (
    "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data.zip"
)
PRODUCTION_CSV_MEMBER = "Production_Crops_Livestock_E_All_Data.csv"

_MISSING_FLAG = "M"


class FaostatParseError(Exception):
    """Raised for a real structural problem with the CSV - never silently
    skips the whole file."""


@dataclass(frozen=True)
class FaostatRecord:
    area_code: str
    area: str
    item_code: str
    item: str
    element: str
    unit: str
    year: int
    value: Decimal | None
    flag: str | None


def _parse_value(raw_value: str, *, flag: str) -> Decimal | None:
    """`None` for a genuinely missing cell (D2's "missing != 0"
    discipline) - and `flag == 'M'` ("Missing value; data cannot exist",
    FAOSTAT's own real flag legend) is authoritative regardless of what
    the value cell happens to contain, matching `app.pipeline.dgcis`'s
    identical "the marker is authoritative regardless" precedent."""
    if flag == _MISSING_FLAG:
        return None
    text = raw_value.strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_production_csv(fh: IO[bytes], *, item_names: set[str]) -> Iterator[FaostatRecord]:
    """Stream one real FAOSTAT production CSV (`fh`, opened from the ZIP
    member, never read whole into memory), yielding one `FaostatRecord`
    per `(area, item, element, year)` for every year present, for rows
    matching `item_names` - every country/region kept, this is
    cross-country reference data by design."""
    reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", newline=""))
    if reader.fieldnames is None:
        raise FaostatParseError("FAOSTAT CSV has no header row")
    required = {"Area Code", "Area", "Item Code", "Item", "Element", "Unit"}
    if not required.issubset(reader.fieldnames):
        raise FaostatParseError(
            f"FAOSTAT CSV is missing expected columns {sorted(required)}; found {reader.fieldnames}"
        )
    year_columns = sorted(
        col for col in reader.fieldnames if col.startswith("Y") and col[1:].isdigit()
    )

    for row in reader:
        if row["Item"] not in item_names:
            continue
        for year_col in year_columns:
            year = int(year_col[1:])
            flag_col = f"{year_col}F"
            raw_value = row.get(year_col, "")
            flag = row.get(flag_col, "").strip() or None
            yield FaostatRecord(
                area_code=row["Area Code"],
                area=row["Area"],
                item_code=row["Item Code"],
                item=row["Item"],
                element=row["Element"],
                unit=row["Unit"],
                year=year,
                value=_parse_value(raw_value, flag=flag or ""),
                flag=flag,
            )


def load_production_zip(zip_path: str, *, item_names: set[str]) -> list[FaostatRecord]:
    """Open `zip_path` once and stream the single real production member
    CSV, returning every real record for `item_names` across every real
    country/region and year present."""
    with zipfile.ZipFile(zip_path) as zf:
        if PRODUCTION_CSV_MEMBER not in zf.namelist():
            raise FaostatParseError(f"{PRODUCTION_CSV_MEMBER!r} not found in {zip_path!r}")
        with zf.open(PRODUCTION_CSV_MEMBER) as fh:
            return list(parse_production_csv(fh, item_names=item_names))


# asyncpg hard-caps bound parameters at 32,767 per statement - real,
# live-confirmed 2026-08-25: a single real item ("Poppy seed", every
# country, every year) produced 5,760 rows x 11 columns = 63,360 params,
# a genuine `InterfaceError` on the very first real end-to-end run. This
# pipeline's other sources (BACI/Agmarknet/MSP) never hit this because
# their per-call result sets were small; FAOSTAT's "one item, every
# country, every year" shape is structurally larger. 1000 rows/batch
# (11,000 params) leaves real headroom below the 2,978-row hard ceiling.
_UPSERT_BATCH_SIZE = 1000


async def upsert_faostat_records(engine: AsyncEngine, records: list[FaostatRecord]) -> int:
    """Idempotent bulk upsert into `raw_faostat_records`, keyed on that
    table's real unique constraint `(area_code, item_code, element,
    year)` - batched to stay under asyncpg's real parameter-count limit
    (see `_UPSERT_BATCH_SIZE`)."""
    if not records:
        return 0
    fetched_at = datetime.now(UTC)
    rows = [
        {
            "fetched_at": fetched_at,
            "area_code": r.area_code,
            "area": r.area,
            "item_code": r.item_code,
            "item": r.item,
            "element": r.element,
            "unit": r.unit,
            "year": r.year,
            "value": r.value,
            "flag": r.flag,
            "raw_payload": {
                "area_code": r.area_code,
                "area": r.area,
                "item_code": r.item_code,
                "item": r.item,
                "element": r.element,
                "unit": r.unit,
                "year": r.year,
                "value": str(r.value) if r.value is not None else None,
                "flag": r.flag,
            },
        }
        for r in records
    ]
    async with engine.begin() as conn:
        for batch_start in range(0, len(rows), _UPSERT_BATCH_SIZE):
            batch = rows[batch_start : batch_start + _UPSERT_BATCH_SIZE]
            stmt = insert(raw_faostat_records).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["area_code", "item_code", "element", "year"],
                set_={
                    "fetched_at": stmt.excluded.fetched_at,
                    "area": stmt.excluded.area,
                    "item": stmt.excluded.item,
                    "unit": stmt.excluded.unit,
                    "value": stmt.excluded.value,
                    "flag": stmt.excluded.flag,
                    "raw_payload": stmt.excluded.raw_payload,
                },
            )
            await conn.execute(stmt)
    return len(rows)
