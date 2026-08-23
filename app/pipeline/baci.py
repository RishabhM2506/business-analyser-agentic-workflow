"""BACI (CEPII) annual bulk-ZIP ingestion (`docs/PLAN.md` §7, build
sequence item 6) — powers D9's check C (`report/mismatch.py`, not yet
implemented there).

**Real, verified format** (2026-08-23, vintage `V202601`, HS22 revision —
the current/most-detailed revision offered, and the smallest download at
~301MB compressed; confirmed live via the real download's own
`Readme.txt`, not assumed from BACI's general docs). One ZIP per
`(hs_revision, vintage)`, containing:
- One CSV per year, columns `t` (year), `i` (exporter code), `j`
  (importer code), `k` (HS6 product code), `v` (value, **thousand USD**),
  `q` (quantity, **metric tons**) — both converted to this pipeline's own
  units (whole USD, kg) at parse time, never stored in BACI's native
  units.
- `product_codes_HS<rev>_V<vintage>.csv` and `country_codes_V<vintage>.csv`
  (not consumed by this module — HS6 codes and country codes here already
  match this pipeline's own conventions, verified next).

**India's BACI code is `699`** — the *same* code Comtrade uses (verified
live against the real downloaded `country_codes_V202601.csv`; an earlier,
unverified guess in this project's own planning notes assumed a
different UN M49 numeric scheme — corrected here by checking, not
assumed).

**A real, flagged coverage gap, not a bug**: the HS22-revision file only
contains years **2022-2024** (HS22 became the active nomenclature that
year) — it does **not** cover 2020-2021, part of this pipeline's
canonical 2020-2024 window. Covering the earlier years would need the
HS17-revision file too (~795MB, confirmed live, not downloaded or parsed
by this module) — a real, open follow-up, not silently absorbed into a
narrower window without saying so.

**Streamed directly from the ZIP member**, never fully extracted to disk
and never loaded whole into memory — each year's CSV is real, ~11 million
rows / ~360MB uncompressed (confirmed live against the actual file).
Filtered while streaming to rows involving India (`exporter_code` or
`importer_code` `== INDIA_CODE`) for the caller's tracked HS6 codes —
the only rows this pipeline has any use for, out of BACI's full
global-bilateral-trade scope.
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

from app.warehouse.schema import raw_baci_records

INDIA_CODE = "699"  # same code Comtrade uses - verified live, not assumed
_THOUSAND = Decimal(1000)


class BaciParseError(Exception):
    """Raised for a real structural problem with the ZIP/CSV — never
    silently skips an entire file."""


@dataclass(frozen=True)
class BaciRecord:
    vintage: str
    hs_revision: str
    year: int
    exporter_code: str
    importer_code: str
    hs6: str
    value_fob_usd: Decimal | None
    quantity_kg: Decimal | None


def _parse_decimal(raw: str) -> Decimal | None:
    """`None` for a genuinely blank cell — never `0` (D2's "ZERO vs
    missing" discipline, extended to this source too, even though no
    blank cell was observed in the real downloaded file — defensive, not
    assumed absent forever)."""
    text = raw.strip()
    if not text or text.upper() == "NA":
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_baci_year_csv(
    fh: IO[bytes], *, vintage: str, hs_revision: str, year: int, hs6_codes: set[str]
) -> Iterator[BaciRecord]:
    """Stream one year's real BACI CSV (`fh`, opened from the ZIP member,
    never read whole into memory), yielding only rows for `hs6_codes`
    where India is the exporter or the importer. `v`/`q` are converted
    from BACI's native thousand-USD/metric-ton units to whole USD/kg —
    the only unit conversion this parser performs; everything else is
    copied verbatim."""
    reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", newline=""))
    required = {"t", "i", "j", "k", "v", "q"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise BaciParseError(
            f"BACI CSV for {hs_revision}/{vintage}/{year} is missing expected columns "
            f"{sorted(required)}; found {reader.fieldnames}"
        )
    for row in reader:
        if row["k"] not in hs6_codes:
            continue
        if row["i"] != INDIA_CODE and row["j"] != INDIA_CODE:
            continue
        value_thousand_usd = _parse_decimal(row["v"])
        quantity_tonnes = _parse_decimal(row["q"])
        yield BaciRecord(
            vintage=vintage,
            hs_revision=hs_revision,
            year=year,
            exporter_code=row["i"],
            importer_code=row["j"],
            hs6=row["k"],
            value_fob_usd=(
                value_thousand_usd * _THOUSAND if value_thousand_usd is not None else None
            ),
            quantity_kg=quantity_tonnes * _THOUSAND if quantity_tonnes is not None else None,
        )


def load_baci_zip(
    zip_path: str, *, vintage: str, hs_revision: str, years: list[int], hs6_codes: set[str]
) -> list[BaciRecord]:
    """Open `zip_path` once, stream every requested year's member CSV in
    turn (each individually, never all held in memory at once), and
    return every real India-involving record for `hs6_codes`. The
    per-year member filename (`BACI_HS<rev>_Y<year>_V<vintage>.csv`) is
    BACI's own real, verified naming convention."""
    records: list[BaciRecord] = []
    with zipfile.ZipFile(zip_path) as zf:
        for year in years:
            member = f"BACI_HS{hs_revision}_Y{year}_V{vintage}.csv"
            if member not in zf.namelist():
                raise BaciParseError(f"{member!r} not found in {zip_path!r}")
            with zf.open(member) as fh:
                records.extend(
                    parse_baci_year_csv(
                        fh, vintage=vintage, hs_revision=hs_revision, year=year, hs6_codes=hs6_codes
                    )
                )
    return records


async def upsert_baci_records(engine: AsyncEngine, records: list[BaciRecord]) -> int:
    """Idempotent bulk upsert into `raw_baci_records`, keyed on
    `(vintage, year, exporter_code, importer_code, hs6)` — the table's own
    real unique key (`docs/PLAN.md` §4). A vintage is immutable once
    loaded (§7's own stated ingestion contract): re-running with the same
    vintage's records updates the same rows in place, never duplicates."""
    if not records:
        return 0
    loaded_at = datetime.now(UTC)
    rows = [
        {
            "loaded_at": loaded_at,
            "vintage": r.vintage,
            "hs_revision": r.hs_revision,
            "year": r.year,
            "exporter_code": r.exporter_code,
            "importer_code": r.importer_code,
            "hs6": r.hs6,
            "value_fob_usd": r.value_fob_usd,
            "quantity_kg": r.quantity_kg,
        }
        for r in records
    ]
    async with engine.begin() as conn:
        stmt = insert(raw_baci_records).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["vintage", "year", "exporter_code", "importer_code", "hs6"],
            set_={
                "value_fob_usd": stmt.excluded.value_fob_usd,
                "quantity_kg": stmt.excluded.quantity_kg,
            },
        )
        await conn.execute(stmt)
    return len(rows)
