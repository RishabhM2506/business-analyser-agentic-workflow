"""DGCIS Tradestat client + parser (`docs/PLAN.md` §1, §7 — build sequence
item 4). Real mechanics verified live, 2026-08-23, against
`tradestat.commerce.gov.in`, not assumed from the site's own labels or
report names (several of which turned out to be misleading — see the
module-level notes below and `docs/PLAN.md` §1 for the full investigation).

**No documented API.** A Laravel application, form-driven:
`GET` a report page to obtain a CSRF token (`<input name="_token">`) and a
session cookie (`indiatrade-session`, 15-minute `Max-Age`), then `POST`
back to the *same URL* with both, within that window. A session that has
expired returns HTTP 419 ("Page Expired") — `DgcisClient` retries once
with a fresh GET/token pair on exactly that status, never silently on
other failures.

**Report used for the annual, per-partner-country series**:
`eidb/commodityx_countries_wise_import` / `_export`. Chosen after
exhaustively checking every other candidate live (`docs/PLAN.md` §1) —
most report *names* on this site are misleading about what they actually
return (e.g. `commodityx_countries_wise_*`'s own country field is
`required` with no "all countries" option, same as reports whose names
don't suggest a country dimension at all). This report returns a full
**5-year annual series in one response** for one (country, HS8) pair —
verified live for HS `12079100` (poppy seeds) x Turkey, real fixture
committed at `tests/fixtures/dgcis/poppy_seed_turkey_import_annual.html`.

Real, verified field names (differ from the *visible* "Enter HS Code"
input, which belongs to an unrelated lookup-assist modal — a second real
trap already found and documented in `docs/PLAN.md` §1):
`searchTerm` (HS8, up to 8 digits), `ContEidbi`/`ContEidbe` (country code,
import/export respectively), `ContEidbyi`/`ContEidbey` (year, e.g. "2024"
for FY2024-25), `ReportEidbi`/`ReportEidbe` (`"1"` = ₹ Crore, `"2"` =
US $ Million).

`fetch_all_countries_annual` loops over every country in
`data/dgcis-country-codes.csv` (251 real codes, captured live 2026-08-23 —
`get_dgcis_countries`, same `lru_cache`d CSV-loading pattern as
`app.knowledge.provider._load_taxonomy`) for one (hs8, flow, year),
pacing requests with a real delay — this portal has no documented rate
limit, so `_DEFAULT_DELAY_SECONDS` is a conservative, unverified starting
point (flagged, not empirically tuned, matching this project's honesty
convention for other untuned thresholds). `upsert_annual_records` writes
parsed records into `raw_dgcis_annual` (`docs/PLAN.md` §4).

**Monthly national-total path** (`meidb/commoditywise_import`/`_export`,
D15's real data source), real mechanics verified live, 2026-08-23:
different real form field names per flow, and — unlike the annual
report's `ContEidbi`/`ContEidbe` symmetry — **not** a simple prefix swap:
import uses `imdd`-prefixed fields (`imddMonth`, `imddYear`,
`imddCommodityLevel`, `imddReportVal`, `imddReportYear`), export uses
bare `dd`-prefixed fields (`ddMonth`, ...) — checked both pages directly
rather than assumed. `comlev`/`comval` (HS code, "specific" scope) are
shared field names on both pages. Genuinely **month-granular**: one
request returns exactly one calendar month's value alongside the *same*
month one year earlier (for the site's own YoY display) — not a multi-
year series like the annual report, so a full year needs one request per
month. **Value and quantity require two separate real requests**
(`imddReportVal`/`ddReportVal` = `"3"` for ₹ Crore, `"2"` for quantity in
the source's own native unit) — no single request returns both,
confirmed live (quantity *is* real and available here, unlike the annual
report, which never returns a quantity at all).

**A real, load-bearing revision-status marker** appears in the response's
own current-month column header, e.g. `"Aug-2026 (A)"` — verified across
three real requests spanning a genuinely unpublished month, a recent
published-but-flagged month, and an older fully finalized month:
`"(R)"` = Revised/Final, `"(F)"` = Flash/provisional (subject to later
revision), `"(A)"` = Advance — the month hasn't been published yet
(confirmed live: for a real `"(A)"`-marked month, *both* the specific
commodity's value **and** the "India's Total Import" national-total
footer row read `0.00` — the whole month's collection genuinely hasn't
happened, not a coincidental real zero for one product). This marker is
preserved verbatim in `raw_dgcis_monthly.raw_payload` for the normalizer
to translate into a D1 status value — never interpreted at parse time,
matching this module's own raw/normalized separation for the annual path.

`ref_hs6_hs8_crosswalk` population (derived from these same responses —
the crosswalk needs `hs6`, which requires a real taxonomy join) is still
not wired in here.
"""

from __future__ import annotations

import asyncio
import csv
import re
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

import httpx
import structlog
from bs4 import BeautifulSoup, Tag
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.warehouse.schema import raw_dgcis_annual, raw_dgcis_monthly

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

BASE_URL = "https://tradestat.commerce.gov.in"
ANNUAL_IMPORT_PATH = "/eidb/commodityx_countries_wise_import"
ANNUAL_EXPORT_PATH = "/eidb/commodityx_countries_wise_export"
MONTHLY_IMPORT_PATH = "/meidb/commoditywise_import"
MONTHLY_EXPORT_PATH = "/meidb/commoditywise_export"
_ALL_PARTNERS = (
    "ALL_PARTNERS"  # §4's documented sentinel - this report has no partner dimension at all
)

_TOKEN_FIELD = "_token"
_SESSION_EXPIRED_STATUS = 419
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_COUNTRY_CODES_PATH = "data/dgcis-country-codes.csv"
# No documented rate limit on this portal - conservative and unverified,
# flagged for empirical tuning once a real ~250-request run has been
# observed for real throttling/blocking behavior.
_DEFAULT_DELAY_SECONDS = 1.0


class DgcisRequestError(Exception):
    """Raised for any DGCIS request/response problem — never silently
    returns a guessed or partial result."""


@dataclass(frozen=True)
class DgcisAnnualRecord:
    """One (country, HS8) pair's real annual series, as returned by
    `commodityx_countries_wise_import`/`_export` in a single response.
    `values_by_year["2020 - 2021"]` etc — keyed on DGCIS's own year label
    (a fiscal-year range string), not yet normalized to a calendar month
    (that's the normalizer's job, not the parser's)."""

    country: str
    hs8: str
    description: str
    unit: str
    report_date: str
    value_type: str
    values_by_year: dict[str, Decimal | None]


def _parse_decimal(text: str) -> Decimal | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def parse_annual_country_response(html: str) -> DgcisAnnualRecord | None:
    """Parse a `commodityx_countries_wise_import`/`_export` response into
    one `DgcisAnnualRecord`. Returns `None` if the expected table isn't
    present — the "genuinely no data for this (country, HS8)" response
    shape hasn't been observed/verified live yet (flagged in
    `docs/PLAN.md` §1), so this is deliberately permissive rather than
    guessing what that looks like; callers should treat `None` as
    `NOT_REPORTED`-worthy but log it for later confirmation.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="example")
    if table is None or not isinstance(table, Tag):
        return None

    header_cell = table.find("td")
    if header_cell is None:
        return None
    header_text = header_cell.get_text(" ", strip=True)

    country_span = header_cell.find("span")
    country = country_span.get_text(strip=True) if country_span else ""

    # HSCODE/Description/Unit/Report Date/value-type all live in bold tags
    # inside this same header cell, in document order - extracted by label
    # rather than position, since DGCIS has already shown (docs/PLAN.md §1)
    # that visual layout isn't a reliable contract.
    hs8 = _text_after_label(header_text, "HSCODE:")
    description = _text_after_label(header_text, "Description:")
    unit = _text_after_label(header_text, "Unit:")
    report_date = _text_after_label(header_text, "Report Date:")
    value_type = header_text.split("Values in", 1)[-1].strip() if "Values in" in header_text else ""

    rows = table.find_all("tr")
    year_header_row = rows[1] if len(rows) > 1 else None
    if year_header_row is None:
        return None
    year_labels = [th.get_text(strip=True) for th in year_header_row.find_all("th")[2:]]

    # The context header row (rows[0]) also contains the substring "Values
    # in" (from its own "...Values in ₹ Crore" trailer) - live-reproduced:
    # naively searching the whole row's text for "Values in" matched that
    # header row instead of the real data row. The real data row's own
    # *label cell* (its 2nd <td>, after S.No.) is what must start with
    # "Values in" - only rows[2:] can be real data rows at all (rows[0] is
    # the header, rows[1] the <th> year header).
    values_row = next(
        (
            r
            for r in rows[2:]
            if len(r.find_all("td")) >= 2
            and r.find_all("td")[1].get_text(strip=True).startswith("Values in")
        ),
        None,
    )
    if values_row is None:
        return None
    value_cells = values_row.find_all("td")[2:]
    values_by_year = {
        year: _parse_decimal(cell.get_text())
        for year, cell in zip(year_labels, value_cells, strict=False)
    }

    return DgcisAnnualRecord(
        country=country,
        hs8=hs8,
        description=description,
        unit=unit,
        report_date=report_date,
        value_type=value_type,
        values_by_year=values_by_year,
    )


_LABEL_BOUNDARIES = ("HSCODE:", "Description:", "Unit:", "Report Date:", "||", "Values in")


def _text_after_label(text: str, label: str) -> str:
    if label not in text:
        return ""
    after = text.split(label, 1)[1].strip()
    # Every label field is immediately followed by the next label (or the
    # "|| Values in ..." trailer, for Report Date specifically) - split on
    # the next known boundary marker.
    for boundary in _LABEL_BOUNDARIES:
        if boundary != label and boundary in after:
            after = after.split(boundary, 1)[0]
    return after.strip()


@dataclass(frozen=True)
class DgcisMonthlyCell:
    """One real month-year column's parsed value from a
    `commoditywise_import`/`_export` response — `value` in whatever unit
    the request asked for (₹ Crore or the source's own native quantity
    unit), `marker` DGCIS's own real revision-status letter (`"R"`/`"F"`/
    `"A"`, verbatim — never interpreted here)."""

    value: Decimal | None
    marker: str
    unit: str | None


_MONTH_HEADER_PATTERN = re.compile(r"^([A-Za-z]{3})-(\d{4})\s*(?:\(([A-Za-z])\))?")


def parse_monthly_response(html: str, *, hs8: str) -> DgcisMonthlyCell | None:
    """Parse one real `commoditywise_import`/`_export` response, returning
    the row matching `hs8`'s *current*-month cell — the later of the
    response's two real month-year header columns (the site always shows
    the requested month alongside the same month one year earlier for its
    own YoY display; the current one is verified live to always be the
    second of the two). The current-month header's column index is found
    by its own text, never a hardcoded position — a quantity-flavored
    request (`report_value="2"`) inserts a real extra `UNIT` header a
    value-flavored one (`report_value="3"`) doesn't have, so a fixed
    index would silently misalign between the two real response shapes.
    `None` if the table or the matching row isn't present."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="example1")
    if table is None or not isinstance(table, Tag):
        return None

    thead = table.find("thead")
    if thead is None or not isinstance(thead, Tag):
        return None
    header_texts = [cell.get_text(" ", strip=True) for cell in thead.find_all("th")]

    has_unit_column = any(text.upper() == "UNIT" for text in header_texts)

    # A real month-year header ("Jun-2022 (R)") matches from the start of
    # the cell text; the report's own YTD columns ("Jan-Jun2022 (R)")
    # never match this pattern (no hyphen right after "Jan"), so this
    # never confuses the two even though both mention the same year.
    month_header_indices = [
        i for i, text in enumerate(header_texts) if _MONTH_HEADER_PATTERN.match(text)
    ]
    if len(month_header_indices) != 2:
        return None
    current_month_index = month_header_indices[1]
    marker_match = _MONTH_HEADER_PATTERN.match(header_texts[current_month_index])
    marker = (marker_match.group(3) or "").upper() if marker_match else ""

    tbody = table.find("tbody")
    if tbody is None or not isinstance(tbody, Tag):
        return None
    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) <= current_month_index or len(cells) < 2:
            continue
        if cells[1].get_text(strip=True) != hs8:
            continue
        value = _parse_decimal(cells[current_month_index].get_text())
        unit = cells[3].get_text(strip=True) if has_unit_column and len(cells) > 3 else None
        return DgcisMonthlyCell(value=value, marker=marker, unit=unit)
    return None


class DgcisClient:
    """Session/CSRF mechanics only (`docs/PLAN.md` §1) - real GET-then-POST
    round trip, with exactly one retry on a `419` (session expired)."""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(base_url=BASE_URL, transport=transport, timeout=30.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get_token(self, path: str) -> str:
        try:
            response = await self._client.get(path)
        except httpx.TransportError as exc:
            raise DgcisRequestError(f"GET {path} failed: {exc}") from exc
        if response.status_code != 200:
            raise DgcisRequestError(f"GET {path} returned status {response.status_code}")
        soup = BeautifulSoup(response.text, "lxml")
        token_input = soup.find("input", attrs={"name": _TOKEN_FIELD})
        if token_input is None or not isinstance(token_input, Tag):
            raise DgcisRequestError(f"GET {path}: no CSRF token found in response")
        token = token_input.get("value")
        if not isinstance(token, str) or not token:
            raise DgcisRequestError(f"GET {path}: CSRF token input had no value")
        return token

    async def fetch_annual(
        self, *, path: str, hs8: str, country_code: str, year: str, report_value: str = "1"
    ) -> str:
        """One real GET+POST round trip for the annual per-country report.
        `path` is `ANNUAL_IMPORT_PATH` or `ANNUAL_EXPORT_PATH`; field names
        differ between them (`ContEidbi`/`ContEidbyi` for import,
        `ContEidbe`/`ContEidbey` for export) - the caller selects via
        `path`, this method picks the matching field names."""
        is_import = path == ANNUAL_IMPORT_PATH
        country_field = "ContEidbi" if is_import else "ContEidbe"
        year_field = "ContEidbyi" if is_import else "ContEidbey"
        report_field = "ReportEidbi" if is_import else "ReportEidbe"

        for attempt in range(2):
            token = await self._get_token(path)
            try:
                response = await self._client.post(
                    path,
                    data={
                        _TOKEN_FIELD: token,
                        "searchTerm": hs8,
                        country_field: country_code,
                        year_field: year,
                        report_field: report_value,
                    },
                    headers={"Referer": f"{BASE_URL}{path}"},
                )
            except httpx.TransportError as exc:
                raise DgcisRequestError(f"POST {path} failed: {exc}") from exc

            if response.status_code == _SESSION_EXPIRED_STATUS and attempt == 0:
                continue
            if response.status_code != 200:
                raise DgcisRequestError(f"POST {path} returned status {response.status_code}")
            return response.text

        raise DgcisRequestError(f"POST {path}: session expired twice in a row")

    async def fetch_monthly(
        self, *, path: str, hs8: str, month: int, year: int, report_value: str
    ) -> str:
        """One real GET+POST round trip for the monthly national-total
        report (D15). `path` is `MONTHLY_IMPORT_PATH` or
        `MONTHLY_EXPORT_PATH`; field names differ between them — real,
        verified live: import uses `imdd`-prefixed fields, export uses
        bare `dd`-prefixed ones, not a simple symmetric swap. `report_value`
        `"3"`=₹ Crore, `"2"`=quantity — the caller makes two separate
        calls to get both (no single request returns both, verified
        live). Report framing is always fixed to Calendar Year
        (`...ReportYear="2"`), matching D15's own calendar-month
        discipline, never DGCIS's fiscal-year framing."""
        is_import = path == MONTHLY_IMPORT_PATH
        prefix = "imdd" if is_import else "dd"

        for attempt in range(2):
            token = await self._get_token(path)
            try:
                response = await self._client.post(
                    path,
                    data={
                        _TOKEN_FIELD: token,
                        "comlev": "specific",
                        "comval": hs8,
                        f"{prefix}CommodityLevel": "8",
                        f"{prefix}Month": str(month),
                        f"{prefix}Year": str(year),
                        f"{prefix}ReportVal": report_value,
                        f"{prefix}ReportYear": "2",
                    },
                    headers={"Referer": f"{BASE_URL}{path}"},
                )
            except httpx.TransportError as exc:
                raise DgcisRequestError(f"POST {path} failed: {exc}") from exc

            if response.status_code == _SESSION_EXPIRED_STATUS and attempt == 0:
                continue
            if response.status_code != 200:
                raise DgcisRequestError(f"POST {path} returned status {response.status_code}")
            return response.text

        raise DgcisRequestError(f"POST {path}: session expired twice in a row")


@dataclass(frozen=True)
class DgcisCountry:
    """One row of the checked-in DGCIS country-code reference
    (`data/dgcis-country-codes.csv`) — DGCIS's own code/name pair, real,
    captured live 2026-08-23. Not yet cross-walked to a UN M49 code
    (`ref_country_crosswalk`, `docs/PLAN.md` §4) - that only matters at the
    normalized layer; the raw layer stores DGCIS's own identity verbatim
    (D7)."""

    code: str
    name: str


def _resolve_path(csv_path: str) -> Path:
    path = Path(csv_path)
    return path if path.is_absolute() else _REPO_ROOT / path


@lru_cache(maxsize=2)
def _load_countries(resolved_path: Path) -> list[DgcisCountry]:
    with resolved_path.open(encoding="utf-8", newline="") as fh:
        return [
            DgcisCountry(code=row["dgcis_country_code"], name=row["dgcis_country_name"])
            for row in csv.DictReader(fh)
        ]


def get_dgcis_countries(*, path: str = _DEFAULT_COUNTRY_CODES_PATH) -> list[DgcisCountry]:
    """Every real DGCIS country code/name pair from the checked-in
    reference CSV — matches `app.knowledge.provider.get_hs6_taxonomy_entries`'s
    exact `lru_cache`d-loader pattern."""
    return _load_countries(_resolve_path(path))


@dataclass(frozen=True)
class DgcisFetchFailure:
    """One country's fetch attempt failed - the caller decides whether to
    dead-letter it; this module never silently drops a failure."""

    country: DgcisCountry
    error: str


async def fetch_all_countries_annual(
    client: DgcisClient,
    *,
    path: str,
    hs8: str,
    year: str,
    countries: Iterable[DgcisCountry] | None = None,
    delay_seconds: float = _DEFAULT_DELAY_SECONDS,
) -> AsyncIterator[DgcisAnnualRecord | DgcisFetchFailure]:
    """Loop over every tracked country for one (hs8, flow, year), pacing
    requests with `delay_seconds` between them. Yields a `DgcisAnnualRecord`
    per country that returned real data, or a `DgcisFetchFailure` for one
    that didn't — never raises for a single country's failure (that would
    abort the whole ~250-country run over one bad response), matching
    `docs/PLAN.md` §7's "job continues to next country... rather than
    aborting the batch" contract. A response with no matching table
    (`parse_annual_country_response` returns `None`) is *not* treated as a
    failure — it's a country DGCIS genuinely has nothing to report for,
    logged but not dead-lettered (the "no data" response shape is still
    unverified live, `docs/PLAN.md` §1 - flagged, not guessed)."""
    country_list = list(countries) if countries is not None else get_dgcis_countries()
    for index, country in enumerate(country_list):
        if index > 0:
            await asyncio.sleep(delay_seconds)
        try:
            html = await client.fetch_annual(
                path=path, hs8=hs8, country_code=country.code, year=year
            )
        except DgcisRequestError as exc:
            logger.warning(
                "dgcis.annual_fetch_failed", country=country.name, hs8=hs8, error=str(exc)
            )
            yield DgcisFetchFailure(country=country, error=str(exc))
            continue

        record = parse_annual_country_response(html)
        if record is None:
            logger.info("dgcis.annual_no_table_in_response", country=country.name, hs8=hs8)
            continue
        yield record


def _row_from_record(
    record: DgcisAnnualRecord, *, flow: str, scraped_at: datetime
) -> dict[str, object]:
    return {
        "scraped_at": scraped_at,
        "fiscal_year_label": None,  # set per-year by the caller, see upsert_annual_records
        "hs8": record.hs8,
        "flow": flow,
        "partner_country": record.country,
        "description": record.description,
        "unit": record.unit,
        "value_inr_paise": None,
        "raw_payload": {
            "country": record.country,
            "hs8": record.hs8,
            "description": record.description,
            "unit": record.unit,
            "report_date": record.report_date,
            "value_type": record.value_type,
            "values_by_year": {
                k: (str(v) if v is not None else None) for k, v in record.values_by_year.items()
            },
        },
    }


async def upsert_annual_records(
    engine: AsyncEngine, records: Iterable[DgcisAnnualRecord], *, flow: str
) -> int:
    """Bulk-upsert every year in every record's `values_by_year` into
    `raw_dgcis_annual` — one row per (fiscal_year_label, hs8, flow,
    partner_country), matching that table's real unique key (`docs/PLAN.md`
    §4). Re-running with the same records is idempotent by construction
    (`ON CONFLICT ... DO UPDATE`, keyed on the same real unique constraint
    the table enforces) - never a second, duplicate row for a re-scraped
    combination. `value_inr_paise` is stored in **paise**, converted here
    once from DGCIS's ₹-Crore-denominated figures (1 crore = 10,000,000
    rupees = 1,000,000,000 paise) - money is never a float (D8): the
    conversion multiplies the parsed `Decimal` by an exact integer, then
    rounds to the nearest paisa once, at the boundary into storage.
    """
    scraped_at = datetime.now(UTC)
    rows: list[dict[str, object]] = []
    for record in records:
        base = _row_from_record(record, flow=flow, scraped_at=scraped_at)
        for year_label, value_crore in record.values_by_year.items():
            row = dict(base)
            row["fiscal_year_label"] = year_label
            row["value_inr_paise"] = (
                int(value_crore * Decimal(1_000_000_000)) if value_crore is not None else None
            )
            rows.append(row)

    if not rows:
        return 0

    async with engine.begin() as conn:
        stmt = insert(raw_dgcis_annual).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["fiscal_year_label", "hs8", "flow", "partner_country"],
            set_={
                "scraped_at": stmt.excluded.scraped_at,
                "description": stmt.excluded.description,
                "unit": stmt.excluded.unit,
                "value_inr_paise": stmt.excluded.value_inr_paise,
                "raw_payload": stmt.excluded.raw_payload,
            },
        )
        await conn.execute(stmt)
    return len(rows)


@dataclass(frozen=True)
class DgcisMonthlyRecord:
    """One (hs8, flow, calendar_month) cell from the real monthly
    national-total report (D15) — raw-layer, mirrors DGCIS's own observed
    shape verbatim (D7): `value_inr_paise`/`quantity_kg`/`unit` as
    directly parsed, plus DGCIS's own real revision-status `marker`
    (`"R"`/`"F"`/`"A"`, verbatim) preserved for the normalizer to
    translate into a D1 status value — never interpreted here, matching
    how this module's annual-report path also leaves status derivation
    entirely to `app.pipeline.normalize`."""

    hs8: str
    flow: str
    calendar_month: date
    value_inr_paise: int | None
    quantity_kg: Decimal | None
    unit: str | None
    marker: str


async def fetch_monthly_record(
    client: DgcisClient, *, path: str, hs8: str, month: int, year: int
) -> DgcisMonthlyRecord | None:
    """Two real POST calls (₹ Crore, then quantity) combined into one
    record — DGCIS's monthly report has no single request returning both
    (verified live). `None` if the value-flavored response has no row for
    `hs8` at all; a genuinely missing quantity-flavored response (rare,
    not yet observed live) leaves `quantity_kg`/`unit` `None` rather than
    failing the whole record — a real value with a missing quantity is a
    `QTY_MISSING` case for the normalizer to assign, not a fetch failure."""
    value_html = await client.fetch_monthly(
        path=path, hs8=hs8, month=month, year=year, report_value="3"
    )
    value_cell = parse_monthly_response(value_html, hs8=hs8)
    if value_cell is None:
        return None

    quantity_html = await client.fetch_monthly(
        path=path, hs8=hs8, month=month, year=year, report_value="2"
    )
    quantity_cell = parse_monthly_response(quantity_html, hs8=hs8)

    value_inr_paise = (
        int(value_cell.value * Decimal(1_000_000_000)) if value_cell.value is not None else None
    )
    is_import = path == MONTHLY_IMPORT_PATH
    return DgcisMonthlyRecord(
        hs8=hs8,
        flow="import" if is_import else "export",
        calendar_month=date(year, month, 1),
        value_inr_paise=value_inr_paise,
        quantity_kg=quantity_cell.value if quantity_cell is not None else None,
        unit=quantity_cell.unit if quantity_cell is not None else None,
        marker=value_cell.marker,
    )


@dataclass(frozen=True)
class DgcisMonthlyFetchFailure:
    """One month's fetch attempt failed — the caller decides whether to
    dead-letter it; this module never silently drops a failure."""

    month: int
    year: int
    error: str


async def fetch_year_monthly(
    client: DgcisClient,
    *,
    path: str,
    hs8: str,
    year: int,
    months: Iterable[int] | None = None,
    delay_seconds: float = _DEFAULT_DELAY_SECONDS,
) -> AsyncIterator[DgcisMonthlyRecord | DgcisMonthlyFetchFailure]:
    """Loop over every requested month (default: all 12) for one
    `(hs8, flow, year)`, pacing requests with `delay_seconds` between
    them — two real POST calls per month, never one (see
    `fetch_monthly_record`). Never raises for one month's failure,
    matching `fetch_all_countries_annual`'s identical "continue the
    batch" contract."""
    month_list = list(months) if months is not None else list(range(1, 13))
    for index, month in enumerate(month_list):
        if index > 0:
            await asyncio.sleep(delay_seconds)
        try:
            record = await fetch_monthly_record(client, path=path, hs8=hs8, month=month, year=year)
        except DgcisRequestError as exc:
            logger.warning(
                "dgcis.monthly_fetch_failed", month=month, year=year, hs8=hs8, error=str(exc)
            )
            yield DgcisMonthlyFetchFailure(month=month, year=year, error=str(exc))
            continue
        if record is None:
            logger.info("dgcis.monthly_no_row_in_response", month=month, year=year, hs8=hs8)
            continue
        yield record


def _fiscal_year_label_for_month(calendar_month: date) -> str:
    """India's fiscal year runs April-March — a calendar month in
    Jan/Feb/Mar belongs to the fiscal year that *started* the previous
    calendar year. Matches the annual report's own real label format
    (`"2020 - 2021"`)."""
    fy_start_year = calendar_month.year if calendar_month.month >= 4 else calendar_month.year - 1
    return f"{fy_start_year} - {fy_start_year + 1}"


async def upsert_monthly_records(engine: AsyncEngine, records: Iterable[DgcisMonthlyRecord]) -> int:
    """Bulk-upsert into `raw_dgcis_monthly` — `partner_country` is always
    the `'ALL_PARTNERS'` sentinel (`docs/PLAN.md` §4's own documented
    policy: this report structurally has no partner-country dimension at
    all, not a scraping gap). DGCIS's own revision-status `marker` is
    preserved in `raw_payload`, not translated to a D1 status here."""
    scraped_at = datetime.now(UTC)
    rows = [
        {
            "scraped_at": scraped_at,
            "fiscal_year": _fiscal_year_label_for_month(r.calendar_month),
            "calendar_month": r.calendar_month,
            "hs8": r.hs8,
            "flow": r.flow,
            "partner_country": _ALL_PARTNERS,
            "value_inr_paise": r.value_inr_paise,
            "quantity": r.quantity_kg,
            "unit": r.unit,
            "raw_payload": {"marker": r.marker},
        }
        for r in records
    ]
    if not rows:
        return 0

    async with engine.begin() as conn:
        stmt = insert(raw_dgcis_monthly).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["fiscal_year", "calendar_month", "hs8", "flow", "partner_country"],
            set_={
                "scraped_at": stmt.excluded.scraped_at,
                "value_inr_paise": stmt.excluded.value_inr_paise,
                "quantity": stmt.excluded.quantity,
                "unit": stmt.excluded.unit,
                "raw_payload": stmt.excluded.raw_payload,
            },
        )
        await conn.execute(stmt)
    return len(rows)
