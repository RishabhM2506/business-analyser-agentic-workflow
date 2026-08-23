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

**Not yet built**: the ~250-country loop, rate limiting, normalization
into `raw_dgcis_monthly`/`ref_hs6_hs8_crosswalk`, and the monthly
national-total path (`meidb/commoditywise_import`) for D15. This module
is the client + parser slice; the loop/normalizer is the next unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import httpx
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://tradestat.commerce.gov.in"
ANNUAL_IMPORT_PATH = "/eidb/commodityx_countries_wise_import"
ANNUAL_EXPORT_PATH = "/eidb/commodityx_countries_wise_export"

_TOKEN_FIELD = "_token"
_SESSION_EXPIRED_STATUS = 419


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
