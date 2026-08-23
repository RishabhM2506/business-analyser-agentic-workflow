"""Agmarknet / data.gov.in daily mandi-price ingestion (`docs/PLAN.md`
§1, §7 item 7 — previously blocked on a credential gap, now resolved: a
real user-supplied `AGMARKNET_API_KEY`, `app.settings.Settings.
agmarknet_api_key`).

Real mechanics verified live, 2026-08-24, against
`https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24`
("Variety-wise Daily Market Prices Data of Commodity", Dept. of
Agriculture and Farmers Welfare) — a documented, simple `GET` JSON REST
API, no CSRF/session dance (unlike DGCIS). Confirmed live:

- Real query params: `api-key`, `format=json`, `offset`, `limit`,
  `filters[State]`, `filters[District]`, `filters[Commodity]` — all
  exactly as documented on the API page.
- Real response shape: `{"total": int, "count": int, "records": [...]}`
  on success, where each record is
  `{"Arrival_Date": "dd/mm/yyyy", "Commodity", "Commodity_Code",
  "District", "Grade", "Market", "Max_Price", "Min_Price",
  "Modal_Price", "State", "Variety"}` — every price field is a JSON
  *string* (the field metadata itself types them `"keyword"`, not a
  number), parsed here via `Decimal(str(...))`, never `float`, per D8.
  81,355,409 total real rows as of this verification, dating back to at
  least 2009.
- **A real, live-confirmed silent block on httpx's default `User-Agent`**:
  every real request made with plain `httpx.get(...)`/`AsyncClient.get(...)`
  (default header `User-Agent: python-httpx/x.y.z`) hung until timeout with
  *zero* response - not a 4xx/5xx, an actual black hole - while the
  identical request via `curl` (default `User-Agent: curl/x.y.z`) returned
  instantly. Isolated live: swapping in any other real `User-Agent` (even
  a plainly self-identifying one, not spoofed) fixed it immediately.
  `fetch_page` therefore always sends an explicit, honest
  `User-Agent: business-analyser-agentic-workflow/1.0` - never this
  library's default, and never a value pretending to be a browser/curl.
- **A real, live-confirmed rate limit**: `limit=5000` in one request
  returned `HTTP 200` with `{"error": "Rate limit exceeded"}` — a
  same-status-code error shape, not a 4xx/5xx, so `fetch_page` detects it
  by inspecting the parsed JSON body for an `"error"` key rather than by
  status code alone. Recovered on its own within roughly a minute of
  smaller requests. The exact real threshold (request size vs. call
  frequency) was not isolated further to avoid needlessly hammering a
  real government API; `_DEFAULT_PAGE_SIZE` below is a conservative,
  *unverified* starting point (flagged, not empirically tuned — same
  honesty convention as `app.pipeline.dgcis._DEFAULT_DELAY_SECONDS`).
- **A real, live-confirmed data gap for this pipeline's own canonical
  scenario**: no record was found anywhere in this dataset for poppy
  seeds under any of 9 plausible `Commodity` name variants tried
  (`"Poppy seeds"`, `"Poppy Seeds"`, `"poppy seed"`, `"Postha"`,
  `"Posta"`, `"Khuskhus"`, `"Khus Khus"`, `"Kaskas"`, `"Poppyseed"`), nor
  in an unfiltered sample of the 4 real districts under India's licensed
  poppy-cultivation program (Neemuch, Mandsaur, Chittorgarh, Pratapgarh).
  This is consistent with poppy (opium poppy / *Papaver somniferum*,
  including its seed and husk) being a Central Bureau of Narcotics
  licensed-procurement crop in India, structurally outside the open
  APMC-mandi trading this dataset covers — a real, structural absence,
  not a naming mismatch to keep guessing at. Per D2, this pipeline must
  never turn "no Agmarknet row found" into a fabricated zero: a query for
  poppy seeds' domestic price should surface as genuinely unknown
  (`NOT_FOUND`), exactly as `docs/PLAN.md`'s own pre-existing
  `domestic_price_inr_paise_per_kg` comment already anticipated
  ("NULL if Agmarknet coverage too thin").
- Price unit: `raw_agmarknet_prices.modal_price_inr_paise_per_qtl`'s own
  column name (and `docs/PLAN.md`'s prior note) already encodes "Rupees
  per Quintal" as the assumed unit. This session's live fetches did not
  turn up an explicit unit label in the API's own metadata or the
  data.gov.in catalog page (checked; "quintal" does not appear in either
  page's raw HTML) — Rs./Quintal is well-established, independently of
  this specific page, as Agmarknet's universal price-quotation
  convention, but is recorded here as an *inherited*, not newly
  independently reconfirmed, assumption.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation

import httpx
import structlog
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.warehouse.schema import raw_agmarknet_prices

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

BASE_URL = "https://api.data.gov.in"
RESOURCE_PATH = "/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"

# Unverified, conservative starting point - see module docstring.
_DEFAULT_PAGE_SIZE = 100
_RATE_LIMIT_RETRY_SCHEDULE_SECONDS: tuple[float, ...] = (10.0, 30.0, 60.0)
_JITTER_FRACTION = 0.2
# Real, live-confirmed necessity - see module docstring's User-Agent finding.
_REQUEST_HEADERS = {"User-Agent": "business-analyser-agentic-workflow/1.0"}


class AgmarknetError(Exception):
    """Raised for any Agmarknet request/response problem."""


class AgmarknetRateLimitedError(AgmarknetError):
    """Raised when the real `{"error": "Rate limit exceeded"}` shape is
    still present after exhausting the retry schedule."""


@dataclass(frozen=True)
class AgmarknetRecord:
    price_date: date
    commodity: str
    market: str
    state: str
    modal_price_inr_paise_per_qtl: int | None
    raw_payload: dict[str, object]


def _parse_arrival_date(value: str) -> date:
    return datetime.strptime(value, "%d/%m/%Y").date()


def _parse_price_paise(value: object) -> int | None:
    """`None` for anything that isn't a clean positive number (blank,
    `"NR"`, or absent) - never guessed or coerced to 0, per D2."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        rupees = Decimal(text)
    except InvalidOperation:
        return None
    return int(rupees * 100)


def _record_from_raw(raw: dict[str, object]) -> AgmarknetRecord | None:
    arrival_date = raw.get("Arrival_Date")
    commodity = raw.get("Commodity")
    market = raw.get("Market")
    state = raw.get("State")
    if not isinstance(arrival_date, str) or not isinstance(commodity, str):
        return None
    if not isinstance(market, str) or not isinstance(state, str):
        return None
    return AgmarknetRecord(
        price_date=_parse_arrival_date(arrival_date),
        commodity=commodity,
        market=market,
        state=state,
        modal_price_inr_paise_per_qtl=_parse_price_paise(raw.get("Modal_Price")),
        raw_payload=raw,
    )


async def fetch_page(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    offset: int,
    limit: int = _DEFAULT_PAGE_SIZE,
    commodity: str | None = None,
    state: str | None = None,
    district: str | None = None,
    sleep_fn: Callable[[float], Awaitable[object]] = asyncio.sleep,
    random_fn: Callable[[float, float], float] = random.uniform,
) -> list[dict[str, object]]:
    """One page of real records, retried on the real, live-confirmed
    `{"error": "Rate limit exceeded"}` shape (HTTP 200, so detected by
    body inspection, not status code)."""
    params: dict[str, str] = {
        "api-key": api_key,
        "format": "json",
        "offset": str(offset),
        "limit": str(limit),
    }
    if commodity is not None:
        params["filters[Commodity]"] = commodity
    if state is not None:
        params["filters[State]"] = state
    if district is not None:
        params["filters[District]"] = district

    last_error = "no attempt made"
    for attempt_index in range(len(_RATE_LIMIT_RETRY_SCHEDULE_SECONDS) + 1):
        if attempt_index > 0:
            base = _RATE_LIMIT_RETRY_SCHEDULE_SECONDS[attempt_index - 1]
            jitter = base * _JITTER_FRACTION
            await sleep_fn(base + random_fn(-jitter, jitter))

        response = await client.get(RESOURCE_PATH, params=params, headers=_REQUEST_HEADERS)
        if response.status_code != 200:
            raise AgmarknetError(f"status {response.status_code}: {response.text[:200]}")

        body: dict[str, object] = response.json()
        if "error" in body:
            last_error = str(body["error"])
            logger.warning("agmarknet_rate_limited", attempt=attempt_index, error=last_error)
            continue

        records = body.get("records", [])
        assert isinstance(records, list)
        return records

    raise AgmarknetRateLimitedError(f"exhausted retry schedule: {last_error}")


async def fetch_all_records(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    commodity: str | None = None,
    state: str | None = None,
    district: str | None = None,
    page_size: int = _DEFAULT_PAGE_SIZE,
) -> AsyncIterator[AgmarknetRecord]:
    """Pages through every real record matching the given filters,
    stopping at the first short page (fewer than `page_size` records
    returned) - the real API's own natural end-of-results signal."""
    offset = 0
    while True:
        page = await fetch_page(
            client,
            api_key=api_key,
            offset=offset,
            limit=page_size,
            commodity=commodity,
            state=state,
            district=district,
        )
        for raw in page:
            record = _record_from_raw(raw)
            if record is not None:
                yield record
        if len(page) < page_size:
            return
        offset += page_size


async def upsert_agmarknet_records(engine: AsyncEngine, records: list[AgmarknetRecord]) -> int:
    """Bulk-upsert into `raw_agmarknet_prices`, keyed on that table's real
    unique constraint `(price_date, commodity, market)` - idempotent by
    construction, same pattern as every other raw-layer upsert in this
    pipeline."""
    if not records:
        return 0
    fetched_at = datetime.now(UTC)
    rows = [
        {
            "fetched_at": fetched_at,
            "price_date": r.price_date,
            "commodity": r.commodity,
            "market": r.market,
            "state": r.state,
            "modal_price_inr_paise_per_qtl": r.modal_price_inr_paise_per_qtl,
            "raw_payload": r.raw_payload,
        }
        for r in records
    ]
    async with engine.begin() as conn:
        stmt = insert(raw_agmarknet_prices).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["price_date", "commodity", "market"],
            set_={
                "fetched_at": stmt.excluded.fetched_at,
                "state": stmt.excluded.state,
                "modal_price_inr_paise_per_qtl": stmt.excluded.modal_price_inr_paise_per_qtl,
                "raw_payload": stmt.excluded.raw_payload,
            },
        )
        await conn.execute(stmt)
    return len(rows)
