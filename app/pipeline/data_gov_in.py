"""Generic data.gov.in / api.data.gov.in resource-API client — shared
low-level HTTP mechanics for every dataset on this platform
(`app.pipeline.agmarknet`, `app.pipeline.msp`, and further Tier-1
agriculture sources), extracted after the second real dataset needed the
exact same behavior `agmarknet.py` had already found and fixed live.

Every dataset on this platform shares one real REST shape, confirmed live
across two independent resources (Agmarknet's 81M-row daily feed and the
MSP-and-cost-of-production 22-row reference table): `GET
/resource/<uuid>?api-key=...&format=json&offset=&limit=&filters[<Field>]=...`,
returning `{"field": [...], "total": int, "count": int, "records": [...]}`
on success.

Two real, live-confirmed quirks, both handled here so every dataset
module gets them for free:

- **A silent block on `httpx`'s default `User-Agent`** (found 2026-08-24,
  building `app.pipeline.agmarknet`): every request with the default
  `User-Agent: python-httpx/x.y.z` hung until timeout with *zero*
  response, while the identical request via `curl`'s default
  `User-Agent: curl/x.y.z` returned instantly. Any other real
  `User-Agent` fixes it — `_REQUEST_HEADERS` always sends an explicit,
  honest one, never this library's default and never a spoofed
  browser/curl string.
- **A real rate limit with two different observed shapes**: `HTTP 200`
  with `{"error": "Rate limit exceeded"}` (Agmarknet, a `limit=5000`
  request) *and*, separately, a real `HTTP 429` with an empty/non-JSON
  body (MSP discovery, 2026-08-24, after several rapid successive calls)
  — `fetch_page` retries on both, not just one.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

import httpx
import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

BASE_URL = "https://api.data.gov.in"

_RATE_LIMIT_RETRY_SCHEDULE_SECONDS: tuple[float, ...] = (10.0, 30.0, 60.0)
_JITTER_FRACTION = 0.2
# Real, live-confirmed necessity - see module docstring's User-Agent finding.
_REQUEST_HEADERS = {"User-Agent": "business-analyser-agentic-workflow/1.0"}


class DataGovInError(Exception):
    """Raised for any data.gov.in request/response problem."""


class DataGovInRateLimitedError(DataGovInError):
    """Raised when a real rate-limit signal (either shape) is still
    present after exhausting the retry schedule."""


async def fetch_page(
    client: httpx.AsyncClient,
    *,
    resource_path: str,
    api_key: str,
    offset: int,
    limit: int,
    filters: dict[str, str] | None = None,
    sleep_fn: Callable[[float], Awaitable[object]] = asyncio.sleep,
    random_fn: Callable[[float, float], float] = random.uniform,
) -> list[dict[str, object]]:
    """One page of real records for `resource_path` (e.g.
    `/resource/<uuid>`), retried on either real rate-limit shape."""
    params: dict[str, str] = {
        "api-key": api_key,
        "format": "json",
        "offset": str(offset),
        "limit": str(limit),
    }
    for field_name, value in (filters or {}).items():
        params[f"filters[{field_name}]"] = value

    last_error = "no attempt made"
    for attempt_index in range(len(_RATE_LIMIT_RETRY_SCHEDULE_SECONDS) + 1):
        if attempt_index > 0:
            base = _RATE_LIMIT_RETRY_SCHEDULE_SECONDS[attempt_index - 1]
            jitter = base * _JITTER_FRACTION
            await sleep_fn(base + random_fn(-jitter, jitter))

        response = await client.get(resource_path, params=params, headers=_REQUEST_HEADERS)

        if response.status_code == 429:
            last_error = "HTTP 429"
            logger.warning("data_gov_in.rate_limited_429", attempt=attempt_index)
            continue
        if response.status_code != 200:
            raise DataGovInError(f"status {response.status_code}: {response.text[:200]}")

        body: dict[str, object] = response.json()
        if "error" in body:
            last_error = str(body["error"])
            logger.warning(
                "data_gov_in.rate_limited_error_body", attempt=attempt_index, error=last_error
            )
            continue

        records = body.get("records", [])
        assert isinstance(records, list)
        return records

    raise DataGovInRateLimitedError(f"exhausted retry schedule: {last_error}")


async def fetch_all_pages(
    client: httpx.AsyncClient,
    *,
    resource_path: str,
    api_key: str,
    filters: dict[str, str] | None = None,
    page_size: int,
) -> list[dict[str, object]]:
    """Pages through every real record matching `filters`, stopping at the
    first *empty* page - not the first page shorter than the requested
    `page_size`. Real, live-confirmed bug found building
    `app.pipeline.msp` (2026-08-24): the MSP-and-cost-of-production
    resource silently caps its own effective page size to 10 regardless
    of the `limit` requested (a real `limit=25` request returned exactly
    10 records, `"limit": 10` echoed back in the response) - a
    "stop at the first short page" heuristic would have silently
    truncated every real dataset to its first page, since every page
    would appear "short" relative to the *requested* page size. Advancing
    `offset` by the page's own *actual* returned length (not the
    requested `page_size`) and stopping only once a page is genuinely
    empty is correct regardless of whether a given resource honors,
    caps, or otherwise silently alters the requested limit.

    Returns the full list rather than an async generator: every Tier-1
    dataset identified so far is a small, static reference table (tens to
    low thousands of rows), unlike Agmarknet's 81M-row feed, so holding
    one dataset's full result set in memory is a deliberate, safe
    simplification here - streaming was `agmarknet.py`'s own necessity,
    not a general requirement of this client."""
    all_records: list[dict[str, object]] = []
    offset = 0
    while True:
        page = await fetch_page(
            client,
            resource_path=resource_path,
            api_key=api_key,
            offset=offset,
            limit=page_size,
            filters=filters,
        )
        if not page:
            return all_records
        all_records.extend(page)
        offset += len(page)
