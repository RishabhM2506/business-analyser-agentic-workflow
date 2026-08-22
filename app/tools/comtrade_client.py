"""Typed UN Comtrade API client (docs/PLAN.md §4.1).

Responsibilities: bounded timeout, retry with jitter, a circuit breaker, and
a fixture record/replay mode so `tests/integration/` never makes a live
network call (docs/PLAN.md §7). Read-only, external (docs/PLAN.md §1.1) —
this is the only tool v1 has.

Verified live against the real UN Comtrade API on 2026-08-16 (no API key
exists yet — see `docs/OPEN-QUESTIONS.md` — so verification used the
no-auth "preview" endpoint, which returns the same record shape):

    GET https://comtradeapi.un.org/public/v1/preview/C/A/HS
        ?reporterCode=699&period=2022&cmdCode=010121&flowCode=M

    {"elapsedTime": "...", "count": 3, "error": "", "data": [
      {"...", "reporterCode": 699, "flowCode": "M", "partnerCode": 0,
       "cmdCode": "010121", "period": "2022", "primaryValue": 1992455.942,
       "cifvalue": 1992455.942, "fobvalue": null, "isReported": false,
       "isAggregate": true, "legacyEstimationFlag": 0, ...},
      {"...", "partnerCode": 826, "primaryValue": 1861448.346,
       "isReported": true, "isAggregate": false, ...},
      {"...", "partnerCode": 36, "primaryValue": 131007.596,
       "isReported": true, "isAggregate": false, ...}
    ]}

Confirms the field list `docs/PHASE0-FINDINGS.md` carried over from earlier
research (`reporterCode`, `partnerCode`, `cmdCode`, `flowCode`, `period`,
`primaryValue`/`cifvalue`/`fobvalue`, `isReported`, `isAggregate`,
`legacyEstimationFlag`) is still accurate. Two things learned only from this
second, Phase-3 round of live verification and NOT present in Phase 0's
notes:

1. Omitting `partnerCode` returns the full partner breakdown for that
   (reporter, period, cmdCode, flowCode) in ONE call — including the
   `partnerCode=0` ("World") aggregate row alongside every reporting
   partner. A live query returned 258 partner rows for one HS6/year/flow.
   `fetch_flow` below relies on this: one HTTP call per year, not one call
   per (year, partner).
2. A comma-separated multi-year `period` (e.g. `"2019,2020,2021"`) was
   rejected with HTTP 400 on this endpoint in a live test, even though it is
   documented as valid on the general Comtrade API surface — so this client
   loops one year per call rather than requesting a year range in one shot.
3. The preview endpoint rate-limits aggressively: a live probe hit HTTP 429
   after only two rapid successive calls. This is direct, empirical
   justification for the circuit breaker below, not a hypothetical risk.
4. Trade-flow records carry `partnerDesc: null` on every row (not populated
   on this path) — human-readable partner names are resolved locally
   instead, from a checked-in reference table (see `resolve_partner_country`
   below), fetched live the same day from
   `comtradeapi.un.org/files/v1/app/reference/partnerAreas.json`.

**Authenticated path, verified live (2026-08-20, real registered key)**: the
authenticated production path (`AUTHENTICATED_PATH` below,
`Ocp-Apim-Subscription-Key` header) was previously unverified — real key
now confirmed: `GET /data/v1/get/C/A/HS?...` with a real subscription key
returned `200 OK`, 154 real records, schema-identical to the already-
verified preview shape (`typeCode`/`reporterCode`/`flowCode`/`partnerCode`/
`cmdCode`/`primaryValue`/`isReported`/`isAggregate`, etc.) — no schema
changes needed. `ComtradeClient` prefers this path whenever a key is
configured (`__init__`), with a one-time sticky fallback to the free
preview tier in `_get_once` if the configured key turns out invalid
(401/403) — handles a placeholder/never-registered key gracefully instead
of failing every request. This also gets real production traffic off the
free/unauthenticated preview tier's aggressive shared rate limiting (429s
observed repeatedly during development) onto the registered key's own
documented 500-calls/day quota.

Reporter is fixed to India (`INDIA_REPORTER_CODE`, Comtrade code 699) — this
whole project is an India-focused trade-data assistant throughout
`docs/PHASE0-FINDINGS.md` and `docs/PLAN.md` (DGCIS/tradestat framing, ITC-HS
comparisons, Indian-fiscal-year-vs-calendar-year caveat), and `TradeQuery`
(the frozen v1 schema, `app/schemas/query.py`) deliberately has no
reporter/country field — reporter is a fixed constant, not a per-request
parameter.
"""

from __future__ import annotations

import csv
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)

# --- Verified upstream constants (see module docstring for evidence) -------
INDIA_REPORTER_CODE = "699"
DEFAULT_BASE_URL = "https://comtradeapi.un.org"
PREVIEW_PATH = "/public/v1/preview/C/A/HS"
AUTHENTICATED_PATH = "/data/v1/get/C/A/HS"  # unverified live — see module docstring
_SUBSCRIPTION_KEY_HEADER = "Ocp-Apim-Subscription-Key"

_PARTNER_AREAS_CSV = Path(__file__).resolve().parents[2] / "data" / "comtrade-partner-areas.csv"

_FLOW_CODES: dict[Literal["import", "export"], str] = {"import": "M", "export": "X"}


class TradeDataProvider(Protocol):
    """Minimal interface every trade-value data source must satisfy —
    deliberately narrow so swapping the source (a paid registered-key
    Comtrade tier, a different official statistics provider, or eventually
    a shipment-level commercial provider) is a new adapter, not a new
    interface. Mirrors `app.models.ModelClient`'s Protocol shape/rationale
    exactly — same pattern, same reason: node code (`app/nodes/fetch_trade.py`)
    depends on this Protocol, never on `ComtradeClient` concretely.

    Named and scoped per the project's 2026-08-20 roadmap decision
    (`docs/PLAN.md` "Trade-data-source flexibility"): UN Comtrade remains
    the sole real implementation for now — this Protocol exists so a future
    provider swap touches one new adapter file, not every call site."""

    async def fetch_flow(
        self,
        *,
        hs_code: str,
        flow: Literal["import", "export"],
        year_start: int,
        year_end: int,
    ) -> list[ComtradeRecord]: ...


class ComtradeRecord(BaseModel):
    """Normalized shape of a single UN Comtrade trade-flow record.

    This is our own client's normalized output, not a verbatim copy of the
    upstream UN Comtrade response schema (that's `_RawComtradeRecord`,
    below) — the exact upstream field mapping is now verified live (see
    module docstring); `aggregate.py` consumes this normalized shape, not
    upstream field names.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hs_code: str
    flow: Literal["import", "export"]
    partner_code: str
    partner_country: str
    year: int
    value: float | None
    is_provisional: bool


class _RawComtradeRecord(BaseModel):
    """Verbatim (subset of) the upstream UN Comtrade record shape, verified
    live 2026-08-16 (see module docstring). `extra="ignore"`: upstream has
    ~40 fields total; we only need the ones that feed `ComtradeRecord` plus
    the completeness flags, and upstream is free to add fields over time
    without breaking us."""

    model_config = ConfigDict(extra="ignore")

    reporterCode: int
    partnerCode: int
    period: str
    cmdCode: str
    flowCode: str
    primaryValue: float | None = None
    cifvalue: float | None = None
    fobvalue: float | None = None
    isReported: bool
    isAggregate: bool


class _ComtradeEnvelope(BaseModel):
    """Verbatim top-level response envelope, verified live 2026-08-16:
    `{"elapsedTime": "...", "count": N, "data": [...], "error": ""}`."""

    model_config = ConfigDict(extra="ignore")

    count: int
    data: list[_RawComtradeRecord]
    error: str = ""


class ComtradeClientError(Exception):
    """Base exception for all `ComtradeClient` failures (timeout, upstream
    error, circuit open)."""


class ComtradeTimeoutError(ComtradeClientError):
    """The request did not complete within `timeout_seconds`, even after
    bounded retries. Maps to `ErrorResponse(error_code="UPSTREAM_TIMEOUT")`."""


class ComtradeUpstreamError(ComtradeClientError):
    """Upstream returned a non-2xx response, or a 2xx envelope with a
    populated `error` field.

    `retryable` distinguishes the two cases that share this exception type
    but must NOT share retry behavior: a 429/5xx (or an `error`-field
    response) is transient and worth retrying, but a 4xx other than 429
    means *our own request* was malformed — retrying it verbatim cannot
    succeed and would just burn retry budget (master brief §7.9: "bounded
    retries only"). This is a plain `bool` attribute rather than two
    exception classes so the retry predicate can inspect it without a second
    `isinstance` branch, and so both cases still map to the same
    `ErrorResponse(error_code="UPSTREAM_...")` family at the node layer.
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ComtradeSchemaValidationError(ComtradeClientError):
    """The response body was not valid JSON, or did not match the verified
    `_ComtradeEnvelope`/`_RawComtradeRecord` schema."""


class ComtradeCircuitOpenError(ComtradeClientError):
    """The circuit breaker is open (too many recent consecutive failures) —
    failing fast without making a network call."""


@lru_cache(maxsize=1)
def _load_partner_names() -> dict[str, str]:
    """Load the checked-in partner-code -> country/area-name reference
    table (`data/comtrade-partner-areas.csv`), needed because live trade-flow
    responses carry `partnerDesc: null` on every record (verified live,
    module docstring point 4) — vendored the same way, and the same day, as
    the HS taxonomy CSV (`docs/PHASE0-FINDINGS.md` §3's "static, versioned,
    checked-in dataset" pattern), from
    `comtradeapi.un.org/files/v1/app/reference/partnerAreas.json`.
    """
    names: dict[str, str] = {}
    with _PARTNER_AREAS_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            names[row["partner_code"]] = row["partner_desc"]
    return names


def resolve_partner_country(partner_code: str) -> str:
    """Best-effort `partner_code` -> human-readable name lookup. Never
    raises: an unmapped code (e.g. a future reference-table addition we
    haven't refreshed yet) falls back to a labeled placeholder instead of
    crashing the fetch — missing metadata is not the same failure class as
    missing trade data, and must not take down the whole request."""
    name = _load_partner_names().get(partner_code)
    if name is None:
        logger.warning("comtrade.unknown_partner_code partner_code=%s", partner_code)
        return f"Unknown partner ({partner_code})"
    return name


def is_aggregate_partner_code(partner_code: str) -> bool:
    """True iff `partner_code` is a non-country catch-all ("World", an ",
    nes" regional bucket, "Bunkers", "Free Zones", "Special Categories" —
    `docs/PHASE0-FINDINGS.md` §4/§5, `docs/PLAN.md` §6) that must be
    stripped before top-10 partner-country ranking. Data-driven from the
    same checked-in reference table, not a hand-maintained denylist prone to
    drift — see `data/comtrade-partner-areas.csv`'s `is_aggregate_code`
    column: true iff the code is `0` ("World"), `837`/`838`/`839`
    ("Bunkers"/"Free Zones"/"Special Categories"), or the reference name
    ends in ", nes" ("not elsewhere specified" — a regional catch-all
    bucket, e.g. "Other Asia, nes")."""
    names = _load_partner_names()
    if partner_code not in names:
        # Unknown to our reference table: treat conservatively as NOT an
        # aggregate (i.e. keep it in ranking) rather than silently dropping
        # a real, if newly-added, partner country.
        return False
    return _load_partner_aggregate_flags().get(partner_code, False)


@lru_cache(maxsize=1)
def _load_partner_aggregate_flags() -> dict[str, bool]:
    flags: dict[str, bool] = {}
    with _PARTNER_AREAS_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            flags[row["partner_code"]] = row["is_aggregate_code"].strip().lower() == "true"
    return flags


class _CircuitBreaker:
    """Minimal closed -> open -> half-open circuit breaker, scoped to one
    `ComtradeClient` instance (and, via the process-wide singleton in
    `get_comtrade_client`, effectively scoped to the whole running process —
    the correct scope: it protects the app from hammering a struggling
    upstream on behalf of any request, not just one).

    Closed: calls proceed normally. After `failure_threshold` consecutive
    failures: opens, and every call fails fast (`ComtradeCircuitOpenError`,
    no network call) until `reset_timeout_seconds` has elapsed. Then:
    half-open — the next call is allowed through as a trial; success closes
    the circuit, failure re-opens it (with a fresh cooldown window).
    """

    def __init__(self, *, failure_threshold: int = 5, reset_timeout_seconds: float = 30.0) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout_seconds = reset_timeout_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def before_call(self) -> None:
        if self._opened_at is None:
            return
        if (time.monotonic() - self._opened_at) >= self._reset_timeout_seconds:
            return  # half-open: let one trial call through
        raise ComtradeCircuitOpenError(
            f"UN Comtrade circuit open after {self._consecutive_failures} consecutive "
            f"failures; retry after a {self._reset_timeout_seconds}s cooldown"
        )

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._opened_at = time.monotonic()

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None


class ComtradeClient:
    """Read-only client for the UN Comtrade API (comtradeapi.un.org).

    Owns: request timeout, bounded retry with jitter, a circuit breaker, and
    (via the `transport` seam) fixture-based record/replay for tests —
    `tests/integration/test_comtrade_client.py` injects `httpx.MockTransport`
    so the exact same request-building/parsing/validation code path runs in
    tests as in production, without ever touching the network.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
        # Raised from 3 (2026-08-20, live-reproduced): a real 5-year/2-flow
        # analysis fires up to 10 sequential-per-flow requests, and a real
        # UN Comtrade key was observed rate-limiting one specific (year,
        # flow) call with three consecutive 429s while every sibling call in
        # the same batch recovered within 1-2 retries — 3 attempts is
        # sometimes not enough headroom against a short-lived rate-limit
        # burst that everything else in the same request rides out fine.
        max_attempts: int = 5,
        circuit_breaker: _CircuitBreaker | None = None,
    ) -> None:
        self._api_key = api_key
        self._max_attempts = max_attempts
        self._circuit_breaker = circuit_breaker or _CircuitBreaker()
        # Finding (2026-08-20, real registered key provided for the first
        # time): AUTHENTICATED_PATH was defined but never actually used —
        # every call went to PREVIEW_PATH regardless of whether a real key
        # was configured, so a genuine subscription key sat completely
        # inert. Now verified live against the real endpoint with a real
        # key (200 OK, 154 real records, same response schema as the
        # already-verified preview shape — no schema changes needed).
        # Prefer the authenticated path whenever any key is configured;
        # `_get_once` falls back to the free preview tier, once and
        # stickily, the first time the authenticated path rejects the key
        # (401/403) — handles a placeholder/never-registered key (a fresh
        # clone's `.env.example` default, or CI's placeholder string)
        # gracefully instead of failing every single request.
        self._use_authenticated_path = bool(api_key)
        # Sent on every request, including preview-endpoint ones (harmless
        # no-op there — confirmed live: preview calls succeed with zero auth
        # headers) so the one code path already carries the header the
        # authenticated endpoint requires, "threaded through for when a real
        # key exists" per the brief.
        headers = {_SUBSCRIPTION_KEY_HEADER: api_key} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
            headers=headers,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ComtradeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def fetch_flow(
        self,
        *,
        hs_code: str,
        flow: Literal["import", "export"],
        year_start: int,
        year_end: int,
    ) -> list[ComtradeRecord]:
        """Fetch one trade-flow direction for an HS6 code across an
        inclusive year range — India as reporter, every reporting partner
        country/area for each year (docs/PLAN.md §2.2's `fetch_imports`/
        `fetch_exports` nodes call this once each, in parallel).

        One HTTP request per year (module docstring points 1-2): omitting
        `partnerCode` returns the full partner breakdown for that year in a
        single call, but a multi-year `period` was rejected live — so this
        loops years, not partners.
        """
        if year_start > year_end:
            raise ValueError(f"year_start ({year_start}) must be <= year_end ({year_end})")
        flow_code = _FLOW_CODES[flow]
        records: list[ComtradeRecord] = []
        for year in range(year_start, year_end + 1):
            records.extend(
                await self._fetch_year(hs_code=hs_code, flow=flow, flow_code=flow_code, year=year)
            )
        return records

    async def _fetch_year(
        self, *, hs_code: str, flow: Literal["import", "export"], flow_code: str, year: int
    ) -> list[ComtradeRecord]:
        self._circuit_breaker.before_call()
        params = {
            "reporterCode": INDIA_REPORTER_CODE,
            "period": str(year),
            "cmdCode": hs_code,
            "flowCode": flow_code,
        }
        try:
            envelope = await self._get_with_retry(params)
        except ComtradeClientError:
            self._circuit_breaker.record_failure()
            raise
        self._circuit_breaker.record_success()
        return [self._normalize(raw, hs_code=hs_code, flow=flow) for raw in envelope.data]

    async def _get_with_retry(self, params: dict[str, str]) -> _ComtradeEnvelope:
        """Bounded retry with jitter (master brief §7.9: "bounded retries
        only... cap them"). Retries only timeouts and retryable upstream
        statuses (429/5xx) — a 4xx other than 429 means our own request is
        malformed and retrying it verbatim cannot help, so it is raised
        straight through instead of burning retry budget."""

        def _is_retryable(exc: BaseException) -> bool:
            if isinstance(exc, ComtradeTimeoutError):
                return True
            if isinstance(exc, ComtradeUpstreamError):
                return exc.retryable
            return False

        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=0.5, max=8.0),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        result: _ComtradeEnvelope = await retrying(self._get_once, params)
        return result

    async def _get_once(self, params: dict[str, str]) -> _ComtradeEnvelope:
        path = AUTHENTICATED_PATH if self._use_authenticated_path else PREVIEW_PATH
        try:
            response = await self._client.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise ComtradeTimeoutError(f"UN Comtrade request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise ComtradeUpstreamError(f"UN Comtrade transport error: {exc}") from exc

        if self._use_authenticated_path and response.status_code in (401, 403):
            # Sticky, one-time fallback (see __init__'s comment): this key
            # doesn't work against the authenticated endpoint — retrying the
            # identical call against the identical path can only ever fail
            # the same way, so switch to the free preview tier for the rest
            # of this client's lifetime rather than rejecting every request
            # a misconfigured/placeholder key would otherwise cause.
            logger.warning(
                "comtrade.authenticated_path_rejected status_code=%s — "
                "falling back to the unauthenticated preview endpoint",
                response.status_code,
            )
            self._use_authenticated_path = False
            try:
                response = await self._client.get(PREVIEW_PATH, params=params)
            except httpx.TimeoutException as exc:
                raise ComtradeTimeoutError(f"UN Comtrade request timed out: {exc}") from exc
            except httpx.TransportError as exc:
                raise ComtradeUpstreamError(f"UN Comtrade transport error: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise ComtradeUpstreamError(
                f"UN Comtrade returned retryable status {response.status_code}",
                status_code=response.status_code,
                retryable=True,
            )
        if response.status_code >= 400:
            raise ComtradeUpstreamError(
                f"UN Comtrade returned client error {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
                retryable=False,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ComtradeSchemaValidationError(
                f"UN Comtrade response was not valid JSON: {exc}"
            ) from exc
        try:
            envelope = _ComtradeEnvelope.model_validate(payload)
        except ValidationError as exc:
            raise ComtradeSchemaValidationError(
                f"UN Comtrade response failed schema validation: {exc}"
            ) from exc
        if envelope.error:
            # A populated `error` field on an otherwise-2xx envelope is an
            # upstream-reported condition of unknown permanence (never
            # observed on a live call during verification — see
            # tests/fixtures/comtrade/README.md); treated as retryable, same
            # as a 5xx, still bounded by max_attempts either way.
            raise ComtradeUpstreamError(
                f"UN Comtrade reported an error: {envelope.error}", retryable=True
            )
        return envelope

    @staticmethod
    def _normalize(
        raw: _RawComtradeRecord, *, hs_code: str, flow: Literal["import", "export"]
    ) -> ComtradeRecord:
        # primaryValue is populated on every verified live record regardless
        # of flow direction; cifvalue/fobvalue are the documented
        # flow-specific fallbacks if a future response ever omits it.
        value = raw.primaryValue
        if value is None:
            value = raw.cifvalue if flow == "import" else raw.fobvalue
        return ComtradeRecord(
            hs_code=hs_code,
            flow=flow,
            partner_code=str(raw.partnerCode),
            partner_country=resolve_partner_country(str(raw.partnerCode)),
            year=int(raw.period),
            value=value,
            is_provisional=not raw.isReported,
        )


@lru_cache(maxsize=1)
def get_comtrade_client() -> ComtradeClient:
    """Process-wide singleton (mirrors `app.settings.get_settings()`'s
    pattern) — one pooled `httpx.AsyncClient` and one shared circuit breaker
    reused across requests within this instance (docs/PLAN.md §1.2: v1 runs
    one instance, no cross-replica cache/breaker state needed).

    Tests never call this directly for real network behavior: they either
    construct `ComtradeClient(transport=httpx.MockTransport(...))` directly,
    or monkeypatch this factory where it's imported in `app.nodes.fetch_trade`.
    """
    from app.settings import get_settings

    settings = get_settings()
    return ComtradeClient(api_key=settings.comtrade_api_key)
