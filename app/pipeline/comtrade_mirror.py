"""UN Comtrade mirror ingestion (`docs/PLAN.md` §7, §8, D5/D6) — bulk,
batched pulls into `raw_comtrade_records`, for the mismatch checks (D9),
never India's own series (`docs/PLAN.md`: "mirror + benchmark only").

Distinct from `app.tools.comtrade_client.ComtradeClient`: that one serves
the existing, separate single-code interactive lookup feature, tuned with
tight exponential backoff suited to a request path. This is a background
job with D6's own fixed retry schedule and a 429-specific circuit
breaker — a different real contract for a different real caller.

D5's bulk-batching fully verified live (`docs/PLAN.md` §1/§8, 2026-08-23,
using the real Comtrade key): comma-joined `period`, `flowCode=M,X`,
omitted `reporterCode`/`partnerCode`, and `cmdCode` itself all batch
correctly in one call, individually and combined. The two query shapes
below are exactly what was verified, not a guess.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

import httpx
import structlog
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.warehouse.schema import raw_comtrade_records

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

BASE_URL = "https://comtradeapi.un.org"
DATA_PATH = "/data/v1/get/C/A/HS"
INDIA_CODE = "699"

# D6: fixed schedule, not exponential — distinct from ComtradeClient's
# wait_exponential_jitter, deliberate for a background job with its own
# real-world-verified worst-case latency budget.
RETRY_SCHEDULE_SECONDS: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0)
_JITTER_FRACTION = 0.2
# D6: "circuit breaker that pauses the worker 15 minutes after 3
# consecutive 429s" — specifically 429s, not any failure (distinct from
# app.tools.comtrade_client._CircuitBreaker's generic any-failure trigger,
# which suits that module's different, tighter-loop use case).
CIRCUIT_FAILURE_THRESHOLD = 3
CIRCUIT_RESET_SECONDS = 15 * 60.0


class ComtradeMirrorError(Exception):
    """Raised for any Comtrade mirror request/response problem."""


class ComtradeMirrorCircuitOpenError(ComtradeMirrorError):
    """Raised by `_MirrorCircuitBreaker.before_call` when the circuit is
    open — no network call is made."""


class _MirrorCircuitBreaker:
    """Closed -> open -> half-open, scoped to one job run (or, via a
    shared instance, the whole process). Opens specifically on
    `CIRCUIT_FAILURE_THRESHOLD` *consecutive 429s* — a genuine rate-limit
    signal, not conflated with a transient 5xx or transport error."""

    def __init__(
        self,
        *,
        threshold: int = CIRCUIT_FAILURE_THRESHOLD,
        reset_seconds: float = CIRCUIT_RESET_SECONDS,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._reset_seconds = reset_seconds
        self._now_fn = now_fn
        self._consecutive_429s = 0
        self._opened_at: float | None = None

    def before_call(self) -> None:
        if self._opened_at is None:
            return
        if (self._now_fn() - self._opened_at) >= self._reset_seconds:
            return  # half-open: let the next call through as a trial
        raise ComtradeMirrorCircuitOpenError(
            f"Comtrade mirror circuit open after {self._consecutive_429s} consecutive 429s; "
            f"retry after a {self._reset_seconds}s cooldown"
        )

    def record_429(self) -> None:
        self._consecutive_429s += 1
        if self._consecutive_429s >= self._threshold:
            self._opened_at = self._now_fn()

    def record_non_429(self) -> None:
        self._consecutive_429s = 0
        self._opened_at = None

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None


def build_query_params(
    *,
    role: Literal["reporter", "partner"],
    cmd_codes: list[str],
    periods: list[str],
    flow_codes: Sequence[str] = ("M", "X"),
) -> dict[str, str]:
    """Query 1 (`role='reporter'`, India as reporter, feeds check A) or
    Query 2 (`role='partner'`, India as partner, feeds check B) —
    `docs/PLAN.md` §8's two verified shapes. The non-India side
    (`partnerCode`/`reporterCode` respectively) is deliberately omitted,
    not set to an explicit "all" value — verified live that omission is
    what returns every reporter/partner in one call.

    `partner2Code=0`, `motCode=0`, and `customsCode=C00` are always pinned
    — a real bug found live, resolved iteratively: Comtrade rows carry
    *three* extra breakdown dimensions (second/consignment partner, mode
    of transport, customs procedure) that `raw_comtrade_records`' unique
    key doesn't track. Leaving any one unconstrained returned genuine
    duplicate `(period, reporter, partner, flow, cmd)` combinations
    differing only in that dimension, which Postgres's
    `ON CONFLICT DO UPDATE` correctly refuses to upsert twice in one
    statement. Pinning all three to their "not broken down further"
    aggregate value (verified live, one dimension at a time, until zero
    duplicate keys remained in a real response) constrains the response
    to exactly this pipeline's needed granularity at the source, rather
    than filtering duplicates out client-side."""
    params = {
        "cmdCode": ",".join(cmd_codes),
        "period": ",".join(periods),
        "flowCode": ",".join(flow_codes),
        "partner2Code": "0",
        "motCode": "0",
        "customsCode": "C00",
    }
    if role == "reporter":
        params["reporterCode"] = INDIA_CODE
    else:
        params["partnerCode"] = INDIA_CODE
    return params


async def fetch_with_retry(
    client: httpx.AsyncClient,
    *,
    params: dict[str, str],
    api_key: str,
    breaker: _MirrorCircuitBreaker,
    sleep_fn: Callable[[float], Awaitable[object]] = asyncio.sleep,
    random_fn: Callable[[float, float], float] = random.uniform,
) -> dict[str, object]:
    """One Comtrade call, retried on D6's fixed schedule with ±20% jitter.
    A `Retry-After` response header always overrides the schedule's own
    delay for the *next* attempt (D6). Only 429/5xx/transport errors are
    retried — any other non-200 status raises immediately, matching
    `ComtradeClient`'s existing "a 4xx other than 429 means our own
    request is malformed" reasoning."""
    last_error: str = "no attempt made"
    next_delay: float | None = None

    for attempt_index in range(len(RETRY_SCHEDULE_SECONDS) + 1):
        if attempt_index > 0:
            delay = (
                next_delay if next_delay is not None else RETRY_SCHEDULE_SECONDS[attempt_index - 1]
            )
            await sleep_fn(delay)
        next_delay = None

        breaker.before_call()
        try:
            response = await client.get(
                DATA_PATH, params=params, headers={"Ocp-Apim-Subscription-Key": api_key}
            )
        except httpx.TransportError as exc:
            last_error = f"transport error: {exc}"
            if attempt_index < len(RETRY_SCHEDULE_SECONDS):
                base = RETRY_SCHEDULE_SECONDS[attempt_index]
                jitter = base * _JITTER_FRACTION
                next_delay = base + random_fn(-jitter, jitter)
            continue

        if response.status_code == 429:
            breaker.record_429()
            last_error = "429 Too Many Requests"
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    next_delay = float(retry_after)
                except ValueError:
                    next_delay = None
            if next_delay is None and attempt_index < len(RETRY_SCHEDULE_SECONDS):
                base = RETRY_SCHEDULE_SECONDS[attempt_index]
                jitter = base * _JITTER_FRACTION
                next_delay = base + random_fn(-jitter, jitter)
            continue

        if response.status_code >= 500:
            breaker.record_non_429()
            last_error = f"status {response.status_code}"
            if attempt_index < len(RETRY_SCHEDULE_SECONDS):
                base = RETRY_SCHEDULE_SECONDS[attempt_index]
                jitter = base * _JITTER_FRACTION
                next_delay = base + random_fn(-jitter, jitter)
            continue

        if response.status_code != 200:
            raise ComtradeMirrorError(
                f"non-retryable status {response.status_code}: {response.text[:200]}"
            )

        breaker.record_non_429()
        result: dict[str, object] = response.json()
        return result

    raise ComtradeMirrorError(f"exhausted retry schedule: {last_error}")


def _row_from_record(raw: dict[str, object], *, fetched_at: datetime) -> dict[str, object]:
    def _decimal_or_none(value: object) -> Decimal | None:
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except Exception:
            return None

    return {
        "fetched_at": fetched_at,
        "period": int(str(raw["period"])),
        "reporter_code": str(raw["reporterCode"]),
        "partner_code": str(raw["partnerCode"]),
        "flow_code": str(raw["flowCode"]),
        "cmd_code": str(raw["cmdCode"]),
        "primary_value_usd": _decimal_or_none(raw.get("primaryValue")),
        "net_weight_kg": _decimal_or_none(raw.get("netWgt")),
        "is_reported": bool(raw.get("isReported", True)),
        "raw_payload": raw,
    }


async def upsert_comtrade_records(engine: AsyncEngine, records: list[dict[str, object]]) -> int:
    """Bulk-upsert real Comtrade rows into `raw_comtrade_records`, keyed
    on that table's real unique constraint
    `(period, reporter_code, partner_code, flow_code, cmd_code)` —
    idempotent by construction, same pattern as
    `app.pipeline.dgcis.upsert_annual_records`."""
    if not records:
        return 0
    fetched_at = datetime.now(UTC)
    rows = [_row_from_record(r, fetched_at=fetched_at) for r in records]

    async with engine.begin() as conn:
        stmt = insert(raw_comtrade_records).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["period", "reporter_code", "partner_code", "flow_code", "cmd_code"],
            set_={
                "fetched_at": stmt.excluded.fetched_at,
                "primary_value_usd": stmt.excluded.primary_value_usd,
                "net_weight_kg": stmt.excluded.net_weight_kg,
                "is_reported": stmt.excluded.is_reported,
                "raw_payload": stmt.excluded.raw_payload,
            },
        )
        await conn.execute(stmt)
    return len(rows)
