"""Typed UN Comtrade API client (docs/PLAN.md §4.1).

Responsibilities: bounded timeout, retry with jitter, a circuit breaker, and
a fixture record/replay mode so `tests/integration/` never makes a live
network call (docs/PLAN.md §7). Read-only, external (docs/PLAN.md §1.1) —
this is the only tool v1 has.

# TODO(Phase 3): implement the real HTTP calls (httpx.AsyncClient), the
# tenacity-based bounded+jittered retry policy, a circuit breaker, and the
# fixture record/replay mode, against the verified UN Comtrade API shape.
"""

from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict


class ComtradeRecord(BaseModel):
    """Normalized shape of a single UN Comtrade trade-flow record.

    This is our own client's normalized output, not a verbatim copy of the
    upstream UN Comtrade response schema — the exact upstream field mapping
    (`reporterCode`, `partnerCode`, `cifvalue`/`fobvalue`, `isReported`, ...)
    is unverified against live API responses (per the "never fabricate"
    rule) and is deferred to Phase 3 implementation against real responses.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hs_code: str
    flow: Literal["import", "export"]
    partner_code: str
    partner_country: str
    year: int
    value: float | None
    is_provisional: bool


class ComtradeClientError(Exception):
    """Base exception for all `ComtradeClient` failures (timeout, upstream
    error, circuit open)."""


class ComtradeClient:
    """Read-only client for the UN Comtrade API (comtradeapi.un.org).

    Owns: request timeout, bounded retry with jitter, a circuit breaker, and
    (in `tests/integration/`) fixture-based record/replay.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://comtradeapi.un.org",
        timeout_seconds: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = httpx.Timeout(timeout_seconds)

    async def fetch_flow(
        self,
        *,
        hs_code: str,
        flow: Literal["import", "export"],
        year_start: int,
        year_end: int,
    ) -> list[ComtradeRecord]:
        """Fetch one trade-flow direction for an HS6 code across a year range."""
        raise NotImplementedError  # TODO(Phase 3): implement against the verified Comtrade API.
