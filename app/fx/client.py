"""Frankfurter-backed FX client (`docs/PLAN.md` §1, §6).

Live-verified during Step 1 planning, not assumed from docs:
- `GET /v2/rate/USD/INR` -> `{"date": "...", "base": "USD", "quote": "INR", "rate": <float>}`.
- Historical: the `date` is a **query parameter** (`?date=YYYY-MM-DD`), not a path segment —
  `/v2/{date}/rate/USD/INR` returns 404.
- Every calendar date (including weekends/holidays) returned a distinct rate with no gap — see
  `docs/PLAN.md` §1 for the full verification and its consequence for `app.fx.cache`.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Protocol

import httpx

FRANKFURTER_BASE_URL = "https://api.frankfurter.dev/v2"
_DEFAULT_TIMEOUT_SECONDS = 10.0


class FxRateFetchError(Exception):
    """Raised when a rate could not be fetched — network failure, non-200 response, or a response
    that doesn't match the verified shape. Callers (`app.fx.cache.FxCache`) are responsible for the
    stale-fallback behavior (`docs/PLAN.md` §6/D8) — this class only ever signals "the live fetch
    itself failed," never silently returns a guessed value.
    """


class FxClient(Protocol):
    """Minimal interface for fetching one day's USD->INR rate — narrow so a test double never
    needs to know about HTTP at all, matching this repo's existing `ModelClient`/`EmbeddingsClient`
    Protocol pattern (`app/models.py`, `app/search/embeddings.py`)."""

    async def get_rate(self, as_of: date) -> Decimal: ...


class FrankfurterClient:
    """Real, network-calling `FxClient` implementation. `transport` mirrors
    `app.tools.comtrade_client.ComtradeClient`'s existing constructor
    pattern — tests inject `httpx.MockTransport` here rather than reaching
    into a private attribute."""

    def __init__(
        self,
        *,
        base_url: str = FRANKFURTER_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_rate(self, as_of: date) -> Decimal:
        try:
            response = await self._client.get("/rate/USD/INR", params={"date": as_of.isoformat()})
        except httpx.TransportError as exc:
            raise FxRateFetchError(f"Frankfurter request failed: {exc}") from exc

        if response.status_code != 200:
            raise FxRateFetchError(
                f"Frankfurter returned status {response.status_code} for {as_of.isoformat()}: "
                f"{response.text[:200]}"
            )

        try:
            payload = response.json()
            rate_value = payload["rate"]
        except (ValueError, KeyError, TypeError) as exc:
            raise FxRateFetchError(
                f"Frankfurter response for {as_of.isoformat()} was not the expected shape: {exc}"
            ) from exc

        try:
            return Decimal(str(rate_value))
        except InvalidOperation as exc:
            raise FxRateFetchError(
                f"Frankfurter returned a non-numeric rate for {as_of.isoformat()}: {rate_value!r}"
            ) from exc
