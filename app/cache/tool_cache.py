"""Tool-result cache (docs/PLAN.md §5.4 level 3): raw Comtrade fetches,
durable TTL for years flagged `finalized` (immutable trade data); short TTL
for years still flagged provisional (docs/PLAN.md §11 risk: a provisional
year may later finalize with different values).

Key granularity: `(hs_code, flow, year)` -> `list[ComtradeRecord]` — adjusted
from the Phase 2 stub's `(hs_code, partner_code, year)` once Phase 3's live
verification of the UN Comtrade API (see `app/tools/comtrade_client.py`'s
module docstring) showed that a single fetch already returns the *entire*
partner breakdown for one (hs_code, flow, year): that's the client's actual
unit of work, and caching at partner-row granularity would mean either N
cache lookups to reconstruct one year (mostly pointless — the client can't
fetch a single partner's row on its own, only the whole year) or writing
every row from one response under N different keys for no benefit. Caching
at fetch granularity is simpler and correct: `is_finalized` is decided once
per fetch, from the constituent records' `is_provisional` flags (a year is
finalized only if every retained record for it was, per Comtrade,
genuinely reported rather than modeled/estimated).

In-process only (a plain dict), matching `docs/PLAN.md` §1.2's documented,
accepted v1 risk for the response cache and general in-process approach (no
Redis, single instance, no cross-replica sharing needed yet).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.tools.comtrade_client import ComtradeRecord

# Trade data for a year Comtrade has finalized is treated as immutable
# (master brief §7.5) -> cache durably. A provisional year may still be
# revised as more countries report -> cache only briefly, so a user request
# soon after ingestion doesn't need to wait on a live refetch every time,
# but the app re-checks upstream well before the year is likely to have
# changed meaningfully.
FINALIZED_TTL_SECONDS = 180 * 24 * 60 * 60  # ~6 months
PROVISIONAL_TTL_SECONDS = 6 * 60 * 60  # 6 hours


@dataclass
class _CacheEntry:
    records: list[ComtradeRecord]
    expires_at: float


class ToolCache:
    """In-process cache keyed on `(hs_code, flow, year)`."""

    def __init__(
        self,
        *,
        finalized_ttl_seconds: float = FINALIZED_TTL_SECONDS,
        provisional_ttl_seconds: float = PROVISIONAL_TTL_SECONDS,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._finalized_ttl_seconds = finalized_ttl_seconds
        self._provisional_ttl_seconds = provisional_ttl_seconds
        self._now_fn = now_fn
        self._entries: dict[tuple[str, str, int], _CacheEntry] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(hs_code: str, flow: Literal["import", "export"], year: int) -> tuple[str, str, int]:
        return (hs_code, flow, year)

    async def get(
        self, *, hs_code: str, flow: Literal["import", "export"], year: int
    ) -> list[ComtradeRecord] | None:
        key = self._key(hs_code, flow, year)
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at < self._now_fn():
                del self._entries[key]
                return None
            return list(entry.records)

    async def set(
        self,
        *,
        hs_code: str,
        flow: Literal["import", "export"],
        year: int,
        records: list[ComtradeRecord],
        is_finalized: bool,
    ) -> None:
        ttl = self._finalized_ttl_seconds if is_finalized else self._provisional_ttl_seconds
        entry = _CacheEntry(records=list(records), expires_at=self._now_fn() + ttl)
        async with self._lock:
            self._entries[self._key(hs_code, flow, year)] = entry


_tool_cache_singleton: ToolCache | None = None


def get_tool_cache() -> ToolCache:
    """Process-wide singleton, matching `app.tools.comtrade_client.get_comtrade_client`'s
    pattern — the whole point of this cache is to be shared across requests
    within the running instance."""
    global _tool_cache_singleton
    if _tool_cache_singleton is None:
        _tool_cache_singleton = ToolCache()
    return _tool_cache_singleton
