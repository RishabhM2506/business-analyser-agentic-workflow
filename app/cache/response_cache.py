"""Application response cache (docs/PLAN.md §5.4 level 2): keyed
`(hs_code, year_range, filter_hash, prompt_version)` -> full
`TradeAnalysisResponse`. This is the cache layer that makes docs/PLAN.md
§5.2's argument true — a given `hs_code` triggers real model spend at most
once.

In-process dict, same accepted v1 scope as `app/cache/tool_cache.py` and
`docs/PLAN.md` §1.2 (single instance, no Redis yet). The *key structure* is
the primary invalidation mechanism here, not TTL: a `prompt_version` bump or
a different `filter_hash` is automatically a cache miss (a new key), which is
the deliberate design named in docs/PLAN.md §5.6 ("a prompt edit is a
cache-busting event, not a silent behavior change against stale cached
output"). A generous TTL is still applied as a safety net — primarily to
bound memory growth and to eventually pick up a provisional-year
refinalization (docs/PLAN.md §11) without a code deploy.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.schemas.response import TradeAnalysisResponse

RESPONSE_TTL_SECONDS = 24 * 60 * 60  # 24h safety-net TTL; see module docstring


@dataclass
class _CacheEntry:
    response: TradeAnalysisResponse
    expires_at: float


class ResponseCache:
    """Cache keyed on `(hs_code, year_range, filter_hash, prompt_version)`."""

    def __init__(
        self,
        *,
        ttl_seconds: float = RESPONSE_TTL_SECONDS,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._now_fn = now_fn
        self._entries: dict[tuple[str, tuple[int, int], str, str], _CacheEntry] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(
        hs_code: str, year_range: tuple[int, int], filter_hash: str, prompt_version: str
    ) -> tuple[str, tuple[int, int], str, str]:
        return (hs_code, year_range, filter_hash, prompt_version)

    async def get(
        self,
        *,
        hs_code: str,
        year_range: tuple[int, int],
        filter_hash: str,
        prompt_version: str,
    ) -> TradeAnalysisResponse | None:
        key = self._key(hs_code, year_range, filter_hash, prompt_version)
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at < self._now_fn():
                del self._entries[key]
                return None
            return entry.response

    async def set(
        self,
        *,
        hs_code: str,
        year_range: tuple[int, int],
        filter_hash: str,
        prompt_version: str,
        response: TradeAnalysisResponse,
    ) -> None:
        key = self._key(hs_code, year_range, filter_hash, prompt_version)
        entry = _CacheEntry(response=response, expires_at=self._now_fn() + self._ttl_seconds)
        async with self._lock:
            self._entries[key] = entry


_response_cache_singleton: ResponseCache | None = None


def get_response_cache() -> ResponseCache:
    """Process-wide singleton — see `app.cache.tool_cache.get_tool_cache`."""
    global _response_cache_singleton
    if _response_cache_singleton is None:
        _response_cache_singleton = ResponseCache()
    return _response_cache_singleton
