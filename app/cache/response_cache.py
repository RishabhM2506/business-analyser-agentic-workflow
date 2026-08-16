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
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.schemas.query import TradeQuery
from app.schemas.response import TradeAnalysisResponse

RESPONSE_TTL_SECONDS = 24 * 60 * 60  # 24h safety-net TTL; see module docstring


def filter_hash(query: TradeQuery) -> str:
    """Stable hash of the `TradeQuery` fields that affect the analysis but
    aren't already their own separate cache-key component (finding
    B8/AWR-01 — this function didn't exist anywhere before).

    Deliberately excludes:
    - `hs_code` — already the cache key's first component.
    - `year_start`/`year_end` — the caller resolves these (see
      `app.nodes.validate_query.resolve_year_range`) into the key's
      separate `year_range` component *before* calling this, so two
      requests that resolve to the identical concrete year range (e.g. one
      passing explicit years, another passing `None`/`None` on the same
      day) still hit the same cache entry rather than being kept apart by
      hashing the raw, pre-resolution fields here.
    - `tenant_id`/`user_id` — identity, never affect what analysis comes
      back for a given `hs_code`/year range/filter combination.

    Includes every field that actually changes the analysis:
    `flow`/`partner_region`/`value_or_volume` — the reserved-for-future
    filters (`app/schemas/query.py`) that don't affect v1's fixed pipeline
    output *yet*, but must already be part of the key so that the moment
    any of them becomes live (docs/PLAN.md §3 roadmap), a differently-
    filtered request can never silently be served another request's cached
    result.

    `json.dumps(..., sort_keys=True)` rather than naive string
    concatenation: a free-form field like `partner_region` could otherwise
    contain a delimiter character and collide with a differently-shaped
    combination of fields.
    """
    canonical = json.dumps(
        {
            "flow": query.flow,
            "partner_region": query.partner_region,
            "value_or_volume": query.value_or_volume,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
