"""Application response cache (docs/PLAN.md §5.4 level 2): keyed
`(hs_code, year_range, filter_hash, prompt_version)` -> full
`TradeAnalysisResponse`. This is the cache layer that makes docs/PLAN.md
§5.2's argument true — a given `hs_code` triggers real model spend at most
once.

# TODO(Phase 3): implement a real backing store. An in-process dict is
# sufficient for v1's single-instance deployment (docs/PLAN.md §1.2's
# documented Redis-deferral); a prompt-version bump or a provisional-year
# refinalization must bust the relevant entries (docs/PLAN.md §5.6, §11).
"""

from __future__ import annotations

from app.schemas.response import TradeAnalysisResponse


class ResponseCache:
    """Cache keyed on `(hs_code, year_range, filter_hash, prompt_version)`."""

    async def get(
        self,
        *,
        hs_code: str,
        year_range: tuple[int, int],
        filter_hash: str,
        prompt_version: str,
    ) -> TradeAnalysisResponse | None:
        raise NotImplementedError  # TODO(Phase 3): implement cache lookup.

    async def set(
        self,
        *,
        hs_code: str,
        year_range: tuple[int, int],
        filter_hash: str,
        prompt_version: str,
        response: TradeAnalysisResponse,
    ) -> None:
        raise NotImplementedError  # TODO(Phase 3): implement cache write.
