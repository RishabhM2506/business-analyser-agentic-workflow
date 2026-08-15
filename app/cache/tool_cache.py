"""Tool-result cache (docs/PLAN.md §5.4 level 3): raw Comtrade fetches keyed
`(hs_code, partner, year)`. Durable TTL for years flagged `finalized`
(immutable trade data); short TTL for years still flagged provisional
(docs/PLAN.md §11 risk: a provisional year may later finalize with
different values).

# TODO(Phase 3): implement a real backing store with the
# finalized/provisional TTL split described above.
"""

from __future__ import annotations

from app.tools.comtrade_client import ComtradeRecord


class ToolCache:
    """Cache keyed on `(hs_code, partner_code, year)`."""

    async def get(self, *, hs_code: str, partner_code: str, year: int) -> ComtradeRecord | None:
        raise NotImplementedError  # TODO(Phase 3): implement cache lookup.

    async def set(
        self,
        *,
        hs_code: str,
        partner_code: str,
        year: int,
        record: ComtradeRecord,
        is_finalized: bool,
    ) -> None:
        raise NotImplementedError  # TODO(Phase 3): implement cache write with TTL split.
