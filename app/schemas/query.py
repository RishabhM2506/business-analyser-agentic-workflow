"""`TradeQuery` — the only way any node receives filter parameters.

Exact field set per docs/PLAN.md §3.1. `partner_region` is unused in v1 and
reserved for a future roadmap filter (master brief §3); `value_or_volume`
supports only `"value"` in v1, `"volume"` is reserved.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TradeQuery(BaseModel):
    """Filter parameters for one trade-data analysis request."""

    model_config = ConfigDict(extra="forbid")

    hs_code: str = Field(pattern=r"^\d{6}$")
    flow: Literal["import", "export", "both"] = "both"
    year_start: int | None = None
    year_end: int | None = None
    partner_region: str | None = None
    value_or_volume: Literal["value", "volume"] = "value"
    tenant_id: str = "default"
    user_id: str = "default"
