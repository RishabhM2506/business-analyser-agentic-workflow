"""`TradeQuery` — the only way any node receives filter parameters.

Exact field set per docs/PLAN.md §3.1. `partner_region` is unused in v1 and
reserved for a future roadmap filter (master brief §3); `value_or_volume`
supports only `"value"` in v1, `"volume"` is reserved.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Generous ceiling for the only free-form string fields on this schema
# (finding B9/QA-02): rejected by Pydantic itself, independent of the
# ASGI-level request body-size middleware (`app/main.py`'s
# `request_size_limit_middleware`) — a within-the-64KB-body-limit but
# still-abusive single field (e.g. a multi-KB `tenant_id`) must not reach
# a node, a log line, or a checkpoint write unnoticed. 128 chars is well
# beyond any real tenant/user identifier or region name.
_FREE_TEXT_FIELD_MAX_LENGTH = 128


class TradeQuery(BaseModel):
    """Filter parameters for one trade-data analysis request."""

    model_config = ConfigDict(extra="forbid")

    hs_code: str = Field(pattern=r"^\d{6}$")
    flow: Literal["import", "export", "both"] = "both"
    year_start: int | None = None
    year_end: int | None = None
    partner_region: str | None = Field(default=None, max_length=_FREE_TEXT_FIELD_MAX_LENGTH)
    value_or_volume: Literal["value", "volume"] = "value"
    tenant_id: str = Field(default="default", max_length=_FREE_TEXT_FIELD_MAX_LENGTH)
    user_id: str = Field(default="default", max_length=_FREE_TEXT_FIELD_MAX_LENGTH)
