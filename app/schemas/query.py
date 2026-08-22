"""`TradeQuery` — the only way any node receives filter parameters.

Exact field set per docs/PLAN.md §3.1. `partner_region` is unused in v1 and
reserved for a future roadmap filter (master brief §3); `value_or_volume`
supports only `"value"` in v1, `"volume"` is reserved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Generous ceiling for the only free-form string fields on this schema
# (finding B9/QA-02): rejected by Pydantic itself, independent of the
# ASGI-level request body-size middleware (`app/main.py`'s
# `request_size_limit_middleware`) — a within-the-64KB-body-limit but
# still-abusive single field (e.g. a multi-KB `tenant_id`) must not reach
# a node, a log line, or a checkpoint write unnoticed. 128 chars is well
# beyond any real tenant/user identifier or region name.
_FREE_TEXT_FIELD_MAX_LENGTH = 128

# Finding M10/QA-04: year_start/year_end had no bounds or ordering
# validation at all. QA-04 live-reproduced three failure modes against the
# unvalidated schema: an inverted range (year_start > year_end) silently
# 200s with an empty-but-valid table instead of a 400; an unbounded span
# (a 75-year range) triggers a proportionally unbounded number of
# sequential upstream calls (150 real HTTP calls observed for one request);
# negative years are accepted without complaint. None of these fabricate
# data (an honestly-empty table is consistent with "missing data stays
# missing"), but all three are a real gap against "input validation... at
# the API edge" and a genuine amplification vector against a rate-limited,
# non-Anthropic-controlled upstream this project doesn't own.
EARLIEST_SUPPORTED_YEAR = 1962  # generous floor - predates UN Comtrade's own annual HS data
MAX_YEAR_SPAN = 20  # generous ceiling on a single request's requested year-range width


def _current_year() -> int:
    return datetime.now(UTC).year


class TradeQuery(BaseModel):
    """Filter parameters for one trade-data analysis request."""

    model_config = ConfigDict(extra="forbid")

    hs_code: str = Field(pattern=r"^\d{6}$")
    flow: Literal["import", "export", "both"] = "both"
    year_start: int | None = Field(default=None, ge=EARLIEST_SUPPORTED_YEAR)
    year_end: int | None = Field(default=None, ge=EARLIEST_SUPPORTED_YEAR)
    partner_region: str | None = Field(default=None, max_length=_FREE_TEXT_FIELD_MAX_LENGTH)
    value_or_volume: Literal["value", "volume"] = "value"
    tenant_id: str = Field(default="default", max_length=_FREE_TEXT_FIELD_MAX_LENGTH)
    user_id: str = Field(default="default", max_length=_FREE_TEXT_FIELD_MAX_LENGTH)

    @field_validator("year_start", "year_end")
    @classmethod
    def _reject_absurdly_future_years(cls, value: int | None) -> int | None:
        """A static `Field(le=...)` can't express "not more than a year
        ahead of whenever this request actually arrives" — the bound has to
        move with the calendar, so it's computed here at validation time
        instead of baked into the class definition. One year of headroom
        past the current year tolerates timezone/clock skew at the edge of
        a year boundary without over-rejecting an otherwise-ordinary
        request; UN Comtrade never has data for the current year anyway
        (`app/nodes/validate_query.py`'s own "latest available" heuristic
        already assumes `current_year - 1`), so this is about rejecting
        nonsense, not about a real data availability boundary."""
        if value is not None and value > _current_year() + 1:
            raise ValueError(
                f"year {value} is more than one year in the future and cannot be valid"
            )
        return value

    @model_validator(mode="after")
    def _validate_year_range_ordering_and_span(self) -> TradeQuery:
        if self.year_start is not None and self.year_end is not None:
            if self.year_start > self.year_end:
                raise ValueError(
                    f"year_start ({self.year_start}) must be <= year_end ({self.year_end})"
                )
            span = self.year_end - self.year_start + 1
            if span > MAX_YEAR_SPAN:
                raise ValueError(
                    f"requested year range spans {span} years, which exceeds the "
                    f"maximum allowed span of {MAX_YEAR_SPAN} years"
                )
        return self


class ProductSearchQuery(BaseModel):
    """Request body for `POST /threads/{thread_id}/search`
    (`app.search.service`) — free text in, ranked HS6 candidates out."""

    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(default="default", max_length=_FREE_TEXT_FIELD_MAX_LENGTH)
    user_id: str = Field(default="default", max_length=_FREE_TEXT_FIELD_MAX_LENGTH)
