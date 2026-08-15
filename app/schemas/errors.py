"""`ErrorResponse` — the schema every failure path returns instead of a raw
stack trace (master brief §9) or a silent partial render (docs/PLAN.md §3.2).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ErrorResponse(BaseModel):
    """User-safe error envelope."""

    model_config = ConfigDict(extra="forbid")

    error_code: str  # e.g. "UPSTREAM_TIMEOUT", "BUDGET_EXCEEDED", "INVALID_HS_CODE"
    message: str  # user-safe, never a raw stack trace (master brief §9)
    retryable: bool
    trace_id: str
