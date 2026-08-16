"""Integration tests for the `summarize` node against `MockLLM`, plus the
adversarial case that matters most: a model output containing a fabricated
number must be caught by the output guardrail and turned into an
`ErrorResponse`, never passed through to the user (master brief §2.2/§8).
"""

from __future__ import annotations

from typing import TypeVar

import pytest
from pydantic import BaseModel

import app.nodes.summarize as summarize_module
from app.models import MockLLM
from app.nodes.summarize import SummarizeOutput, summarize
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.schemas.response import CountryRow, TradeTable
from app.state import AnalysisState

T = TypeVar("T", bound=BaseModel)


def _table() -> TradeTable:
    # Five years, matching the real system's fixed methodology (master brief
    # §2.1: always a 5-year window) — a 3-year fixture would make
    # `len(table.years)` legitimately 3, not 5, and a summary saying
    # "5-year cumulative" would then (correctly, given only 3 years of
    # input) fail the output guardrail's number-grounding check.
    return TradeTable(
        unit="USD",
        years=[2019, 2020, 2021, 2022, 2023],
        years_finalized=[2019, 2020, 2021, 2022, 2023],
        excluded_partner_codes=["0"],
        rows=[
            CountryRow(
                partner_country="USA",
                partner_code="842",
                values_by_year={2019: 0.0, 2020: 50.0, 2021: 100.0, 2022: 150.0, 2023: 150.0},
                cumulative_5yr=450.0,
                rank=1,
            )
        ],
    )


class _FabricatingModelClient:
    """Test double that always returns a schema-valid but factually
    fabricated summary — proves the guardrail integration, not just the
    guardrail function in isolation (tests/unit/test_guardrails.py already
    covers the function itself)."""

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        return schema.model_validate(
            {"analytical_summary": "USA imported a record 99,999,999 units."}
        )


class _GroundedModelClient:
    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        return schema.model_validate(
            {"analytical_summary": "USA led imports with a 5-year cumulative value of 450."}
        )


@pytest.mark.integration
async def test_summarize_writes_analytical_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(summarize_module, "get_model_for_role", lambda role, provider: MockLLM())
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "imports_table": _table(),
        "exports_table": _table(),
    }

    result = await summarize(state)

    assert "analytical_summary" in result
    assert "error" not in result


@pytest.mark.integration
async def test_summarize_grounded_output_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        summarize_module, "get_model_for_role", lambda role, provider: _GroundedModelClient()
    )
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "imports_table": _table(),
        "exports_table": _table(),
    }

    result = await summarize(state)

    assert result["analytical_summary"] == "USA led imports with a 5-year cumulative value of 450."


@pytest.mark.integration
async def test_summarize_fabricated_number_is_caught_by_output_guardrail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        summarize_module, "get_model_for_role", lambda role, provider: _FabricatingModelClient()
    )
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "imports_table": _table(),
        "exports_table": _table(),
        "trace_id": "t-99",
    }

    result = await summarize(state)

    assert "analytical_summary" not in result
    error = result["error"]
    assert isinstance(error, ErrorResponse)
    assert error.error_code == "UNGROUNDED_SUMMARY"
    assert error.trace_id == "t-99"


@pytest.mark.integration
async def test_summarize_short_circuits_on_existing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(summarize_module, "get_model_for_role", lambda role, provider: MockLLM())
    state: AnalysisState = {
        "error": ErrorResponse(error_code="X", message="x", retryable=False, trace_id="t")
    }
    assert await summarize(state) == {}


@pytest.mark.integration
async def test_summarize_defensive_noop_when_state_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(summarize_module, "get_model_for_role", lambda role, provider: MockLLM())
    assert await summarize({}) == {}
    assert await summarize({"query": TradeQuery(hs_code="010121")}) == {}


@pytest.mark.integration
def test_summarize_output_schema_rejects_empty_string() -> None:
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        SummarizeOutput(analytical_summary="")
