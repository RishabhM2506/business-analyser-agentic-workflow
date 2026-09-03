"""Integration tests for the `summarize` node against `MockLLM`, plus the
adversarial case that matters most: a model output containing a fabricated
number must be caught by the output guardrail and turned into an
`ErrorResponse`, never passed through to the user (master brief §2.2/§8).
"""

from __future__ import annotations

from typing import TypeVar

import pytest
import structlog
from pydantic import BaseModel

import app.nodes.summarize as summarize_module
from app.budget import BudgetTracker
from app.models import MockLLM
from app.nodes.summarize import SummarizeOutput, summarize
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.schemas.response import CountryRow, TradeTable
from app.state import AnalysisState

T = TypeVar("T", bound=BaseModel)


def _patch_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fresh, generous-ceiling tracker per test (isolated from the
    # process-wide singleton) — these tests cover `summarize`'s own
    # behavior, not budget enforcement (see `tests/unit/test_budget.py`).
    monkeypatch.setattr(
        summarize_module,
        "get_budget_tracker",
        lambda: BudgetTracker(max_calls_per_thread=100, max_calls_per_day=100),
    )


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


class _RaisingModelClient:
    """Simulates a total model-call failure, e.g. every key in a
    load-balanced pool exhausted (`app.models.LoadBalancedGeminiModelClient`
    re-raises the real last exception on total exhaustion)."""

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        raise RuntimeError("503 UNAVAILABLE - high demand")


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
    _patch_budget(monkeypatch)
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "imports_table": _table(),
        "exports_table": _table(),
        "thread_id": "t-summarize-1",
    }

    result = await summarize(state)

    assert "analytical_summary" in result
    assert "error" not in result


@pytest.mark.integration
async def test_summarize_grounded_output_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        summarize_module, "get_model_for_role", lambda role, provider: _GroundedModelClient()
    )
    _patch_budget(monkeypatch)
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "imports_table": _table(),
        "exports_table": _table(),
        "thread_id": "t-summarize-2",
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
    _patch_budget(monkeypatch)
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "imports_table": _table(),
        "exports_table": _table(),
        "thread_id": "t-summarize-3",
        "trace_id": "t-99",
    }

    with structlog.testing.capture_logs() as captured_logs:
        result = await summarize(state)

    assert "analytical_summary" not in result
    error = result["error"]
    assert isinstance(error, ErrorResponse)
    assert error.error_code == "UNGROUNDED_SUMMARY"

    # QA finding (2026-08-20): a real guardrail rejection previously left
    # zero trace of *why* in the logs -- the rejected text and the specific
    # offending number(s) were simply discarded, undiagnosable after the
    # fact. Assert the fix actually logs enough to debug a real production
    # rejection without needing to reproduce it.
    rejection_logs = [
        log for log in captured_logs if log["event"] == "summarize.guardrail_rejected"
    ]
    assert len(rejection_logs) == 1
    assert rejection_logs[0]["log_level"] == "warning"
    assert rejection_logs[0]["hs_code"] == "010121"
    assert rejection_logs[0]["ungrounded_numbers"] == [99999999.0]
    assert "99,999,999" in rejection_logs[0]["rejected_summary"]
    assert error.trace_id == "t-99"


@pytest.mark.integration
async def test_summarize_model_call_failure_returns_upstream_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-09-03 fix, live-reproduced: a total model-call failure (e.g.
    every key in a load-balanced pool exhausted) previously propagated
    uncaught past this node into an unclassified `INTERNAL_ERROR` 500 -
    contradicting this module's own "defined error fallback, never a
    silent partial render" principle."""
    monkeypatch.setattr(
        summarize_module, "get_model_for_role", lambda role, provider: _RaisingModelClient()
    )
    _patch_budget(monkeypatch)
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "imports_table": _table(),
        "exports_table": _table(),
        "thread_id": "t-summarize-4",
        "trace_id": "t-88",
    }

    with structlog.testing.capture_logs() as captured_logs:
        result = await summarize(state)

    assert "analytical_summary" not in result
    error = result["error"]
    assert isinstance(error, ErrorResponse)
    assert error.error_code == "UPSTREAM_TIMEOUT"
    assert error.retryable is True
    assert error.trace_id == "t-88"

    failure_logs = [log for log in captured_logs if log["event"] == "summarize.model_call_failed"]
    assert len(failure_logs) == 1
    assert failure_logs[0]["log_level"] == "warning"
    assert failure_logs[0]["hs_code"] == "010121"


@pytest.mark.integration
async def test_summarize_short_circuits_on_existing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(summarize_module, "get_model_for_role", lambda role, provider: MockLLM())
    _patch_budget(monkeypatch)
    state: AnalysisState = {
        "error": ErrorResponse(error_code="X", message="x", retryable=False, trace_id="t")
    }
    assert await summarize(state) == {}


@pytest.mark.integration
async def test_summarize_defensive_noop_when_state_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(summarize_module, "get_model_for_role", lambda role, provider: MockLLM())
    _patch_budget(monkeypatch)
    assert await summarize({}) == {}
    assert await summarize({"query": TradeQuery(hs_code="010121")}) == {}
    # tables present but no thread_id: still a defensive no-op (see
    # `app/state.py`'s `thread_id` field docstring).
    assert (
        await summarize(
            {
                "query": TradeQuery(hs_code="010121"),
                "imports_table": _table(),
                "exports_table": _table(),
            }
        )
        == {}
    )


@pytest.mark.integration
async def test_summarize_budget_exceeded_short_circuits_without_calling_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def _tracking_mock_llm(role: str, provider: str) -> MockLLM:
        nonlocal called
        called = True
        return MockLLM()

    monkeypatch.setattr(summarize_module, "get_model_for_role", _tracking_mock_llm)
    monkeypatch.setattr(
        summarize_module,
        "get_budget_tracker",
        lambda: BudgetTracker(max_calls_per_thread=0, max_calls_per_day=100),
    )
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "imports_table": _table(),
        "exports_table": _table(),
        "thread_id": "t-summarize-budget",
        "trace_id": "t-77",
    }

    result = await summarize(state)

    assert called is False
    assert "analytical_summary" not in result
    error = result["error"]
    assert error.error_code == "BUDGET_EXCEEDED"
    assert error.trace_id == "t-77"


@pytest.mark.integration
def test_summarize_output_schema_rejects_empty_string() -> None:
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        SummarizeOutput(analytical_summary="")
