"""Integration tests for the `describe_item` node against `MockLLM`
(`LLM_PROVIDER=mock` path — zero token spend, master brief §6). Cassette-
based tests exercising the Gemini-structured-output-shaped path live under
`tests/llm/` instead (the `llm` marker).
"""

from __future__ import annotations

import pytest

import app.nodes.describe_item as describe_item_module
from app.budget import BudgetTracker
from app.models import MockLLM
from app.nodes.describe_item import describe_item
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.state import AnalysisState


def _patch_model_and_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        describe_item_module, "get_model_for_role", lambda role, provider: MockLLM()
    )
    # Fresh tracker per test (isolated from the process-wide singleton and
    # from every other test) rather than the real default ceiling of 2 —
    # this test file only cares about `describe_item`'s own behavior, not
    # budget enforcement (see `tests/unit/test_budget.py` for that), so it
    # uses a tracker generous enough to never trip.
    monkeypatch.setattr(
        describe_item_module,
        "get_budget_tracker",
        lambda: BudgetTracker(max_calls_per_thread=100, max_calls_per_day=100),
    )


@pytest.mark.integration
async def test_describe_item_writes_item_description(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_and_budget(monkeypatch)
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "taxonomy_text": "HS 010121: Horses; live, pure-bred breeding animals",
        "thread_id": "t-describe-1",
    }

    result = await describe_item(state)

    assert "item_description" in result
    assert isinstance(result["item_description"], str)
    assert len(result["item_description"]) > 0


@pytest.mark.integration
async def test_describe_item_short_circuits_on_existing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model_and_budget(monkeypatch)
    state: AnalysisState = {
        "error": ErrorResponse(error_code="X", message="x", retryable=False, trace_id="t")
    }
    assert await describe_item(state) == {}


@pytest.mark.integration
async def test_describe_item_defensive_noop_when_state_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model_and_budget(monkeypatch)
    assert await describe_item({}) == {}
    assert await describe_item({"query": TradeQuery(hs_code="010121")}) == {}  # no taxonomy_text
    # taxonomy_text present but no thread_id: still a defensive no-op, same
    # class of "should never happen once app/main.py seeds it" as the case
    # above (see `app/state.py`'s `thread_id` field docstring).
    assert (
        await describe_item(
            {
                "query": TradeQuery(hs_code="010121"),
                "taxonomy_text": "HS 010121: Horses; live, pure-bred breeding animals",
            }
        )
        == {}
    )


@pytest.mark.integration
async def test_describe_item_budget_exceeded_short_circuits_without_calling_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def _tracking_mock_llm(role: str, provider: str) -> MockLLM:
        nonlocal called
        called = True
        return MockLLM()

    monkeypatch.setattr(describe_item_module, "get_model_for_role", _tracking_mock_llm)
    monkeypatch.setattr(
        describe_item_module,
        "get_budget_tracker",
        lambda: BudgetTracker(max_calls_per_thread=0, max_calls_per_day=100),
    )
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "taxonomy_text": "HS 010121: Horses; live, pure-bred breeding animals",
        "thread_id": "t-describe-budget",
        "trace_id": "t-99",
    }

    result = await describe_item(state)

    assert called is False  # never reached the model call
    assert "item_description" not in result
    error = result["error"]
    assert error.error_code == "BUDGET_EXCEEDED"
    assert error.trace_id == "t-99"
