"""Integration tests for the `describe_item` node against `MockLLM`
(`LLM_PROVIDER=mock` path — zero token spend, master brief §6). Cassette-
based tests exercising the Gemini-structured-output-shaped path live under
`tests/llm/` instead (the `llm` marker).
"""

from __future__ import annotations

from typing import TypeVar

import pytest
import structlog
from pydantic import BaseModel

import app.nodes.describe_item as describe_item_module
from app.budget import BudgetTracker
from app.models import MockLLM
from app.nodes.describe_item import describe_item
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.state import AnalysisState

T = TypeVar("T", bound=BaseModel)


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


class _NumberInventingModelClient:
    """Test double that always returns a schema-valid `description`
    containing a number — proves the code-level guardrail integration
    (finding M2/AWR-04), not just that the prompt *asks* the model not to
    do this (master brief §8 explicitly forbids relying on prompt wording
    alone as the only control)."""

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        return schema.model_validate(
            {"description": "This category represents about 4 percent of typical trade volume."}
        )


@pytest.mark.integration
async def test_describe_item_output_containing_a_number_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        describe_item_module,
        "get_model_for_role",
        lambda role, provider: _NumberInventingModelClient(),
    )
    monkeypatch.setattr(
        describe_item_module,
        "get_budget_tracker",
        lambda: BudgetTracker(max_calls_per_thread=100, max_calls_per_day=100),
    )
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "taxonomy_text": "HS 010121: Horses; live, pure-bred breeding animals",
        "thread_id": "t-describe-ungrounded",
        "trace_id": "t-88",
    }

    with structlog.testing.capture_logs() as captured_logs:
        result = await describe_item(state)

    assert "item_description" not in result
    error = result["error"]
    assert isinstance(error, ErrorResponse)
    assert error.error_code == "UNGROUNDED_DESCRIPTION"

    # QA finding (2026-08-20) -- mirrors summarize.py's identical fix; see
    # that test file's comment for the full rationale.
    rejection_logs = [
        log for log in captured_logs if log["event"] == "describe_item.guardrail_rejected"
    ]
    assert len(rejection_logs) == 1
    assert rejection_logs[0]["log_level"] == "warning"
    assert rejection_logs[0]["hs_code"] == "010121"
    assert rejection_logs[0]["found_numbers"] == [4.0]
    assert "4 percent" in rejection_logs[0]["rejected_description"]
    assert error.trace_id == "t-88"


@pytest.mark.integration
async def test_describe_item_mock_llm_output_never_contains_a_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`MockLLM`'s `description`-field output must be number-free by
    construction (app/models.py's `_mock_text_for`) — its input always
    contains the hs_code's own digits, so this specifically regression-tests
    that the mock doesn't defeat its own purpose (running the real pipeline
    end-to-end with zero token spend) by tripping the new guardrail on
    every single call under `LLM_PROVIDER=mock`."""
    _patch_model_and_budget(monkeypatch)
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "taxonomy_text": "HS 010121: Horses; live, pure-bred breeding animals",
        "thread_id": "t-describe-mock-clean",
    }

    result = await describe_item(state)

    assert "error" not in result
    assert "item_description" in result
