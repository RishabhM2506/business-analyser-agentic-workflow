"""Integration tests for the `describe_item` node against `MockLLM`
(`LLM_PROVIDER=mock` path — zero token spend, master brief §6). Cassette-
based tests exercising the Gemini-structured-output-shaped path live under
`tests/llm/` instead (the `llm` marker).
"""

from __future__ import annotations

import pytest

import app.nodes.describe_item as describe_item_module
from app.models import MockLLM
from app.nodes.describe_item import describe_item
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.state import AnalysisState


@pytest.mark.integration
async def test_describe_item_writes_item_description(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(describe_item_module, "get_model_for_role", lambda role, provider: MockLLM())
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "taxonomy_text": "HS 010121: Horses; live, pure-bred breeding animals",
    }

    result = await describe_item(state)

    assert "item_description" in result
    assert isinstance(result["item_description"], str)
    assert len(result["item_description"]) > 0


@pytest.mark.integration
async def test_describe_item_short_circuits_on_existing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(describe_item_module, "get_model_for_role", lambda role, provider: MockLLM())
    state: AnalysisState = {
        "error": ErrorResponse(error_code="X", message="x", retryable=False, trace_id="t")
    }
    assert await describe_item(state) == {}


@pytest.mark.integration
async def test_describe_item_defensive_noop_when_state_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(describe_item_module, "get_model_for_role", lambda role, provider: MockLLM())
    assert await describe_item({}) == {}
    assert await describe_item({"query": TradeQuery(hs_code="010121")}) == {}  # no taxonomy_text
