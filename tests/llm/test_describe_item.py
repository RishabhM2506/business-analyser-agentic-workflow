"""Cassette-replay tests for the `describe_item` node (docs/PLAN.md §7's
`llm`-marked layer). See `tests/llm/cassette.py`'s module docstring: every
cassette here is **synthetic** (no real Gemini key exists yet), not a real
recording — deterministic and free either way, which is the actual point
of this test layer (master brief §6).
"""

from __future__ import annotations

import pytest

import app.nodes.describe_item as describe_item_module
from app.knowledge.provider import build_taxonomy_text
from app.nodes.describe_item import describe_item
from app.schemas.query import TradeQuery
from app.state import AnalysisState
from tests.llm.cassette import CASSETTES_DIR, CassetteModelClient

_DESCRIBE_ITEM_CASSETTES = CASSETTES_DIR / "describe_item"


@pytest.mark.llm
@pytest.mark.parametrize("hs_code", ["010121", "090111", "847130"])
async def test_describe_item_replays_cassette_for_real_hs6_codes(
    monkeypatch: pytest.MonkeyPatch, hs_code: str
) -> None:
    cassette = CassetteModelClient(_DESCRIBE_ITEM_CASSETTES / f"{hs_code}.json")
    monkeypatch.setattr(describe_item_module, "get_model_for_role", lambda role, provider: cassette)

    taxonomy_text = build_taxonomy_text(hs_code)
    assert taxonomy_text is not None  # all three are real checked-in HS6 codes
    state: AnalysisState = {"query": TradeQuery(hs_code=hs_code), "taxonomy_text": taxonomy_text}

    result = await describe_item(state)

    assert "item_description" in result
    assert hs_code in taxonomy_text  # sanity: we're describing the code we asked for
    assert len(result["item_description"]) > 0


@pytest.mark.llm
async def test_describe_item_cassette_output_is_schema_valid_and_marked_synthetic() -> None:
    import json

    for cassette_path in _DESCRIBE_ITEM_CASSETTES.glob("*.json"):
        payload = json.loads(cassette_path.read_text(encoding="utf-8"))
        assert payload["_meta"]["synthetic"] is True, f"{cassette_path} must be marked synthetic"
        assert "description" in payload["output"]
        assert len(payload["output"]["description"]) > 0
