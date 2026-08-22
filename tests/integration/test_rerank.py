"""Integration tests for `app.search.rerank.rerank_candidates` — the search
feature's code-identity guardrail. Mirrors
`tests/integration/test_describe_item.py`'s `_NumberInventingModelClient`
pattern exactly: a fake model client that returns something it was never
given, proving the code-level guard (not prompt wording alone) is what
actually stops it from reaching a user. This is the single most important
test in the whole search feature — the code-identity equivalent of the
number-grounding guardrail's own most important test.
"""

from __future__ import annotations

from typing import TypeVar

import pytest
import structlog
from pydantic import BaseModel, ValidationError

from app.search.candidates import SearchCandidate
from app.search.rerank import (
    HIGH_CONFIDENCE_THRESHOLD,
    RerankOutput,
    UngroundedRerankError,
    rerank_candidates,
)

T = TypeVar("T", bound=BaseModel)


def _candidates() -> list[SearchCandidate]:
    return [
        SearchCandidate(
            hs_code="090111",
            description="Coffee, not roasted, not decaffeinated",
            fusion_score=0.9,
        ),
        SearchCandidate(
            hs_code="090121", description="Coffee, roasted, not decaffeinated", fusion_score=0.5
        ),
        SearchCandidate(hs_code="090190", description="Coffee husks and skins", fusion_score=0.2),
    ]


class _EchoingModelClient:
    """A well-behaved fake: ranks exactly the candidates it was given, in
    the order given, with descending scores — used for the happy-path
    tests below (never invents anything)."""

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        candidates = _candidates()
        payload = {
            "ranked_candidates": [
                {"hs_code": c.hs_code, "relevance_score": max(0.1, 0.9 - 0.3 * i)}
                for i, c in enumerate(candidates)
            ]
        }
        return schema.model_validate(payload)


class _InventedCodeModelClient:
    """Test double that always returns an HS6 code that was never in the
    candidate set it was given — the exact failure mode
    `check_hs_codes_grounded` exists to catch (a model surfacing a real,
    plausible-looking HS6 code from its own training-data knowledge of the
    HS taxonomy instead of confining itself to what search actually
    found)."""

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        payload = {
            "ranked_candidates": [
                {"hs_code": "090111", "relevance_score": 0.9},
                {"hs_code": "999999", "relevance_score": 0.8},  # never a candidate
            ]
        }
        return schema.model_validate(payload)


@pytest.mark.integration
async def test_rerank_candidates_happy_path_returns_sorted_ranked_candidates() -> None:
    result = await rerank_candidates("coffee", _candidates(), model_client=_EchoingModelClient())

    assert [r.hs_code for r in result] == ["090111", "090121", "090190"]
    scores = [r.relevance_score for r in result]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
async def test_rerank_candidates_sorts_by_score_not_model_list_order() -> None:
    """Never trust the model's own list order (app/search/rerank.py's own
    docstring) — even if the model returns candidates out of score order,
    `rerank_candidates` must resort them in code."""

    class _OutOfOrderModelClient:
        async def generate_structured(
            self, *, system_prompt: str, user_content: str, schema: type[T]
        ) -> T:
            payload = {
                "ranked_candidates": [
                    {"hs_code": "090190", "relevance_score": 0.2},
                    {"hs_code": "090111", "relevance_score": 0.95},
                    {"hs_code": "090121", "relevance_score": 0.6},
                ]
            }
            return schema.model_validate(payload)

    result = await rerank_candidates("coffee", _candidates(), model_client=_OutOfOrderModelClient())

    assert [r.hs_code for r in result] == ["090111", "090121", "090190"]


@pytest.mark.integration
async def test_rerank_candidates_rejects_an_invented_hs_code() -> None:
    with (
        structlog.testing.capture_logs() as captured_logs,
        pytest.raises(UngroundedRerankError) as exc_info,
    ):
        await rerank_candidates("coffee", _candidates(), model_client=_InventedCodeModelClient())

    assert exc_info.value.ungrounded_codes == ["999999"]

    rejection_logs = [log for log in captured_logs if log["event"] == "rerank.guardrail_rejected"]
    assert len(rejection_logs) == 1
    assert rejection_logs[0]["log_level"] == "warning"
    assert rejection_logs[0]["ungrounded_codes"] == ["999999"]
    assert set(rejection_logs[0]["candidate_codes"]) == {"090111", "090121", "090190"}


@pytest.mark.integration
async def test_rerank_candidates_invented_code_invalidates_whole_result_not_just_entry() -> None:
    """All-or-nothing (mirrors `check_numbers_grounded`'s own behavior): a
    result with one invented code and one genuinely valid code must still
    raise, not silently drop just the bad entry and return the rest."""
    with pytest.raises(UngroundedRerankError):
        await rerank_candidates("coffee", _candidates(), model_client=_InventedCodeModelClient())


@pytest.mark.integration
async def test_rerank_output_schema_rejects_out_of_range_relevance_score() -> None:
    with pytest.raises(ValidationError):
        RerankOutput.model_validate(
            {"ranked_candidates": [{"hs_code": "090111", "relevance_score": 1.5}]}
        )


@pytest.mark.integration
async def test_rerank_output_schema_rejects_malformed_hs_code() -> None:
    with pytest.raises(ValidationError):
        RerankOutput.model_validate(
            {"ranked_candidates": [{"hs_code": "not-a-code", "relevance_score": 0.5}]}
        )


@pytest.mark.integration
def test_high_confidence_threshold_is_a_real_probability() -> None:
    # Cheap regression guard against an accidental typo turning the
    # auto-select threshold into something nonsensical (e.g. 75 instead of
    # 0.75) - the boundary that decides auto_selected vs disambiguate.
    assert 0.0 < HIGH_CONFIDENCE_THRESHOLD <= 1.0
