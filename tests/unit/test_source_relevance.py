"""Unit tests for `app.report.source_relevance.is_agriculture_relevant` —
the chapter rule (no model call), the boundary-chapter model fallback,
and the budget-check-only-when-a-real-call-happens sequencing.
"""

from __future__ import annotations

from typing import TypeVar

import pytest
from pydantic import BaseModel

from app.budget import BudgetExceededError, BudgetTracker
from app.report.source_relevance import AgricultureRelevanceCheck, is_agriculture_relevant

pytestmark = pytest.mark.unit

T = TypeVar("T", bound=BaseModel)


class _FixedModelClient:
    def __init__(self, *, is_agricultural: bool) -> None:
        self._is_agricultural = is_agricultural
        self.call_count = 0
        self.user_content: str | None = None
        self.schema: type[BaseModel] | None = None

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        self.call_count += 1
        self.user_content = user_content
        self.schema = schema
        return schema.model_validate({"is_agricultural": self._is_agricultural})


def _tracker(*, max_calls: int = 100) -> BudgetTracker:
    return BudgetTracker(max_calls_per_thread=max_calls, max_calls_per_day=max_calls)


async def test_chapter_01_is_relevant_with_no_model_call() -> None:
    client = _FixedModelClient(is_agricultural=False)  # would say no, if ever asked

    result = await is_agriculture_relevant(
        "010121",
        commodity_description="Live horses",
        model_client=client,
        budget_tracker=_tracker(),
        thread_id="t1",
        tenant_id="default",
    )

    assert result is True
    assert client.call_count == 0  # the chapter rule alone answered this


async def test_chapter_12_poppy_seeds_is_relevant_with_no_model_call() -> None:
    client = _FixedModelClient(is_agricultural=False)

    result = await is_agriculture_relevant(
        "120791",
        commodity_description="Oil seeds; poppy seeds, whether or not broken",
        model_client=client,
        budget_tracker=_tracker(),
        thread_id="t1",
        tenant_id="default",
    )

    assert result is True
    assert client.call_count == 0


async def test_chapter_25_cement_is_not_applicable_with_no_model_call() -> None:
    client = _FixedModelClient(is_agricultural=True)  # would say yes, if ever asked

    result = await is_agriculture_relevant(
        "252329",
        commodity_description="Cement; portland, other than white",
        model_client=client,
        budget_tracker=_tracker(),
        thread_id="t1",
        tenant_id="default",
    )

    assert result is False
    assert client.call_count == 0  # the chapter rule alone answered this


async def test_boundary_chapter_calls_the_model() -> None:
    """Chapter 52 (cotton) is a real boundary chapter - the rule alone
    can't decide, so the one real model call happens."""
    client = _FixedModelClient(is_agricultural=True)

    result = await is_agriculture_relevant(
        "520100",
        commodity_description="Cotton, not carded or combed",
        model_client=client,
        budget_tracker=_tracker(),
        thread_id="t1",
        tenant_id="default",
    )

    assert result is True
    assert client.call_count == 1
    assert client.schema is AgricultureRelevanceCheck


async def test_boundary_chapter_can_resolve_to_not_relevant() -> None:
    client = _FixedModelClient(is_agricultural=False)

    result = await is_agriculture_relevant(
        "520100",
        commodity_description="Synthetic cotton-blend yarn",
        model_client=client,
        budget_tracker=_tracker(),
        thread_id="t1",
        tenant_id="default",
    )

    assert result is False


async def test_boundary_chapter_checks_budget_before_the_model_call() -> None:
    client = _FixedModelClient(is_agricultural=True)
    exhausted_tracker = _tracker(max_calls=0)

    with pytest.raises(BudgetExceededError):
        await is_agriculture_relevant(
            "520100",
            commodity_description="Cotton",
            model_client=client,
            budget_tracker=exhausted_tracker,
            thread_id="t1",
            tenant_id="default",
        )

    assert client.call_count == 0  # never reached - budget failed closed first


async def test_non_boundary_non_agriculture_chapter_never_checks_budget() -> None:
    """The common case (an ordinary industrial-goods chapter) must cost
    nothing at all - not even a budget-tracker call."""
    client = _FixedModelClient(is_agricultural=True)
    exhausted_tracker = _tracker(max_calls=0)  # would raise if ever touched

    result = await is_agriculture_relevant(
        "854231",
        commodity_description="Electronic integrated circuits",
        model_client=client,
        budget_tracker=exhausted_tracker,
        thread_id="t1",
        tenant_id="default",
    )

    assert result is False
