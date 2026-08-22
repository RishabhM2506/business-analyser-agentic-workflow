"""Unit tests for `app.search.normalize.normalize_query`."""

from __future__ import annotations

from typing import TypeVar

import pytest
from pydantic import BaseModel

from app.search.normalize import NormalizedQuery, normalize_query

T = TypeVar("T", bound=BaseModel)


class _FixedModelClient:
    """Fake `ModelClient` that always returns a canned `NormalizedQuery`,
    recording exactly what it was called with."""

    def __init__(self, normalized_text: str) -> None:
        self._normalized_text = normalized_text
        self.system_prompt: str | None = None
        self.user_content: str | None = None
        self.schema: type[BaseModel] | None = None

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        self.system_prompt = system_prompt
        self.user_content = user_content
        self.schema = schema
        return schema.model_validate({"normalized_query": self._normalized_text})


@pytest.mark.unit
async def test_normalize_query_returns_the_models_normalized_text() -> None:
    client = _FixedModelClient("poppy seeds")

    result = await normalize_query("posta dana", model_client=client)

    assert result == "poppy seeds"


@pytest.mark.unit
async def test_normalize_query_calls_the_model_with_the_normalized_query_schema() -> None:
    client = _FixedModelClient("poppy seeds")

    await normalize_query("posta dana", model_client=client)

    assert client.schema is NormalizedQuery
    assert client.user_content == "posta dana"
    assert client.system_prompt is not None and len(client.system_prompt) > 0


@pytest.mark.unit
async def test_normalize_query_strips_surrounding_whitespace_from_the_models_output() -> None:
    client = _FixedModelClient("  poppy seeds  \n")

    result = await normalize_query("posta dana", model_client=client)

    assert result == "poppy seeds"
