"""Tests for `app.knowledge.provider` against the real checked-in taxonomy
CSVs (`data/harmonized-system.csv`, `data/harmonized-system-sections.csv`) —
real file I/O, no network, no model (docs/PLAN.md §7)."""

from __future__ import annotations

import pytest

from app.knowledge.provider import (
    StaticKnowledgeProvider,
    build_taxonomy_text,
    get_taxonomy_entry,
    is_known_hs6_code,
)


@pytest.mark.integration
def test_is_known_hs6_code_true_for_real_code() -> None:
    assert is_known_hs6_code("010121") is True


@pytest.mark.integration
def test_is_known_hs6_code_false_for_absent_code() -> None:
    assert is_known_hs6_code("000000") is False


@pytest.mark.integration
def test_is_known_hs6_code_false_for_non_hs6_level() -> None:
    assert is_known_hs6_code("0101") is False  # a real code, but level 4


@pytest.mark.integration
def test_get_taxonomy_entry_returns_full_row() -> None:
    entry = get_taxonomy_entry("010121")
    assert entry is not None
    assert entry.description == "Horses; live, pure-bred breeding animals"
    assert entry.section == "I"
    assert entry.parent == "0101"
    assert entry.level == "6"


@pytest.mark.integration
def test_get_taxonomy_entry_none_for_unknown_code() -> None:
    assert get_taxonomy_entry("000000") is None


@pytest.mark.integration
def test_build_taxonomy_text_includes_description_breadcrumb_and_section() -> None:
    text = build_taxonomy_text("010121")
    assert text is not None
    assert "Horses; live, pure-bred breeding animals" in text
    assert "Horses, asses, mules and hinnies; live" in text  # level-4 parent
    assert "Animals; live" in text  # level-2 grandparent
    assert "Section I" in text
    assert "Live animals; animal products" in text  # section name, capitalized


@pytest.mark.integration
def test_build_taxonomy_text_none_for_unknown_code() -> None:
    assert build_taxonomy_text("000000") is None


@pytest.mark.integration
async def test_static_knowledge_provider_retrieve_success() -> None:
    provider = StaticKnowledgeProvider()
    text = await provider.retrieve("160220")
    assert "160220" in text


@pytest.mark.integration
async def test_static_knowledge_provider_retrieve_raises_for_unknown_code() -> None:
    provider = StaticKnowledgeProvider()
    with pytest.raises(KeyError):
        await provider.retrieve("000000")
