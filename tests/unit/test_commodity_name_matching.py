"""Unit tests for `app.report.facts._commodity_name_matches` — the
heuristic normalized-substring match used to link a domain-specific
source's own commodity/item name (Agmarknet, MSP, FAOSTAT) to this HS6's
taxonomy description, in the absence of a curated crosswalk.
"""

from __future__ import annotations

import pytest

from app.report.facts import _commodity_name_matches, _normalize_commodity_text

pytestmark = pytest.mark.unit


def test_normalize_lowercases_and_strips_trailing_s() -> None:
    assert _normalize_commodity_text("Poppy Seeds") == "poppy seed"


def test_normalize_strips_punctuation() -> None:
    assert _normalize_commodity_text("Oil seeds; poppy seeds, whether or not broken") == (
        "oil seed poppy seed whether or not broken"
    )


def test_matches_faostats_singular_item_against_the_real_taxonomy_description() -> None:
    """The exact real, live-confirmed shape mismatch this heuristic exists
    for: FAOSTAT's real item is "Poppy seed" (singular); this pipeline's
    real taxonomy description for HS6 120791 is "Oil seeds; poppy seeds,
    whether or not broken" (plural, embedded in a longer phrase)."""
    assert _commodity_name_matches("Poppy seed", "Oil seeds; poppy seeds, whether or not broken")


def test_matches_agmarknets_own_commodity_string() -> None:
    assert _commodity_name_matches("Cotton (Medium Staple)", "Cotton (Medium Staple)")


def test_does_not_match_an_unrelated_commodity() -> None:
    assert not _commodity_name_matches(
        "Green Peas", "Oil seeds; poppy seeds, whether or not broken"
    )


def test_does_not_match_when_either_side_is_empty() -> None:
    assert not _commodity_name_matches("", "Poppy seeds")
    assert not _commodity_name_matches("Poppy seeds", "")
