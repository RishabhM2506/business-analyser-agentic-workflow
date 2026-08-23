"""Unit tests for `app.pipeline.normalize`'s pure transformation logic —
the fiscal-year-label parser and the crosswalk resolver's UNMAPPED
fallback. The two async normalizer functions themselves are exercised
against a real Postgres in `tests/integration/test_normalize_upsert.py`,
matching this project's established split (see `test_comtrade_mirror.py`
vs. `test_comtrade_mirror_upsert.py`).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.pipeline.normalize import (
    CountryCrosswalk,
    _derive_status,
    _dgcis_fiscal_year_to_period_month,
)

pytestmark = pytest.mark.unit


def test_derive_status_not_reported_when_value_is_none() -> None:
    assert _derive_status(value=None, quantity=Decimal("10")) == "NOT_REPORTED"


def test_derive_status_zero_takes_priority_over_missing_quantity() -> None:
    """A real zero-value flow has no meaningful quantity to be missing -
    ZERO, not QTY_MISSING."""
    assert _derive_status(value=0, quantity=None) == "ZERO"


def test_derive_status_qty_missing_when_value_present_but_quantity_absent() -> None:
    assert _derive_status(value=1000, quantity=None) == "QTY_MISSING"


def test_derive_status_ok_when_both_value_and_quantity_present() -> None:
    assert _derive_status(value=1000, quantity=Decimal("50")) == "OK"


def test_dgcis_fiscal_year_label_parses_the_first_year() -> None:
    year, period_month = _dgcis_fiscal_year_to_period_month("2020 - 2021")

    assert year == 2020
    assert period_month == date(2020, 1, 1)


def test_dgcis_fiscal_year_label_tolerates_no_spaces() -> None:
    year, period_month = _dgcis_fiscal_year_to_period_month("2020-2021")

    assert year == 2020
    assert period_month == date(2020, 1, 1)


def test_dgcis_fiscal_year_label_raises_on_malformed_input() -> None:
    """A normalizer silently inventing a date would be a worse failure
    than a loud one - never guess."""
    with pytest.raises(ValueError):
        _dgcis_fiscal_year_to_period_month("not a year")


def test_crosswalk_resolves_a_known_name() -> None:
    crosswalk = CountryCrosswalk(by_dgcis_name={"TURKEY": "792"})

    assert crosswalk.resolve("TURKEY") == "792"


def test_crosswalk_falls_back_to_unmapped_for_an_unknown_name() -> None:
    """Never a silent drop, never a guessed code - the documented
    ref_country_crosswalk policy (docs/PLAN.md §4)."""
    crosswalk = CountryCrosswalk(by_dgcis_name={"TURKEY": "792"})

    assert crosswalk.resolve("RURITANIA") == "UNMAPPED"
