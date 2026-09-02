"""Unit tests for `app.pipeline.normalize`'s pure transformation logic —
the fiscal-year-label parser and the crosswalk resolver's per-country
UNMAPPED fallback. The two async normalizer functions themselves are exercised
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
    _to_paise,
)

pytestmark = pytest.mark.unit


def test_to_paise_rounds_half_up_instead_of_truncating() -> None:
    """Architect-review regression (2026-08-26): the old code used Python's
    `int()`, which truncates toward zero — a systematic downward bias on
    every converted row. $1.2349 (Decimal, matching the real
    Numeric(18, 2)-sourced values this is always fed in practice) -> 123.49
    paise; truncation would give 123, half-up rounding gives 123 too (below
    the midpoint) - use a genuinely midpoint-or-above case instead to prove
    rounding, not truncation, is happening."""
    # 100.005 * 100 = 10000.5 - exactly the rounding-mode boundary: half-up
    # rounds to 10001, truncation would give 10000.
    assert _to_paise(Decimal("100.005")) == 10001


def test_to_paise_matches_plain_int_for_an_exact_value() -> None:
    assert _to_paise(Decimal("20.00")) == 2000


def test_to_paise_rounds_a_real_fx_converted_value() -> None:
    # A real shape this is actually fed: value_usd * rate, both Decimals.
    value_usd = Decimal("1000.00")
    rate = Decimal("83.12345")
    assert _to_paise(value_usd * rate) == 8312345  # 1000 * 83.12345 * 100, exact here


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
    ref_country_crosswalk policy (docs/PLAN.md §4). The real name is
    embedded in the sentinel, not collapsed to a bare 'UNMAPPED' constant
    - a real bug found live: two distinct unmapped countries sharing one
    flat sentinel collide on normalized_trade_flows' unique key."""
    crosswalk = CountryCrosswalk(by_dgcis_name={"TURKEY": "792"})

    assert crosswalk.resolve("RURITANIA") == "UNMAPPED:RURITANIA"


def test_crosswalk_gives_distinct_unmapped_countries_distinct_codes() -> None:
    """Regression test for the real collision bug: two different real,
    unmapped countries must never resolve to the same code, or a bulk
    upsert with both in the same batch raises a real
    CardinalityViolationError (found live during the first ~250-country
    DGCIS run, with genuinely many distinct unmapped countries at once)."""
    crosswalk = CountryCrosswalk(by_dgcis_name={})

    assert crosswalk.resolve("RURITANIA") != crosswalk.resolve("FREEDONIA")
