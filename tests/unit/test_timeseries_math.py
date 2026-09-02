"""Unit tests for `app.analytics.timeseries_math` — hand-computed fixtures,
no I/O, no model, ever (same discipline as `tests/unit/test_aggregate.py`,
which these were originally written and located in before this module was
extracted, 2026-09-02, Step 4 hardening, so `app.report.facts` could reuse
the identical pure math without importing `app.nodes.aggregate`).
"""

from __future__ import annotations

import pytest

from app.analytics.timeseries_math import cagr, coefficient_of_variation

# --- coefficient of variation -------------------------------------------------


@pytest.mark.unit
def test_coefficient_of_variation_none_for_a_single_data_point() -> None:
    assert coefficient_of_variation({2023: 100.0}) is None


@pytest.mark.unit
def test_coefficient_of_variation_zero_for_identical_values() -> None:
    cov = coefficient_of_variation({2021: 100.0, 2022: 100.0, 2023: 100.0})
    assert cov == pytest.approx(0.0)


@pytest.mark.unit
def test_coefficient_of_variation_none_for_zero_mean_no_divide_by_zero() -> None:
    assert coefficient_of_variation({2022: -100.0, 2023: 100.0}) is None


@pytest.mark.unit
def test_coefficient_of_variation_skips_missing_years() -> None:
    cov = coefficient_of_variation({2021: None, 2022: 100.0, 2023: 100.0})
    assert cov == pytest.approx(0.0)


# --- CAGR ----------------------------------------------------------------------


@pytest.mark.unit
def test_cagr_none_for_a_single_real_year() -> None:
    assert cagr({2021: None, 2022: None, 2023: 100.0}) is None


@pytest.mark.unit
def test_cagr_uses_real_endpoints_not_the_full_declared_year_range() -> None:
    """A series with a gap at the start must use its own earliest real
    year, never silently substitute the missing year with 0."""
    values = {2019: None, 2020: None, 2021: 100.0, 2022: None, 2023: 144.0}
    result = cagr(values)
    # 100 -> 144 over 2 years (2021 -> 2023) = 20% CAGR, not measured from
    # a fabricated 2019/2020 zero.
    assert result == pytest.approx(0.2, abs=1e-9)


@pytest.mark.unit
def test_cagr_none_for_non_positive_start_value() -> None:
    assert cagr({2021: 0.0, 2023: 100.0}) is None
    assert cagr({2021: -50.0, 2023: 100.0}) is None
