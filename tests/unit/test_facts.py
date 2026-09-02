"""Unit tests for `app.report.facts`'s pure helpers — the worst-status
ordering and the real partner-name CSV lookup. `assemble_facts` itself is
exercised against a real Postgres in `tests/integration/test_facts.py`.
"""

from __future__ import annotations

import pytest

from app.analytics.timeseries_math import cagr, coefficient_of_variation
from app.report.facts import (
    AllOtherPartnersFact,
    AnnualSeriesYear,
    _display_name,
    _overall_value_series,
    _partner_value_series,
    _RankingRow,
    _worst_status,
)

pytestmark = pytest.mark.unit


def test_worst_status_ok_beats_nothing() -> None:
    assert _worst_status(["OK"]) == "OK"


def test_worst_status_fetch_failed_dominates_everything() -> None:
    assert _worst_status(["OK", "ZERO", "FETCH_FAILED", "QTY_MISSING"]) == "FETCH_FAILED"


def test_worst_status_qty_missing_beats_zero() -> None:
    assert _worst_status(["ZERO", "QTY_MISSING"]) == "QTY_MISSING"


def test_display_name_resolves_a_real_partner_code() -> None:
    """792 is Turkey - the real, live-verified canonical-scenario code
    (data/comtrade-partner-areas.csv)."""
    assert _display_name("792") == "Türkiye"


def test_display_name_passes_through_the_all_partners_sentinel() -> None:
    assert _display_name("ALL_PARTNERS") == "ALL_PARTNERS"


def test_display_name_surfaces_the_real_country_name_for_an_unmapped_code() -> None:
    assert _display_name("UNMAPPED:RURITANIA") == "RURITANIA (unmapped)"


def test_display_name_falls_back_to_the_code_for_an_unknown_code() -> None:
    assert _display_name("999999") == "999999"


# --- _partner_value_series / _overall_value_series (Step 4 hardening, 2026-09-02) --


def _row(
    *, partner_country_code: str, value_inr_paise: int | None, rank: int | None = 1
) -> _RankingRow:
    return _RankingRow(
        partner_country_code=partner_country_code,
        rank=rank,
        value_inr_paise=value_inr_paise,
        status="OK",
    )


def test_partner_value_series_present_every_year() -> None:
    years = [2021, 2022, 2023]
    rankings_by_year = {
        2021: [_row(partner_country_code="792", value_inr_paise=100)],
        2022: [_row(partner_country_code="792", value_inr_paise=200)],
        2023: [_row(partner_country_code="792", value_inr_paise=300)],
    }
    series = _partner_value_series(rankings_by_year, years=years)
    assert series == {"792": {2021: 100.0, 2022: 200.0, 2023: 300.0}}


def test_partner_value_series_reads_none_for_a_year_the_partner_is_absent_from() -> None:
    """A partner missing from some years must read `None` there, never be
    silently absent from the dict entirely."""
    years = [2021, 2022, 2023]
    rankings_by_year = {
        2021: [_row(partner_country_code="792", value_inr_paise=100)],
        2022: [],  # no rows for any partner this year
        2023: [_row(partner_country_code="792", value_inr_paise=300)],
    }
    series = _partner_value_series(rankings_by_year, years=years)
    assert series == {"792": {2021: 100.0, 2022: None, 2023: 300.0}}


def test_partner_value_series_single_real_year_feeds_none_cagr_and_cov() -> None:
    """A partner with only one real year in a multi-year window must
    produce a series that CAGR/CoV both honestly read as `None` from -
    proving the pivot's shape, not just the pure math functions in
    isolation."""
    years = [2021, 2022, 2023]
    rankings_by_year = {
        2021: [],
        2022: [_row(partner_country_code="792", value_inr_paise=200)],
        2023: [],
    }
    series = _partner_value_series(rankings_by_year, years=years)["792"]
    assert series == {2021: None, 2022: 200.0, 2023: None}
    assert cagr(series) is None
    assert coefficient_of_variation(series) is None


def _annual_year(year: int, total_inr_paise: int | None) -> AnnualSeriesYear:
    return AnnualSeriesYear(
        year=year,
        flow="import",
        total_inr_paise=total_inr_paise,
        status="OK",
        partners=[],
        all_other_partners=AllOtherPartnersFact(value_inr_paise=0, status="OK"),
    )


def test_overall_value_series_reads_none_for_a_year_with_no_total_and_feeds_real_cagr() -> None:
    annual_series = [
        _annual_year(2021, 100),
        _annual_year(2022, None),  # a real gap, mid-window
        _annual_year(2023, 144),
    ]
    series = _overall_value_series(annual_series)
    assert series == {2021: 100.0, 2022: None, 2023: 144.0}
    # 100 -> 144 over 2 years (2021 -> 2023) = 20% CAGR, using the real
    # endpoints on either side of the None gap, not a fabricated 0.
    assert cagr(series) == pytest.approx(0.2, abs=1e-9)
