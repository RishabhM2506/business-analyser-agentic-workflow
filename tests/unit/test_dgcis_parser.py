"""Unit tests for `app.pipeline.dgcis.parse_annual_country_response`, run
against the real, committed fixture (`tests/fixtures/dgcis/README.md`) —
never a live call. The fixture values are independently known from the
live investigation that produced it (`docs/PLAN.md` §1, `docs/BUILD-LOG.md`),
not just re-derived from the parser under test.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from app.pipeline.dgcis import parse_annual_country_response

pytestmark = pytest.mark.unit

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "dgcis"
    / "poppy_seed_turkey_import_annual.html"
)


def _fixture_html() -> str:
    return _FIXTURE_PATH.read_text(encoding="utf-8")


def test_parses_country_hs8_description_and_unit() -> None:
    record = parse_annual_country_response(_fixture_html())

    assert record is not None
    assert record.country == "TURKEY"
    assert record.hs8 == "12079100"
    assert record.description == "POPPY SEEDS W/N BROKEN"
    assert record.unit == "KGS"


def test_parses_the_full_five_year_annual_series() -> None:
    record = parse_annual_country_response(_fixture_html())

    assert record is not None
    assert record.values_by_year == {
        "2020 - 2021": Decimal("4.91"),
        "2021 - 2022": Decimal("0.00"),
        "2022 - 2023": Decimal("424.66"),
        "2023 - 2024": Decimal("0.00"),
        "2024 - 2025": Decimal("0.00"),
    }


def test_parses_value_type_and_report_date_without_the_context_row_trailer() -> None:
    """Regression test for a real bug: the header row's own '...Values in
    ₹ Crore' trailer text was originally bleeding into the parsed
    report_date field (fixed by only matching the real data row's own
    label cell, not any row whose text happens to contain 'Values in')."""
    record = parse_annual_country_response(_fixture_html())

    assert record is not None
    assert record.value_type == "₹ Crore"
    assert record.report_date == "23 Aug 2026"
    assert "Values in" not in record.report_date


def test_returns_none_for_html_with_no_matching_table() -> None:
    assert parse_annual_country_response("<html><body>no data here</body></html>") is None
