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

from app.pipeline.dgcis import parse_annual_country_response, parse_monthly_response

pytestmark = pytest.mark.unit

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "dgcis"
_FIXTURE_PATH = _FIXTURE_DIR / "poppy_seed_turkey_import_annual.html"


def _fixture_html() -> str:
    return _FIXTURE_PATH.read_text(encoding="utf-8")


def _monthly_fixture_html(name: str) -> str:
    return (_FIXTURE_DIR / name).read_text(encoding="utf-8")


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


def test_parses_a_real_finalized_monthly_value() -> None:
    """ "(R)" = Revised/Final - a fully finalized past month. Real fixture:
    June 2022, ₹166.50 crore."""
    cell = parse_monthly_response(
        _monthly_fixture_html("poppy_seed_monthly_import_jun2022_value.html"), hs8="12079100"
    )

    assert cell is not None
    assert cell.value == Decimal("166.50")
    assert cell.marker == "R"
    assert cell.unit is None  # value-flavored response has no UNIT column


def test_parses_a_real_finalized_monthly_quantity() -> None:
    """A quantity-flavored response (report_value="2") inserts a real
    extra UNIT header a value-flavored one doesn't have - the column
    index must shift accordingly, not stay fixed."""
    cell = parse_monthly_response(
        _monthly_fixture_html("poppy_seed_monthly_import_jun2022_quantity.html"), hs8="12079100"
    )

    assert cell is not None
    assert cell.value == Decimal("6347970")
    assert cell.marker == "R"
    assert cell.unit == "KGS"


def test_parses_a_real_flash_provisional_month() -> None:
    """ "(F)" = Flash/provisional - published but subject to later
    revision. Real fixture: June 2026."""
    cell = parse_monthly_response(
        _monthly_fixture_html("poppy_seed_monthly_import_jun2026_flash.html"), hs8="12079100"
    )

    assert cell is not None
    assert cell.marker == "F"


def test_parses_a_real_not_yet_published_month() -> None:
    """ "(A)" = Advance - the month hasn't been published yet. Real
    fixture: August 2026, the literal current month at capture time."""
    cell = parse_monthly_response(
        _monthly_fixture_html("poppy_seed_monthly_import_aug2026_not_yet_published.html"),
        hs8="12079100",
    )

    assert cell is not None
    assert cell.marker == "A"


def test_monthly_parser_returns_none_for_html_with_no_matching_table() -> None:
    assert parse_monthly_response("<html><body>no data here</body></html>", hs8="12079100") is None


def test_monthly_parser_returns_none_for_an_unmatched_hs8() -> None:
    cell = parse_monthly_response(
        _monthly_fixture_html("poppy_seed_monthly_import_jun2022_value.html"), hs8="99999999"
    )

    assert cell is None
