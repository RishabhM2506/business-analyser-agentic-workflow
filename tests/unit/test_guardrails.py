"""Unit tests for `app.guardrails` — the deterministic number-grounding
check (docs/PLAN.md §7, master brief §2.2/§8). Adversarial cases are the
point: a summary with any fabricated number must fail.
"""

from __future__ import annotations

import pytest

from app.guardrails import (
    check_hs_code_allowlisted,
    check_numbers_grounded,
    extract_numbers,
    find_ungrounded_numbers,
)
from app.schemas.response import CountryRow, TradeTable


def _table(rows: list[CountryRow] | None = None) -> TradeTable:
    return TradeTable(
        unit="USD",
        years=[2019, 2020, 2021, 2022, 2023],
        years_finalized=[2019, 2020, 2021, 2022],
        excluded_partner_codes=["0"],
        rows=(
            rows
            if rows is not None
            else [
                CountryRow(
                    partner_country="United States",
                    partner_code="842",
                    values_by_year={
                        2019: 100.0,
                        2020: None,
                        2021: 120.0,
                        2022: 130.5,
                        2023: 1992455.942,
                    },
                    cumulative_5yr=490.0,
                    rank=1,
                ),
                CountryRow(
                    partner_country="United Kingdom",
                    partner_code="826",
                    values_by_year={2019: 50.0, 2020: 60.0, 2021: 70.0, 2022: 80.0, 2023: 90.0},
                    cumulative_5yr=350.0,
                    rank=2,
                ),
            ]
        ),
    )


# --- extract_numbers ---------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("no numbers here", []),
        ("USD 1,234,567.89 total", [1234567.89]),
        ("2019, 2020, and 2021 saw growth", [2019.0, 2020.0, 2021.0]),
        ("value was -3.1 percent", [-3.1]),
        ("42", [42.0]),
        ("$1,992,456", [1992456.0]),
    ],
)
def test_extract_numbers(text: str, expected: list[float]) -> None:
    assert extract_numbers(text) == expected


# --- check_numbers_grounded: should PASS -------------------------------------


@pytest.mark.unit
def test_grounded_summary_referencing_exact_table_values_passes() -> None:
    table = _table()
    prose = (
        "The United States was the top partner with a 5-year cumulative value of 490.0, "
        "while the United Kingdom ranked 2 with 350.0."
    )
    assert check_numbers_grounded(prose, table) is True


@pytest.mark.unit
def test_grounded_summary_tolerates_whole_unit_rounding() -> None:
    table = _table()
    # 1992455.942 rounds to 1992456 - a model copying the rounded display
    # value must not fail the check over sub-dollar precision.
    prose = "In 2023, the United States recorded 1,992,456 in value."
    assert check_numbers_grounded(prose, table) is True


@pytest.mark.unit
def test_grounded_summary_referencing_years_passes() -> None:
    table = _table()
    prose = "Trade grew between 2019 and 2023."
    assert check_numbers_grounded(prose, table) is True


@pytest.mark.unit
def test_grounded_summary_referencing_structural_constants_passes() -> None:
    table = _table()
    # "5 years" / "top 2" are structural facts about the table (len(years)==5,
    # len(rows)==2 here) - true by construction, not fabricated data.
    prose = "Over the past 5 years, the top 2 partners dominated trade."
    assert check_numbers_grounded(prose, table) is True


@pytest.mark.unit
def test_empty_prose_is_trivially_grounded() -> None:
    assert check_numbers_grounded("No numeric claims made.", _table()) is True


# --- check_numbers_grounded: should FAIL (adversarial) -----------------------


@pytest.mark.unit
def test_fabricated_large_number_fails() -> None:
    table = _table()
    prose = "The United States imported goods worth 99,000,000 in 2023."
    assert check_numbers_grounded(prose, table) is False
    assert find_ungrounded_numbers(prose, table) == [99000000.0]


@pytest.mark.unit
def test_fabricated_percentage_growth_figure_fails() -> None:
    table = _table()
    # A derived growth-rate the model computed itself - never a raw table
    # cell, so it can never legitimately be "grounded" even if plausible.
    prose = "Imports from the United States grew by 37% over the period."
    assert check_numbers_grounded(prose, table) is False


@pytest.mark.unit
def test_slightly_altered_real_number_fails() -> None:
    table = _table()
    # 491.0 is NOT 490.0 (the real cumulative_5yr) - a materially different
    # (if close-looking) number must still be caught.
    prose = "The United States cumulative value over 5 years was 491."
    assert check_numbers_grounded(prose, table) is False


@pytest.mark.unit
def test_number_from_a_different_hs_codes_table_fails() -> None:
    # Simulates a summary that leaked a number belonging to a different
    # query/table entirely - still must fail against *this* table.
    table = _table()
    prose = "Value reached 555555 in the latest year."
    assert check_numbers_grounded(prose, table) is False


@pytest.mark.unit
def test_multiple_tables_union_grounds_correctly() -> None:
    imports = _table()
    exports = _table(
        rows=[
            CountryRow(
                partner_country="Germany",
                partner_code="276",
                values_by_year={2019: 10.0, 2020: 20.0, 2021: 30.0, 2022: 40.0, 2023: 50.0},
                cumulative_5yr=150.0,
                rank=1,
            )
        ]
    )
    prose = "Germany's exports reached a 5-year cumulative value of 150.0."
    assert check_numbers_grounded(prose, imports, exports) is True
    # But a number that belongs to neither table still fails.
    assert check_numbers_grounded("Germany's exports reached 12345.", imports, exports) is False


@pytest.mark.unit
def test_find_ungrounded_numbers_returns_only_offenders() -> None:
    table = _table()
    prose = "The real value was 490.0 but someone invented 77777 too."
    assert find_ungrounded_numbers(prose, table) == [77777.0]


# --- check_hs_code_allowlisted ------------------------------------------------


@pytest.mark.integration  # reads the checked-in taxonomy CSV from disk
def test_check_hs_code_allowlisted_true_for_real_hs6_code() -> None:
    assert check_hs_code_allowlisted("010121") is True


@pytest.mark.integration
def test_check_hs_code_allowlisted_false_for_unknown_code() -> None:
    # Not "999999": that's a real HS6 entry in the checked-in taxonomy
    # ("Commodities not specified according to kind", section "TOTAL",
    # data/harmonized-system.csv line 6940) - a genuine catch-all code, not
    # an absent one. "000000" is genuinely absent from the CSV.
    assert check_hs_code_allowlisted("000000") is False


@pytest.mark.integration
def test_check_hs_code_allowlisted_false_for_non_hs6_level_code() -> None:
    # "01" is a real code in the taxonomy, but at level 2 (a chapter), not
    # level 6 (a sub-heading) - the allowlist is HS6-specific.
    assert check_hs_code_allowlisted("01") is False
