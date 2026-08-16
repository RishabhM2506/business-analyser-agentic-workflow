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


def _production_shape_table() -> TradeTable:
    """A table at the *real* production shape: 10 ranked rows (`TOP_N_PARTNERS`,
    app/nodes/aggregate.py), 5 years — not the small 2-row fixture `_table()`
    below, which only exercises ranks 1-2 and masks how much wider the
    guardrail's hole was at the real shape (finding B7/AWR-02: every bare
    integer 1-10 was previously "grounded" purely because production tables
    always have up to 10 ranks and a 5-year window, regardless of context)."""
    years = [2019, 2020, 2021, 2022, 2023]
    rows = [
        CountryRow(
            partner_country=f"Country{rank}",
            partner_code=str(100 + rank),
            values_by_year={year: 1000.0 * rank + year for year in years},
            cumulative_5yr=float(5000 * rank),
            rank=rank,
        )
        for rank in range(1, 11)
    ]
    return TradeTable(
        unit="USD",
        years=years,
        years_finalized=years,
        excluded_partner_codes=["0"],
        rows=rows,
    )


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


# --- check_numbers_grounded: hs_code grounding ------------------------------
# `summarize.py` puts "HS code 010121" in the model's own prompt, and a
# leading-zero HS code reads back as a plain number once float-parsed
# ("010121" -> 10121.0) - regression coverage for the bug where a summary
# legitimately mentioning the code it's analyzing was rejected as ungrounded
# on every single run (mock and real), because 10121 was never in
# `_flatten_table_numbers`'s output.


@pytest.mark.unit
def test_hs_code_number_is_grounded_when_supplied() -> None:
    table = _table()
    prose = "For HS code 010121, imports totaled 490.0."
    assert check_numbers_grounded(prose, table, hs_code="010121") is True
    assert check_numbers_grounded(prose, table, hs_code=None) is False


@pytest.mark.unit
def test_hs_code_grounding_does_not_widen_to_unrelated_numbers() -> None:
    # Grounding "10121" must not accidentally also ground some other
    # fabricated number that happens to be nearby in value.
    table = _table()
    prose = "For HS code 010121, imports somehow reached 10122."
    assert check_numbers_grounded(prose, table, hs_code="010121") is False
    assert find_ungrounded_numbers(prose, table, hs_code="010121") == [10122.0]


@pytest.mark.unit
def test_hs_code_grounding_ignores_non_numeric_or_absent_code() -> None:
    table = _table()
    # A malformed/absent hs_code must not raise - just contributes nothing
    # extra to the grounded set (validated separately by
    # `check_hs_code_allowlisted` before this ever runs in the real graph).
    prose = "Trade grew between 2019 and 2023."
    assert check_numbers_grounded(prose, table, hs_code="not-a-code") is True
    assert check_numbers_grounded(prose, table, hs_code="") is True


# --- check_numbers_grounded: production-shape regression (finding B7/AWR-02) -
# Reproduces AWR-02's exact exploit at the *real* production table shape (10
# rows, 5 years - `_production_shape_table()`), not the small 2-row `_table()`
# fixture used everywhere else in this file, which happens to mask the bug:
# with only 2 ranked rows, "7" and "9" were never in the old pooled
# rank/count set at all, so the exploit wouldn't have reproduced against
# `_table()` even before the fix.


@pytest.mark.unit
def test_awr02_exploit_sentence_now_fails_at_production_table_shape() -> None:
    """The exact adversarial sentence AWR-02 demonstrated: every number in
    it is fabricated (no derived figure was ever computed by aggregate.py),
    yet the pre-fix guardrail reported zero offending numbers because 1-10
    was unconditionally "grounded" via pooled rank/count values at this
    table shape. Must now fail, flagging every fabricated number."""
    table = _production_shape_table()
    prose = (
        "Imports grew approximately 8% while exports declined 6%. "
        "Market concentration increased by about 3 points, and the top country added "
        "4 new product lines worth an estimated 7 million versus 9 million last cycle."
    )
    assert check_numbers_grounded(prose, table) is False
    offenders = set(find_ungrounded_numbers(prose, table))
    assert offenders == {8.0, 6.0, 3.0, 4.0, 7.0, 9.0}


@pytest.mark.unit
@pytest.mark.parametrize(
    "prose",
    [
        "Imports were up 8% year over year.",
        "Trade volume grew by 15% compared to the prior period.",
        "The top partner's share declined by 6 percentage points.",
        "Growth of 3pp was observed across all partners.",
    ],
)
def test_percentage_or_points_figures_always_rejected_regardless_of_value(prose: str) -> None:
    """A number immediately adjacent to a %/percent/points/pp suffix must be
    rejected unconditionally - these are definitionally derived figures the
    model must never produce (prompts/summarize.md's own "Hard rules"), so
    no accidental collision with a grounded value should ever save one."""
    table = _production_shape_table()
    assert check_numbers_grounded(prose, table) is False


@pytest.mark.unit
def test_verb_by_prefix_figures_always_rejected_regardless_of_value() -> None:
    table = _production_shape_table()
    prose = "Exports increased by 5 compared to last year."
    # 5 collides with len(table.years) (a real structural number), but
    # "increased by" is a computed-delta assertion, not a structural
    # reference - must still fail even though the bare digit is in range.
    assert check_numbers_grounded(prose, table) is False


@pytest.mark.unit
def test_bare_small_integer_without_structural_context_requires_real_grounding() -> None:
    """A bare integer in the 1-10 range that ISN'T adjacent to the narrow
    structural allowlist (top/rank/no./#/years/partners/countries) must be
    checked like any other number, not waved through just because it's
    small - the core of the AWR-02 exploit."""
    table = _production_shape_table()
    # 9 is a real rank (Country9) but nothing here refers to a rank or a
    # count - it reads as a fabricated raw figure.
    prose = "A total of 9 was recorded last cycle."
    assert check_numbers_grounded(prose, table) is False
    assert find_ungrounded_numbers(prose, table) == [9.0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "prose",
    [
        "Over the past 5 years, the top 10 partners accounted for all recorded trade.",
        "The partner ranked 1 and the partner ranked 2 both placed among the top 10 partners.",
        "The #1 partner led, followed closely by the #2 and #3 partners.",
        "Across all 10 countries and 5 years, trade was broadly distributed.",
    ],
)
def test_structural_allowlist_phrasing_still_passes_at_production_shape(prose: str) -> None:
    """Legitimate structural references (ranks, the row count, the year
    count) adjacent to the narrow allowlist must still pass - the fix
    narrows *when* a structural number counts as grounded, it doesn't
    remove structural grounding altogether."""
    table = _production_shape_table()
    assert check_numbers_grounded(prose, table) is True


# --- extract_numbers / check_numbers_grounded: hyphenated ranges (M1/AWR-03) -


@pytest.mark.unit
def test_extract_numbers_hyphenated_range_reads_as_two_positive_numbers() -> None:
    # Before the fix: "2019-2023" misparsed as 2019 followed by *negative*
    # 2023, because the leading `-?` had no lookbehind guarding it from a
    # hyphen directly between two digit runs.
    assert extract_numbers("2019-2023") == [2019.0, 2023.0]


@pytest.mark.unit
def test_extract_numbers_still_reads_genuine_negative_numbers() -> None:
    # The lookbehind fix must not regress genuinely negative numbers - only
    # a hyphen immediately preceded by a digit is treated as a separator.
    assert extract_numbers("value was -3.1 percent") == [-3.1]
    assert extract_numbers("-5 to -10 range") == [-5.0, -10.0]


@pytest.mark.unit
def test_grounded_summary_using_hyphenated_year_range_passes() -> None:
    """Regression for the exact false-positive AWR-03 demonstrated: a real
    Gemini call is entirely likely to write a year range as "2019-2023"
    (no spaces) rather than the spelled-out "between 2019 and 2023" the
    other year-range test in this file uses - that must not be rejected as
    if it had fabricated a number."""
    table = _table()
    prose = "Imports rose steadily over 2019-2023, led by the United States at 490.0."
    assert check_numbers_grounded(prose, table) is True


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
