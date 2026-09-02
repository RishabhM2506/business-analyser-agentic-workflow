"""Exhaustive unit tests for `app.nodes.aggregate`'s pure functions
(docs/PLAN.md §7: "the easiest thing in the whole system to test well ...
do it properly") — hand-computed `ComtradeRecord` fixtures, no fixture
files, no I/O beyond the tiny cached reference-table lookups
`is_aggregate_partner_code` makes (per docs/PLAN.md §7's own table, this
module is classified `unit`, not `integration`).
"""

from __future__ import annotations

import pytest

from app.nodes.aggregate import (
    HIGH_VOLATILITY_COV_THRESHOLD,
    REST_OF_WORLD_PARTNER_CODE,
    _compute_hhi,
    _rest_of_world_row,
    _world_total_reconciles,
    aggregate,
    build_trade_table,
    compute_trade_balance,
    find_excluded_partner_codes,
    flag_years_finalized,
    flag_years_no_data,
    rank_top_partners,
    strip_aggregate_partners,
)
from app.schemas.query import TradeQuery
from app.schemas.response import CountryRow
from app.state import AnalysisState, FetchIssue
from app.tools.comtrade_client import ComtradeRecord

YEARS = [2019, 2020, 2021, 2022, 2023]


def _record(
    *,
    partner_code: str,
    partner_country: str,
    year: int,
    value: float | None,
    is_provisional: bool = False,
    flow: str = "import",
    hs_code: str = "010121",
) -> ComtradeRecord:
    return ComtradeRecord(
        hs_code=hs_code,
        flow=flow,  # type: ignore[arg-type]
        partner_code=partner_code,
        partner_country=partner_country,
        year=year,
        value=value,
        is_provisional=is_provisional,
    )


# --- strip_aggregate_partners / find_excluded_partner_codes ------------------


@pytest.mark.unit
def test_strip_aggregate_partners_removes_world_and_nes_codes() -> None:
    records = [
        _record(partner_code="0", partner_country="World", year=2023, value=1000.0),
        _record(partner_code="490", partner_country="Other Asia, nes", year=2023, value=200.0),
        _record(partner_code="842", partner_country="USA", year=2023, value=300.0),
    ]
    kept = strip_aggregate_partners(records)
    assert [r.partner_code for r in kept] == ["842"]


@pytest.mark.unit
def test_strip_aggregate_partners_empty_input() -> None:
    assert strip_aggregate_partners([]) == []


@pytest.mark.unit
def test_strip_aggregate_partners_keeps_all_when_no_aggregates_present() -> None:
    records = [
        _record(partner_code="842", partner_country="USA", year=2023, value=300.0),
        _record(partner_code="826", partner_country="United Kingdom", year=2023, value=100.0),
    ]
    assert strip_aggregate_partners(records) == records


@pytest.mark.unit
def test_find_excluded_partner_codes_only_lists_present_aggregates() -> None:
    records = [
        _record(partner_code="0", partner_country="World", year=2023, value=1000.0),
        _record(partner_code="842", partner_country="USA", year=2023, value=300.0),
    ]
    # "837" (Bunkers) is a real aggregate code but never appears in `records`
    # - it must NOT show up in the transparency list for this result.
    assert find_excluded_partner_codes(records) == ["0"]


@pytest.mark.unit
def test_find_excluded_partner_codes_sorted_and_deduplicated() -> None:
    records = [
        _record(partner_code="490", partner_country="Other Asia, nes", year=2021, value=1.0),
        _record(partner_code="490", partner_country="Other Asia, nes", year=2022, value=2.0),
        _record(partner_code="0", partner_country="World", year=2022, value=3.0),
    ]
    assert find_excluded_partner_codes(records) == ["0", "490"]


@pytest.mark.unit
def test_strip_aggregate_partners_removes_the_reporters_own_code() -> None:
    """M20/PBO-02: India (the fixed reporter, `INDIA_REPORTER_CODE = "699"`)
    live-reproduced on HS 851713 as one of its own top-10 import "trading
    partners," unexplained — a country cannot be its own bilateral partner.
    `data/comtrade-partner-areas.csv` correctly marks "699" as NOT a
    generic aggregate code (`is_aggregate_code=false`, it's a real
    country), so `is_aggregate_partner_code` alone never caught this."""
    from app.tools.comtrade_client import INDIA_REPORTER_CODE

    records = [
        _record(partner_code=INDIA_REPORTER_CODE, partner_country="India", year=2023, value=500.0),
        _record(partner_code="842", partner_country="USA", year=2023, value=300.0),
    ]
    kept = strip_aggregate_partners(records)
    assert [r.partner_code for r in kept] == ["842"]


@pytest.mark.unit
def test_find_excluded_partner_codes_includes_the_reporters_own_code() -> None:
    from app.tools.comtrade_client import INDIA_REPORTER_CODE

    records = [
        _record(partner_code=INDIA_REPORTER_CODE, partner_country="India", year=2023, value=500.0),
        _record(partner_code="0", partner_country="World", year=2023, value=1000.0),
        _record(partner_code="842", partner_country="USA", year=2023, value=300.0),
    ]
    assert find_excluded_partner_codes(records) == ["0", INDIA_REPORTER_CODE]


@pytest.mark.unit
def test_build_trade_table_strips_reporter_self_reference_before_ranking() -> None:
    """End-to-end regression at the same level PBO-02 reproduced it: the
    reporter's own code must never reach the ranked rows, even after the
    full aggregate pipeline (rank + pivot + completeness flagging)."""
    from app.tools.comtrade_client import INDIA_REPORTER_CODE

    records = [
        _record(
            partner_code=INDIA_REPORTER_CODE,
            partner_country="India",
            year=2023,
            value=999_999.0,  # would rank #1 if not stripped
        ),
        _record(partner_code="842", partner_country="USA", year=2023, value=300.0),
    ]
    table = build_trade_table(records, years=[2023])
    assert [r.partner_country for r in table.rows] == ["USA"]
    assert INDIA_REPORTER_CODE in table.excluded_partner_codes


# --- rank_top_partners ---------------------------------------------------------


@pytest.mark.unit
def test_rank_top_partners_orders_by_cumulative_value_descending() -> None:
    records = [
        _record(partner_code="842", partner_country="USA", year=2023, value=300.0),
        _record(partner_code="826", partner_country="United Kingdom", year=2023, value=100.0),
        _record(partner_code="276", partner_country="Germany", year=2023, value=200.0),
    ]
    rows = rank_top_partners(records, years=[2023])
    assert [r.partner_country for r in rows] == ["USA", "Germany", "United Kingdom"]
    assert [r.rank for r in rows] == [1, 2, 3]


@pytest.mark.unit
def test_rank_top_partners_sums_across_years_for_cumulative_value() -> None:
    records = [
        _record(partner_code="842", partner_country="USA", year=2021, value=100.0),
        _record(partner_code="842", partner_country="USA", year=2022, value=150.0),
        _record(partner_code="842", partner_country="USA", year=2023, value=200.0),
    ]
    rows = rank_top_partners(records, years=[2021, 2022, 2023])
    assert len(rows) == 1
    assert rows[0].cumulative_5yr == pytest.approx(450.0)
    assert rows[0].values_by_year == {2021: 100.0, 2022: 150.0, 2023: 200.0}


@pytest.mark.unit
def test_rank_top_partners_missing_year_is_none_not_zero_or_interpolated() -> None:
    records = [
        _record(partner_code="842", partner_country="USA", year=2021, value=100.0),
        _record(partner_code="842", partner_country="USA", year=2023, value=300.0),
        # no 2022 record at all for USA
    ]
    rows = rank_top_partners(records, years=[2021, 2022, 2023])
    assert rows[0].values_by_year[2022] is None
    # cumulative sums only the two years actually present - not zero-filled.
    assert rows[0].cumulative_5yr == pytest.approx(400.0)


@pytest.mark.unit
def test_rank_top_partners_caps_at_top_n_without_padding() -> None:
    records = [
        _record(
            partner_code=str(code), partner_country=f"Country{code}", year=2023, value=float(code)
        )
        for code in range(1, 16)  # 15 distinct partners
    ]
    rows = rank_top_partners(records, years=[2023], top_n=10)
    assert len(rows) == 10
    # highest value (15) ranked first
    assert rows[0].partner_country == "Country15"
    assert rows[0].rank == 1
    assert rows[-1].rank == 10


@pytest.mark.unit
def test_rank_top_partners_fewer_than_top_n_returns_all_no_padding() -> None:
    records = [
        _record(partner_code="842", partner_country="USA", year=2023, value=300.0),
        _record(partner_code="826", partner_country="United Kingdom", year=2023, value=100.0),
    ]
    rows = rank_top_partners(records, years=[2023], top_n=10)
    assert len(rows) == 2  # not padded to 10


@pytest.mark.unit
def test_rank_top_partners_ties_broken_alphabetically_by_country_name() -> None:
    records = [
        _record(partner_code="1", partner_country="Zambia", year=2023, value=100.0),
        _record(partner_code="2", partner_country="Albania", year=2023, value=100.0),
        _record(partner_code="3", partner_country="Mexico", year=2023, value=100.0),
    ]
    rows = rank_top_partners(records, years=[2023])
    assert [r.partner_country for r in rows] == ["Albania", "Mexico", "Zambia"]


@pytest.mark.unit
def test_rank_top_partners_empty_input_returns_empty_list() -> None:
    assert rank_top_partners([], years=YEARS) == []


@pytest.mark.unit
def test_rank_top_partners_ignores_years_outside_requested_range() -> None:
    records = [
        _record(
            partner_code="842", partner_country="USA", year=2018, value=999.0
        ),  # outside window
        _record(partner_code="842", partner_country="USA", year=2023, value=100.0),
    ]
    rows = rank_top_partners(records, years=[2023])
    assert rows[0].cumulative_5yr == pytest.approx(100.0)
    assert 2018 not in rows[0].values_by_year


@pytest.mark.unit
def test_rank_top_partners_null_value_record_treated_as_missing() -> None:
    records = [_record(partner_code="842", partner_country="USA", year=2023, value=None)]
    rows = rank_top_partners(records, years=[2023])
    assert rows[0].values_by_year[2023] is None
    assert rows[0].cumulative_5yr == 0.0


# --- flag_years_finalized -----------------------------------------------------


@pytest.mark.unit
def test_flag_years_finalized_true_when_all_records_reported() -> None:
    records = [
        _record(
            partner_code="842", partner_country="USA", year=2022, value=100.0, is_provisional=False
        ),
        _record(
            partner_code="826", partner_country="UK", year=2022, value=50.0, is_provisional=False
        ),
    ]
    assert flag_years_finalized(records, years=[2022]) == [2022]


@pytest.mark.unit
def test_flag_years_finalized_false_when_any_record_provisional() -> None:
    records = [
        _record(
            partner_code="842", partner_country="USA", year=2022, value=100.0, is_provisional=False
        ),
        _record(
            partner_code="826", partner_country="UK", year=2022, value=50.0, is_provisional=True
        ),
    ]
    assert flag_years_finalized(records, years=[2022]) == []


@pytest.mark.unit
def test_flag_years_finalized_false_for_year_with_zero_records() -> None:
    # Vacuous truth trap: all([]) is True in Python, but a year nobody
    # reported anything for is NOT "finalized" - it's simply absent.
    assert flag_years_finalized([], years=[2022]) == []


@pytest.mark.unit
def test_flag_years_finalized_handles_mixed_years_independently() -> None:
    records = [
        _record(
            partner_code="842", partner_country="USA", year=2021, value=100.0, is_provisional=False
        ),
        _record(
            partner_code="842", partner_country="USA", year=2022, value=110.0, is_provisional=True
        ),
        _record(
            partner_code="842", partner_country="USA", year=2023, value=120.0, is_provisional=False
        ),
    ]
    assert flag_years_finalized(records, years=[2021, 2022, 2023]) == [2021, 2023]


# --- flag_years_no_data (finding M21/PBO-03) -----------------------------------


@pytest.mark.unit
def test_flag_years_no_data_true_for_year_with_zero_records() -> None:
    # The exact PBO-03 shape: an HS6 code that structurally cannot have
    # records for a year (e.g. it postdates the code's own nomenclature
    # creation) looks identical, at this layer, to "nothing reported yet" -
    # both are zero retained records for the year.
    assert flag_years_no_data([], years=[2021]) == [2021]


@pytest.mark.unit
def test_flag_years_no_data_false_when_any_record_present_even_if_provisional() -> None:
    # A year with at least one record is "provisional", never "no data" -
    # these two states must never overlap.
    records = [
        _record(
            partner_code="842", partner_country="USA", year=2022, value=100.0, is_provisional=True
        )
    ]
    assert flag_years_no_data(records, years=[2022]) == []


@pytest.mark.unit
def test_flag_years_no_data_false_when_fully_finalized() -> None:
    records = [
        _record(
            partner_code="842", partner_country="USA", year=2022, value=100.0, is_provisional=False
        )
    ]
    assert flag_years_no_data(records, years=[2022]) == []


@pytest.mark.unit
def test_flag_years_no_data_disjoint_from_flag_years_finalized() -> None:
    # 2021: zero records (no data). 2022: one provisional record (genuinely
    # provisional). 2023: fully finalized. All three must land in exactly
    # one bucket, never zero and never two.
    records = [
        _record(
            partner_code="842", partner_country="USA", year=2022, value=100.0, is_provisional=True
        ),
        _record(
            partner_code="842", partner_country="USA", year=2023, value=100.0, is_provisional=False
        ),
    ]
    years = [2021, 2022, 2023]
    no_data = flag_years_no_data(records, years=years)
    finalized = flag_years_finalized(records, years=years)
    assert no_data == [2021]
    assert finalized == [2023]
    assert set(no_data) & set(finalized) == set()
    # The genuinely-provisional year (2022) lands in neither bucket - it's
    # exactly the complement the UI still renders as "provisional".
    provisional = [y for y in years if y not in no_data and y not in finalized]
    assert provisional == [2022]


@pytest.mark.unit
def test_flag_years_no_data_excludes_a_year_whose_fetch_itself_failed() -> None:
    """2026-08-20 roadmap decision (live user-reported finding): a year
    with zero records because the *fetch* failed (retries exhausted) is a
    third, distinct case from both `years_finalized`'s complement and a
    genuine zero-records response — asserting we don't actually know
    whether data exists would be dishonest, so it must NOT land in
    `years_no_data` just because `flag_years_no_data`'s usual zero-records
    check can't otherwise tell the two apart."""
    # 2021: genuinely zero records (real "no data"). 2022: also zero
    # records, but only because the fetch for it failed.
    assert flag_years_no_data([], years=[2021, 2022], fetch_failed_years=frozenset({2022})) == [
        2021
    ]


@pytest.mark.unit
def test_flag_years_no_data_fetch_failed_years_defaults_to_empty() -> None:
    # Backward-compatible default: every pre-existing call site (no
    # `fetch_failed_years` argument at all) behaves exactly as before.
    assert flag_years_no_data([], years=[2021]) == [2021]


# --- build_trade_table (end-to-end pure pipeline) -----------------------------


@pytest.mark.unit
def test_build_trade_table_full_pipeline() -> None:
    records = [
        _record(
            partner_code="0", partner_country="World", year=2022, value=10000.0, is_provisional=True
        ),
        _record(
            partner_code="842", partner_country="USA", year=2021, value=100.0, is_provisional=False
        ),
        _record(
            partner_code="842", partner_country="USA", year=2022, value=150.0, is_provisional=False
        ),
        _record(
            partner_code="826", partner_country="UK", year=2021, value=50.0, is_provisional=False
        ),
        _record(
            partner_code="826", partner_country="UK", year=2022, value=60.0, is_provisional=True
        ),
    ]
    table = build_trade_table(records, years=[2021, 2022])

    assert table.unit == "USD"
    assert table.years == [2021, 2022]
    assert table.excluded_partner_codes == ["0"]
    assert table.years_finalized == [2021]  # 2022 has UK's provisional record
    assert table.years_no_data == []  # both years have real records
    assert [r.partner_country for r in table.rows] == ["USA", "UK"]
    assert table.rows[0].cumulative_5yr == pytest.approx(250.0)
    assert table.rows[0].rank == 1
    assert table.rows[1].rank == 2


@pytest.mark.unit
def test_build_trade_table_flags_zero_record_years_as_no_data_not_provisional() -> None:
    """M21/PBO-03 end-to-end: a year with zero retained partner records
    (e.g. an HS6 code that didn't exist in the HS nomenclature yet) must
    land in `years_no_data`, not just fall into the `years_finalized`
    complement undifferentiated."""
    records = [
        _record(partner_code="842", partner_country="USA", year=2022, value=100.0),
    ]
    table = build_trade_table(records, years=[2021, 2022])
    assert table.years_finalized == [2022]
    assert table.years_no_data == [2021]


@pytest.mark.unit
def test_build_trade_table_empty_records_produces_empty_but_valid_table() -> None:
    table = build_trade_table([], years=YEARS)
    assert table.rows == []
    assert table.years_finalized == []
    assert table.years_no_data == YEARS  # every year has zero records
    assert table.excluded_partner_codes == []
    assert table.years == YEARS


@pytest.mark.unit
def test_build_trade_table_fetch_issues_default_to_empty_list() -> None:
    # The overwhelmingly common case: no `fetch_issues` argument at all.
    table = build_trade_table([], years=[2021])
    assert table.fetch_issues == []
    assert table.fetch_issue_years == []


@pytest.mark.unit
def test_build_trade_table_formats_fetch_issues_and_excludes_them_from_no_data() -> None:
    """2026-08-20 roadmap decision, end-to-end through `build_trade_table`:
    a failed year is rendered as a real, honest one-line note (`"{year}:
    {reason}"`) and is excluded from `years_no_data` even though it also
    has zero records — the two must never both claim the same year."""
    records = [
        _record(partner_code="842", partner_country="USA", year=2021, value=100.0),
    ]
    issues = [FetchIssue(year=2022, reason="UN Comtrade returned retryable status 429")]
    table = build_trade_table(records, years=[2021, 2022], fetch_issues=issues)

    assert table.fetch_issues == ["2022: UN Comtrade returned retryable status 429"]
    assert table.fetch_issue_years == [2022]
    assert 2022 not in table.years_no_data
    assert table.years_no_data == []


@pytest.mark.unit
def test_build_trade_table_fetch_issues_sorted_by_year_regardless_of_input_order() -> None:
    issues = [
        FetchIssue(year=2023, reason="reason for 2023"),
        FetchIssue(year=2021, reason="reason for 2021"),
    ]
    table = build_trade_table([], years=[2021, 2022, 2023], fetch_issues=issues)
    assert table.fetch_issues == ["2021: reason for 2021", "2023: reason for 2023"]
    assert table.fetch_issue_years == [2021, 2023]


# --- aggregate() node wrapper --------------------------------------------------


@pytest.mark.unit
def test_aggregate_node_builds_both_tables() -> None:
    query = TradeQuery(hs_code="010121", year_start=2022, year_end=2023)
    state: AnalysisState = {
        "query": query,
        "raw_imports": [
            _record(
                partner_code="842", partner_country="USA", year=2022, value=100.0, flow="import"
            )
        ],
        "raw_exports": [
            _record(partner_code="826", partner_country="UK", year=2023, value=200.0, flow="export")
        ],
    }
    result = aggregate(state)
    assert result["imports_table"].rows[0].partner_country == "USA"
    assert result["exports_table"].rows[0].partner_country == "UK"
    assert result["imports_table"].years == [2022, 2023]


@pytest.mark.unit
def test_aggregate_node_threads_fetch_issues_into_the_matching_table() -> None:
    query = TradeQuery(hs_code="010121", year_start=2021, year_end=2022)
    state: AnalysisState = {
        "query": query,
        "raw_imports": [],
        "raw_exports": [
            _record(partner_code="826", partner_country="UK", year=2021, value=200.0, flow="export")
        ],
        "import_fetch_issues": [FetchIssue(year=2022, reason="simulated import failure")],
        "export_fetch_issues": [],
    }
    result = aggregate(state)

    assert result["imports_table"].fetch_issues == ["2022: simulated import failure"]
    assert 2022 not in result["imports_table"].years_no_data
    assert result["exports_table"].fetch_issues == []


@pytest.mark.unit
def test_aggregate_node_missing_fetch_issues_keys_defaults_to_empty() -> None:
    # `import_fetch_issues`/`export_fetch_issues` are always written
    # alongside their `raw_*` sibling by `app.nodes.fetch_trade` in real
    # use, but `AnalysisState` is `total=False` — the node must not crash
    # if they're absent (e.g. a state built by hand, as every other test in
    # this file already does for `raw_imports`/`raw_exports`).
    query = TradeQuery(hs_code="010121", year_start=2021, year_end=2021)
    state: AnalysisState = {
        "query": query,
        "raw_imports": [],
        "raw_exports": [],
    }
    result = aggregate(state)
    assert result["imports_table"].fetch_issues == []
    assert result["exports_table"].fetch_issues == []


@pytest.mark.unit
def test_aggregate_node_honors_query_top_n_instead_of_the_hardcoded_default() -> None:
    # Regression test for the years/top_n configurability gap: aggregate()
    # used to call build_trade_table with no top_n at all, silently falling
    # back to its hardcoded TOP_N_PARTNERS=10 default regardless of what the
    # caller actually asked for.
    query = TradeQuery(hs_code="010121", year_start=2023, year_end=2023, top_n=3)
    state: AnalysisState = {
        "query": query,
        "raw_imports": [
            _record(
                partner_code="842", partner_country="USA", year=2023, value=400.0, flow="import"
            ),
            _record(
                partner_code="826", partner_country="UK", year=2023, value=300.0, flow="import"
            ),
            _record(
                partner_code="276", partner_country="Germany", year=2023, value=200.0, flow="import"
            ),
            _record(
                partner_code="392", partner_country="Japan", year=2023, value=100.0, flow="import"
            ),
        ],
        "raw_exports": [],
    }
    result = aggregate(state)
    assert len(result["imports_table"].rows) == 3  # capped at query.top_n, not TOP_N_PARTNERS
    assert [row.partner_country for row in result["imports_table"].rows] == [
        "USA",
        "UK",
        "Germany",
    ]


@pytest.mark.unit
def test_aggregate_node_short_circuits_on_existing_error() -> None:
    from app.schemas.errors import ErrorResponse

    state: AnalysisState = {
        "error": ErrorResponse(error_code="X", message="x", retryable=False, trace_id="t")
    }
    assert aggregate(state) == {}


@pytest.mark.unit
def test_aggregate_node_defensive_noop_when_state_incomplete() -> None:
    assert aggregate({}) == {}
    assert aggregate({"query": TradeQuery(hs_code="010121")}) == {}  # no raw_imports/raw_exports


# --- rest_of_world / world_total (Concern 1: "preserve the denominator") -----


@pytest.mark.unit
def test_rest_of_world_row_none_when_nothing_truncated() -> None:
    records = [
        _record(partner_code="842", partner_country="USA", year=2023, value=300.0),
        _record(partner_code="826", partner_country="UK", year=2023, value=200.0),
    ]
    assert _rest_of_world_row(records, years=[2023], top_n=10) is None


@pytest.mark.unit
def test_rest_of_world_row_sums_exactly_the_truncated_countries_at_the_boundary() -> None:
    """10 real countries -> no rest_of_world row; the 11th tips it over -
    the exact top_n boundary this feature exists to handle."""
    ten_countries = [
        _record(partner_code=str(code), partner_country=f"Country{code}", year=2023, value=100.0)
        for code in range(1, 11)
    ]
    assert _rest_of_world_row(ten_countries, years=[2023], top_n=10) is None

    eleven_countries = [
        *ten_countries,
        _record(partner_code="11", partner_country="Country11", year=2023, value=50.0),
    ]
    row = _rest_of_world_row(eleven_countries, years=[2023], top_n=10)
    assert row is not None
    assert row.partner_code == REST_OF_WORLD_PARTNER_CODE
    assert row.cumulative_5yr == pytest.approx(50.0)
    assert row.values_by_year[2023] == pytest.approx(50.0)
    assert row.rank == 11


@pytest.mark.unit
def test_build_trade_table_captures_world_total_without_double_counting_it() -> None:
    records = [
        _record(partner_code="0", partner_country="World", year=2023, value=1000.0),
        _record(partner_code="842", partner_country="USA", year=2023, value=300.0),
    ]
    table = build_trade_table(records, years=[2023])
    assert table.world_total_comtrade[2023] == pytest.approx(1000.0)
    # World never leaks into rows/rest_of_world - only into excluded_partner_codes once.
    assert "0" not in [row.partner_code for row in table.rows]
    assert table.rest_of_world is None
    assert table.excluded_partner_codes == ["0"]


@pytest.mark.unit
def test_build_trade_table_world_total_none_when_comtrade_never_reported_it() -> None:
    records = [_record(partner_code="842", partner_country="USA", year=2023, value=300.0)]
    table = build_trade_table(records, years=[2023])
    assert table.world_total_comtrade[2023] is None  # distinct from a reported 0.0


@pytest.mark.unit
def test_world_total_reconciles_none_when_world_total_missing() -> None:
    rows = [
        CountryRow(
            partner_country="USA",
            partner_code="842",
            values_by_year={2023: 300.0},
            cumulative_5yr=300.0,
            rank=1,
        )
    ]
    result = _world_total_reconciles(rows, None, {2023: None}, years=[2023])
    assert result == {2023: None}


@pytest.mark.unit
def test_world_total_reconciles_true_within_tolerance() -> None:
    rows = [
        CountryRow(
            partner_country="USA",
            partner_code="842",
            values_by_year={2023: 300.0},
            cumulative_5yr=300.0,
            rank=1,
        )
    ]
    rest_of_world = CountryRow(
        partner_country="All Other Countries",
        partner_code=REST_OF_WORLD_PARTNER_CODE,
        values_by_year={2023: 700.0},
        cumulative_5yr=700.0,
        rank=2,
    )
    result = _world_total_reconciles(rows, rest_of_world, {2023: 1000.0}, years=[2023])
    assert result == {2023: True}


@pytest.mark.unit
def test_world_total_reconciles_false_on_a_real_mismatch() -> None:
    """A deliberately inconsistent fixture (top-N + rest_of_world sums to
    far less than Comtrade's own World total) must be flagged, not silently
    pass - this is exactly the class of internal inconsistency the check
    exists to surface."""
    rows = [
        CountryRow(
            partner_country="USA",
            partner_code="842",
            values_by_year={2023: 300.0},
            cumulative_5yr=300.0,
            rank=1,
        )
    ]
    result = _world_total_reconciles(rows, None, {2023: 1000.0}, years=[2023])
    assert result == {2023: False}


# --- coefficient of variation (Concern 1: volatility) -------------------------
# Pure-function CoV/CAGR fixture tests live in tests/unit/test_timeseries_math.py
# (2026-09-02, Step 4 hardening) — this integration-level test stays here since
# it exercises `rank_top_partners`'s own wiring, not the pure math itself.


@pytest.mark.unit
def test_rank_top_partners_flags_high_volatility_but_not_a_stable_partner() -> None:
    records = [
        _record(partner_code="842", partner_country="Spiky", year=2019, value=30_000_000.0),
        _record(partner_code="842", partner_country="Spiky", year=2020, value=0.0),
        _record(partner_code="842", partner_country="Spiky", year=2021, value=0.0),
        _record(partner_code="826", partner_country="Steady", year=2019, value=9_000_000.0),
        _record(partner_code="826", partner_country="Steady", year=2020, value=9_000_000.0),
        _record(partner_code="826", partner_country="Steady", year=2021, value=9_000_000.0),
    ]
    rows = rank_top_partners(records, years=[2019, 2020, 2021])
    by_country = {row.partner_country: row for row in rows}
    assert by_country["Spiky"].coefficient_of_variation is not None
    assert by_country["Spiky"].coefficient_of_variation > HIGH_VOLATILITY_COV_THRESHOLD
    assert by_country["Spiky"].is_high_volatility is True
    assert by_country["Steady"].coefficient_of_variation == pytest.approx(0.0)
    assert by_country["Steady"].is_high_volatility is False


# --- HHI concentration index (Concern 3: new metrics) -------------------------


@pytest.mark.unit
def test_compute_hhi_hand_computed_value() -> None:
    # 50% + 30% + 20% shares -> 0.25 + 0.09 + 0.04 = 0.38
    assert _compute_hhi([500.0, 300.0, 200.0]) == pytest.approx(0.38)


@pytest.mark.unit
def test_compute_hhi_none_for_non_positive_total() -> None:
    assert _compute_hhi([]) is None
    assert _compute_hhi([0.0, 0.0]) is None


@pytest.mark.unit
def test_build_trade_table_hhi_reflects_every_real_country_not_the_truncated_view() -> None:
    """HHI must be computed over every real country's own cumulative value,
    not the truncated top-N-plus-one-rest_of_world-bucket view - lumping a
    diffuse tail into one synthetic row would overstate concentration
    (squaring one big lumped share vs. summing many smaller squared
    shares)."""
    # 10 equal top-N countries + 10 more equally-sized countries in the
    # tail: true HHI (20 equal 5% shares) is 20 * 0.05**2 = 0.05. Lumping
    # the tail 10 into one 50%-share rest_of_world bucket would instead
    # compute 10 * 0.05**2 + 0.5**2 = 0.275 - a very different answer.
    records = [
        _record(partner_code=str(code), partner_country=f"Country{code}", year=2023, value=100.0)
        for code in range(1, 21)
    ]
    table = build_trade_table(records, years=[2023], top_n=10)
    assert table.hhi is not None
    assert table.hhi == pytest.approx(0.05)


# --- trade balance (Concern 3: new metrics) -----------------------------------


@pytest.mark.unit
def test_compute_trade_balance_positive_when_exports_exceed_imports() -> None:
    imports = build_trade_table(
        [_record(partner_code="0", partner_country="World", year=2023, value=100.0)], years=[2023]
    )
    exports = build_trade_table(
        [_record(partner_code="0", partner_country="World", year=2023, value=250.0)], years=[2023]
    )
    balance = compute_trade_balance(imports, exports)
    assert balance.by_year[2023] == pytest.approx(150.0)
    assert balance.cumulative == pytest.approx(150.0)


@pytest.mark.unit
def test_compute_trade_balance_none_for_a_year_missing_either_sides_world_total() -> None:
    """One side never reported a World total for a year -> that year's
    balance must be None, never a one-sided, misleading number."""
    imports = build_trade_table(
        [_record(partner_code="842", partner_country="USA", year=2023, value=100.0)],
        years=[2023],  # no "0" record - world_total_comtrade[2023] is None
    )
    exports = build_trade_table(
        [_record(partner_code="0", partner_country="World", year=2023, value=250.0)], years=[2023]
    )
    balance = compute_trade_balance(imports, exports)
    assert balance.by_year[2023] is None
    assert balance.cumulative is None


@pytest.mark.unit
def test_compute_trade_balance_cumulative_skips_none_years() -> None:
    imports = build_trade_table(
        [
            _record(partner_code="0", partner_country="World", year=2022, value=100.0),
            _record(partner_code="842", partner_country="USA", year=2023, value=50.0),
        ],
        years=[2022, 2023],
    )
    exports = build_trade_table(
        [_record(partner_code="0", partner_country="World", year=2022, value=150.0)],
        years=[2022, 2023],  # no World row for 2023
    )
    balance = compute_trade_balance(imports, exports)
    assert balance.by_year[2022] == pytest.approx(50.0)
    assert balance.by_year[2023] is None
    assert balance.cumulative == pytest.approx(50.0)  # only the real year counted
