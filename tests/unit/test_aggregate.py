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
    aggregate,
    build_trade_table,
    find_excluded_partner_codes,
    flag_years_finalized,
    rank_top_partners,
    strip_aggregate_partners,
)
from app.schemas.query import TradeQuery
from app.state import AnalysisState
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
    assert [r.partner_country for r in table.rows] == ["USA", "UK"]
    assert table.rows[0].cumulative_5yr == pytest.approx(250.0)
    assert table.rows[0].rank == 1
    assert table.rows[1].rank == 2


@pytest.mark.unit
def test_build_trade_table_empty_records_produces_empty_but_valid_table() -> None:
    table = build_trade_table([], years=YEARS)
    assert table.rows == []
    assert table.years_finalized == []
    assert table.excluded_partner_codes == []
    assert table.years == YEARS


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
