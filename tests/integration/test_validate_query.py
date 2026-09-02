"""Tests for the `validate_query` node — hs_code allowlist check (reads the
real checked-in taxonomy CSV, hence `integration` not `unit`) and
year-range defaulting (docs/PLAN.md §2.2, §6)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.nodes.validate_query import resolve_year_range, validate_query
from app.schemas.query import TradeQuery
from app.state import AnalysisState


@pytest.mark.integration
def test_validate_query_accepts_known_hs6_code_and_resolves_years() -> None:
    state: AnalysisState = {"query": TradeQuery(hs_code="010121")}
    result = validate_query(state)
    assert "error" not in result
    normalized = result["query"]
    assert normalized.hs_code == "010121"
    assert normalized.year_start is not None
    assert normalized.year_end is not None
    assert normalized.year_end - normalized.year_start == 4  # 5-year inclusive window


@pytest.mark.integration
def test_validate_query_preserves_explicit_year_range() -> None:
    state: AnalysisState = {"query": TradeQuery(hs_code="010121", year_start=2015, year_end=2019)}
    result = validate_query(state)
    assert result["query"].year_start == 2015
    assert result["query"].year_end == 2019


@pytest.mark.integration
def test_validate_query_rejects_unknown_hs_code() -> None:
    # Shape-valid (6 digits) but absent from the checked-in taxonomy.
    state: AnalysisState = {"query": TradeQuery(hs_code="000000"), "trace_id": "t-1"}
    result = validate_query(state)
    assert "query" not in result
    error = result["error"]
    assert error.error_code == "INVALID_HS_CODE"
    assert error.retryable is False
    assert error.trace_id == "t-1"


@pytest.mark.integration
def test_validate_query_missing_query_produces_error() -> None:
    state: AnalysisState = {}
    result = validate_query(state)
    assert result["error"].error_code == "INVALID_QUERY"


@pytest.mark.integration
def test_validate_query_mints_trace_id_when_absent() -> None:
    state: AnalysisState = {"query": TradeQuery(hs_code="000000")}
    result = validate_query(state)
    assert result["error"].trace_id  # non-empty, minted rather than crashing


@pytest.mark.unit
def test_resolve_year_range_defaults_to_last_five_complete_years() -> None:
    query = TradeQuery(hs_code="010121")
    fixed_now = datetime(2026, 8, 16, tzinfo=UTC)
    year_start, year_end = resolve_year_range(query, now=fixed_now)
    assert year_end == 2025  # last fully-completed calendar year
    assert year_start == 2021
    assert year_end - year_start == 4


@pytest.mark.unit
def test_resolve_year_range_respects_explicit_year_end_only() -> None:
    query = TradeQuery(hs_code="010121", year_end=2020)
    year_start, year_end = resolve_year_range(query)
    assert year_end == 2020
    assert year_start == 2016


@pytest.mark.unit
def test_resolve_year_range_honors_relative_years_field() -> None:
    query = TradeQuery(hs_code="010121", years=3)
    fixed_now = datetime(2026, 8, 16, tzinfo=UTC)
    year_start, year_end = resolve_year_range(query, now=fixed_now)
    assert year_end == 2025  # unchanged "latest available" heuristic
    assert year_start == 2023  # 3-year window instead of the 5-year default
    assert year_end - year_start == 2


@pytest.mark.integration
def test_validate_query_resolves_relative_years_field() -> None:
    state: AnalysisState = {"query": TradeQuery(hs_code="010121", years=8)}
    result = validate_query(state)
    normalized = result["query"]
    assert normalized.year_end - normalized.year_start == 7  # 8-year inclusive window


@pytest.mark.unit
def test_trade_query_rejects_years_combined_with_explicit_year_start() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TradeQuery(hs_code="010121", years=3, year_start=2015)


@pytest.mark.unit
def test_trade_query_rejects_years_combined_with_explicit_year_end() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TradeQuery(hs_code="010121", years=3, year_end=2020)


@pytest.mark.unit
def test_trade_query_top_n_defaults_to_ten() -> None:
    assert TradeQuery(hs_code="010121").top_n == 10


@pytest.mark.unit
def test_trade_query_top_n_rejects_out_of_bounds_values() -> None:
    with pytest.raises(ValueError):
        TradeQuery(hs_code="010121", top_n=2)  # below MIN_TOP_N (3)
    with pytest.raises(ValueError):
        TradeQuery(hs_code="010121", top_n=26)  # above MAX_TOP_N (25)
