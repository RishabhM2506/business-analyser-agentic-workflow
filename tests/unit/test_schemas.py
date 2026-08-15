"""Schema validation round-trips for the data contracts in docs/PLAN.md §3.

These are real tests against real (fully implemented, Phase-2-approved)
code — `app/schemas/*.py` — not against any stubbed business logic.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.schemas.response import CountryRow, Provenance, TradeAnalysisResponse, TradeTable


@pytest.mark.unit
def test_trade_query_defaults() -> None:
    query = TradeQuery(hs_code="010121")
    assert query.flow == "both"
    assert query.year_start is None
    assert query.year_end is None
    assert query.partner_region is None
    assert query.value_or_volume == "value"
    assert query.tenant_id == "default"
    assert query.user_id == "default"


@pytest.mark.unit
def test_trade_query_round_trip() -> None:
    query = TradeQuery(
        hs_code="010121",
        flow="import",
        year_start=2019,
        year_end=2023,
        partner_region="APAC",
        value_or_volume="value",
        tenant_id="acme",
        user_id="u-1",
    )
    restored = TradeQuery.model_validate_json(query.model_dump_json())
    assert restored == query


@pytest.mark.unit
@pytest.mark.parametrize("bad_hs_code", ["", "1234", "12345678", "abcdef", "12-345"])
def test_trade_query_rejects_non_hs6_codes(bad_hs_code: str) -> None:
    with pytest.raises(ValidationError):
        TradeQuery(hs_code=bad_hs_code)


@pytest.mark.unit
def test_trade_query_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TradeQuery.model_validate({"hs_code": "010121", "unexpected_field": "x"})


def _sample_trade_table() -> TradeTable:
    return TradeTable(
        unit="USD",
        years=[2019, 2020, 2021, 2022, 2023],
        years_finalized=[2019, 2020, 2021, 2022],
        excluded_partner_codes=["0-nes"],
        rows=[
            CountryRow(
                partner_country="United States",
                partner_code="842",
                values_by_year={2019: 100.0, 2020: None, 2021: 120.0, 2022: 130.0, 2023: 140.0},
                cumulative_5yr=490.0,
                rank=1,
            )
        ],
    )


@pytest.mark.unit
def test_trade_table_round_trip_with_missing_year() -> None:
    table = _sample_trade_table()
    restored = TradeTable.model_validate_json(table.model_dump_json())
    assert restored == table
    # A missing year is represented as None, never interpolated.
    assert restored.rows[0].values_by_year[2020] is None


@pytest.mark.unit
def test_trade_analysis_response_round_trip() -> None:
    response = TradeAnalysisResponse(
        thread_id="thread-1",
        message_id="msg-1",
        hs_code="010121",
        item_description="Placeholder description.",
        imports=_sample_trade_table(),
        exports=_sample_trade_table(),
        analytical_summary="Placeholder summary.",
        provenance=Provenance(
            source="UN Comtrade (comtradeapi.un.org)",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            period_type="calendar_year",
            currency="USD",
            prompt_version="v0-scaffold",
        ),
    )
    restored = TradeAnalysisResponse.model_validate_json(response.model_dump_json())
    assert restored == response


@pytest.mark.unit
def test_provenance_rejects_wrong_source_literal() -> None:
    with pytest.raises(ValidationError):
        Provenance(
            source="some other source",  # type: ignore[arg-type]
            retrieved_at=datetime.now(UTC),
            period_type="calendar_year",
            currency="USD",
            prompt_version="v0-scaffold",
        )


@pytest.mark.unit
def test_error_response_round_trip() -> None:
    error = ErrorResponse(
        error_code="BUDGET_EXCEEDED",
        message="Daily model-call budget exceeded. Try again tomorrow.",
        retryable=False,
        trace_id="trace-123",
    )
    restored = ErrorResponse.model_validate_json(error.model_dump_json())
    assert restored == error
