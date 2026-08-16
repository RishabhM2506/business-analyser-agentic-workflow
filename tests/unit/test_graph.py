"""Unit tests for the pure-function pieces of `app/graph.py`:
`assemble_response` (no I/O — constructs a response from state already in
memory) and `_checkpointer_conn_info` (a string/URL translation, no
network). Full graph assembly + real invocation belongs in
`tests/integration/test_graph.py` instead (it needs the real taxonomy CSV
and a real, if in-memory, checkpointer).
"""

from __future__ import annotations

import pytest

from app.graph import COMBINED_PROMPT_VERSION, _checkpointer_conn_info, assemble_response
from app.nodes.describe_item import PROMPT_VERSION as DESCRIBE_ITEM_PROMPT_VERSION
from app.nodes.summarize import PROMPT_VERSION as SUMMARIZE_PROMPT_VERSION
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.schemas.response import CountryRow, TradeTable
from app.state import AnalysisState


def _table() -> TradeTable:
    return TradeTable(
        unit="USD",
        years=[2021, 2022],
        years_finalized=[2021, 2022],
        excluded_partner_codes=["0"],
        rows=[
            CountryRow(
                partner_country="USA",
                partner_code="842",
                values_by_year={2021: 100.0, 2022: 150.0},
                cumulative_5yr=250.0,
                rank=1,
            )
        ],
    )


def _complete_state(**overrides: object) -> AnalysisState:
    base: dict[str, object] = {
        "query": TradeQuery(hs_code="010121"),
        "item_description": "A short description.",
        "analytical_summary": "A short summary.",
        "imports_table": _table(),
        "exports_table": _table(),
        "thread_id": "thread-1",
        "message_id": "message-1",
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


@pytest.mark.unit
def test_combined_prompt_version_encodes_both_node_prompt_versions() -> None:
    assert DESCRIBE_ITEM_PROMPT_VERSION in COMBINED_PROMPT_VERSION
    assert SUMMARIZE_PROMPT_VERSION in COMBINED_PROMPT_VERSION


@pytest.mark.unit
def test_assemble_response_short_circuits_on_existing_error() -> None:
    state: AnalysisState = {
        "error": ErrorResponse(error_code="X", message="x", retryable=False, trace_id="t")
    }
    assert assemble_response(state) == {}


@pytest.mark.unit
def test_assemble_response_builds_full_response_from_complete_state() -> None:
    result = assemble_response(_complete_state())

    assert "error" not in result
    response = result["response"]
    assert response.thread_id == "thread-1"
    assert response.message_id == "message-1"
    assert response.hs_code == "010121"
    assert response.item_description == "A short description."
    assert response.analytical_summary == "A short summary."
    assert response.imports.rows[0].partner_country == "USA"
    assert response.exports.rows[0].partner_country == "USA"
    assert response.provenance.source == "UN Comtrade (comtradeapi.un.org)"
    assert response.provenance.period_type == "calendar_year"
    assert response.provenance.currency == "USD"
    assert response.provenance.prompt_version == COMBINED_PROMPT_VERSION
    assert response.provenance.reporter_country == "India"  # finding M22/PBO-04


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing_field",
    [
        "query",
        "item_description",
        "analytical_summary",
        "imports_table",
        "exports_table",
        "thread_id",
        "message_id",
    ],
)
def test_assemble_response_missing_field_produces_incomplete_state_error(
    missing_field: str,
) -> None:
    state = _complete_state()
    del state[missing_field]  # type: ignore[misc]

    result = assemble_response(state)

    assert "response" not in result
    error = result["error"]
    assert isinstance(error, ErrorResponse)
    assert error.error_code == "INCOMPLETE_STATE"
    assert error.retryable is True


@pytest.mark.unit
def test_assemble_response_mints_trace_id_when_absent_on_incomplete_state() -> None:
    state = _complete_state()
    del state["query"]  # type: ignore[misc]
    result = assemble_response(state)
    assert result["error"].trace_id  # non-empty, minted rather than crashing


@pytest.mark.unit
def test_checkpointer_conn_info_translates_sqlite_file_url() -> None:
    backend, conn_info = _checkpointer_conn_info("sqlite+aiosqlite:///./local.db")
    assert backend == "sqlite"
    assert conn_info == "./local.db"


@pytest.mark.unit
def test_checkpointer_conn_info_translates_sqlite_in_memory_url() -> None:
    backend, conn_info = _checkpointer_conn_info("sqlite+aiosqlite:///:memory:")
    assert backend == "sqlite"
    assert conn_info == ":memory:"


@pytest.mark.unit
def test_checkpointer_conn_info_translates_postgres_url_and_drops_driver_suffix() -> None:
    backend, conn_info = _checkpointer_conn_info("postgresql+asyncpg://user:pass@host:5432/dbname")
    assert backend == "postgres"
    assert conn_info == "postgresql://user:pass@host:5432/dbname"
    assert "+asyncpg" not in conn_info  # psycopg (not asyncpg) is what actually connects


@pytest.mark.unit
def test_checkpointer_conn_info_rejects_unsupported_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported database_url backend"):
        _checkpointer_conn_info("mysql://user:pass@host/db")
