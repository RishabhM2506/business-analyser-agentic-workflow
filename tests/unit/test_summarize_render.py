"""Unit tests for `app.nodes.summarize.render_table` — the pure function
that turns a `TradeTable` into the compact prompt text a model sees. No
model, no I/O (docs/PLAN.md §7)."""

from __future__ import annotations

import pytest

from app.guardrails import extract_numbers
from app.nodes.summarize import render_table
from app.schemas.response import CountryRow, TradeTable


def _table(**overrides: object) -> TradeTable:
    defaults: dict[str, object] = dict(
        unit="USD",
        years=[2021, 2022, 2023],
        years_finalized=[2021, 2022],
        excluded_partner_codes=["0"],
        rows=[
            CountryRow(
                partner_country="USA",
                partner_code="842",
                values_by_year={2021: 100.0, 2022: None, 2023: 1992455.942},
                cumulative_5yr=1992555.942,
                rank=1,
            )
        ],
    )
    defaults.update(overrides)
    return TradeTable(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
def test_render_table_includes_label_country_and_rank() -> None:
    text = render_table("IMPORTS", _table())
    assert "IMPORTS" in text
    assert "USA" in text
    assert "#1" in text


@pytest.mark.unit
def test_render_table_missing_year_renders_as_no_data_not_zero() -> None:
    text = render_table("IMPORTS", _table())
    assert "2022: no data" in text
    assert "2022: 0" not in text


@pytest.mark.unit
def test_render_table_renders_whole_unit_rounded_values() -> None:
    text = render_table("IMPORTS", _table())
    # 1992455.942 rounded to whole units, comma-grouped
    assert "1,992,456" in text


@pytest.mark.unit
def test_render_table_notes_excluded_partner_codes_when_present() -> None:
    text = render_table("IMPORTS", _table(excluded_partner_codes=["0", "837"]))
    assert "excluded" in text.lower()
    assert "0" in text and "837" in text


@pytest.mark.unit
def test_render_table_omits_excluded_note_when_none_excluded() -> None:
    text = render_table("IMPORTS", _table(excluded_partner_codes=[]))
    assert "excluded" not in text.lower()


@pytest.mark.unit
def test_render_table_notes_unfinalized_years() -> None:
    text = render_table("IMPORTS", _table())
    assert "2023" in text
    assert "provisional" in text.lower() or "not yet finalized" in text.lower()


@pytest.mark.unit
def test_render_table_omits_unfinalized_note_when_all_finalized() -> None:
    text = render_table("IMPORTS", _table(years=[2021, 2022], years_finalized=[2021, 2022]))
    assert "provisional" not in text.lower()


@pytest.mark.unit
def test_render_table_empty_rows_notes_no_data() -> None:
    text = render_table("EXPORTS", _table(rows=[]))
    assert "no partner-country data available" in text


@pytest.mark.unit
def test_render_table_numbers_are_all_extractable_and_whole_unit_grounded() -> None:
    """Every number rendered must round-trip through the same extraction
    the output guardrail uses (app.guardrails.extract_numbers), and match
    the table's own rounded values — proves the prompt text a model sees
    is exactly what the guardrail will later check it against."""
    table = _table()
    text = render_table("IMPORTS", table)
    numbers = extract_numbers(text)
    assert round(1992455.942) in {round(n) for n in numbers}
    assert round(table.rows[0].cumulative_5yr) in {round(n) for n in numbers}
