"""Unit tests for `app.pipeline.faostat`'s pure CSV-streaming parser —
real, verified format, the item-name filter, the wide-year unpivot, and
the `M`-flag "missing != 0" discipline (D2).
"""

from __future__ import annotations

import csv
import io
from decimal import Decimal

import pytest

from app.pipeline.faostat import FaostatParseError, _parse_value, parse_production_csv

pytestmark = pytest.mark.unit

_HEADER = [
    "Area Code",
    "Area Code (M49)",
    "Area",
    "Item Code",
    "Item Code (CPC)",
    "Item",
    "Element Code",
    "Element",
    "Unit",
    "Y2022",
    "Y2022F",
    "Y2022N",
    "Y2023",
    "Y2023F",
    "Y2023N",
]


def _row(
    *,
    area_code: str,
    area: str,
    item: str,
    v2022: str = "",
    f2022: str = "",
    v2023: str = "",
    f2023: str = "",
) -> list[str]:
    return [
        area_code,
        f"'{area_code}",
        area,
        "296",
        "'01449",
        item,
        "5510",
        "Production",
        "t",
        v2022,
        f2022,
        "",
        v2023,
        f2023,
        "",
    ]


def _csv_stream(*rows: list[str]) -> io.BytesIO:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_HEADER)
    writer.writerows(rows)
    return io.BytesIO(buf.getvalue().encode("utf-8"))


def test_parse_value_is_none_for_the_real_missing_flag() -> None:
    """The real FAOSTAT flag legend: M = "Missing value; data cannot
    exist" - authoritative regardless of the cell content."""
    assert _parse_value("", flag="M") is None


def test_parse_value_ignores_a_stray_value_alongside_an_m_flag() -> None:
    """The marker is authoritative regardless - never trust a value next
    to a real missing-data flag."""
    assert _parse_value("999", flag="M") is None


def test_parse_value_parses_a_real_official_figure() -> None:
    assert _parse_value("7922.000000", flag="A") == Decimal("7922.000000")


def test_parse_value_is_none_for_a_genuinely_blank_cell_with_no_flag() -> None:
    assert _parse_value("", flag="") is None


def test_parse_production_csv_filters_by_item_name() -> None:
    stream = _csv_stream(
        _row(area_code="2", area="Afghanistan", item="Almonds, in shell", v2022="100", f2022="A"),
        _row(area_code="356", area="India", item="Poppy seed", f2022="M", f2023="M"),
    )
    records = list(parse_production_csv(stream, item_names={"Poppy seed"}))

    assert {r.item for r in records} == {"Poppy seed"}
    assert {r.area for r in records} == {"India"}


def test_parse_production_csv_decodes_real_utf8_country_names_correctly() -> None:
    """Regression test for a real bug found live, 2026-08-25: the parser's
    first draft decoded this file as latin-1, but the real FAOSTAT CSV is
    UTF-8 - "Türkiye" (raw bytes `\\xc3\\xbc` for "ü") came out as the
    mojibake "TÃ¼rkiye" in the database until this was fixed."""
    stream = _csv_stream(
        _row(area_code="792", area="Türkiye", item="Poppy seed", v2023="7922", f2023="A")
    )
    records = list(parse_production_csv(stream, item_names={"Poppy seed"}))

    assert {r.area for r in records} == {"Türkiye"}


def test_parse_production_csv_unpivots_every_year_column() -> None:
    stream = _csv_stream(
        _row(
            area_code="792",
            area="Turkiye",
            item="Poppy seed",
            v2022="7000",
            f2022="A",
            v2023="7922",
            f2023="A",
        )
    )
    records = list(parse_production_csv(stream, item_names={"Poppy seed"}))

    assert {r.year for r in records} == {2022, 2023}
    y2023 = next(r for r in records if r.year == 2023)
    assert y2023.value == Decimal("7922")
    assert y2023.flag == "A"


def test_parse_production_csv_preserves_a_real_missing_value_as_none() -> None:
    stream = _csv_stream(
        _row(area_code="356", area="India", item="Poppy seed", f2022="M", f2023="M")
    )
    records = list(parse_production_csv(stream, item_names={"Poppy seed"}))

    assert all(r.value is None for r in records)  # never coerced to 0
    assert all(r.flag == "M" for r in records)


def test_parse_production_csv_raises_on_missing_expected_columns() -> None:
    csv_text = "year,area,item\n2023,India,Poppy seed\n"
    with pytest.raises(FaostatParseError):
        list(parse_production_csv(io.BytesIO(csv_text.encode("utf-8")), item_names={"Poppy seed"}))
