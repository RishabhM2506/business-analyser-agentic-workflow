"""Unit tests for `app.pipeline.baci`'s pure CSV-streaming parser — real,
verified BACI format (`t,i,j,k,v,q`), the thousand-USD/metric-ton ->
whole-USD/kg conversion, the India-involvement filter, and the
missing-column structural error. `load_baci_zip`/`upsert_baci_records`
(real ZIP/DB I/O) are exercised separately.
"""

from __future__ import annotations

import io
from decimal import Decimal

import pytest

from app.pipeline.baci import INDIA_CODE, BaciParseError, parse_baci_year_csv

pytestmark = pytest.mark.unit


def _csv_stream(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


def test_parse_keeps_a_row_where_india_is_the_importer() -> None:
    csv_text = "t,i,j,k,v,q\n2022,156,699,120791,23685.927,8047.4\n"
    records = list(
        parse_baci_year_csv(
            _csv_stream(csv_text),
            vintage="V202601",
            hs_revision="22",
            year=2022,
            hs6_codes={"120791"},
        )
    )

    assert len(records) == 1
    r = records[0]
    assert r.exporter_code == "156"
    assert r.importer_code == INDIA_CODE
    assert r.value_fob_usd == Decimal("23685927.000")  # thousand USD -> USD, exact
    assert r.quantity_kg == Decimal("8047400.0")  # metric tons -> kg, exact


def test_parse_keeps_a_row_where_india_is_the_exporter() -> None:
    csv_text = "t,i,j,k,v,q\n2022,699,124,120791,212.054,12.398\n"
    records = list(
        parse_baci_year_csv(
            _csv_stream(csv_text),
            vintage="V202601",
            hs_revision="22",
            year=2022,
            hs6_codes={"120791"},
        )
    )

    assert len(records) == 1
    assert records[0].exporter_code == INDIA_CODE
    assert records[0].importer_code == "124"


def test_parse_drops_a_row_not_involving_india() -> None:
    csv_text = "t,i,j,k,v,q\n2022,4,20,120791,0.412,0.002\n"
    records = list(
        parse_baci_year_csv(
            _csv_stream(csv_text),
            vintage="V202601",
            hs_revision="22",
            year=2022,
            hs6_codes={"120791"},
        )
    )

    assert records == []


def test_parse_drops_a_row_for_an_untracked_hs6_code() -> None:
    csv_text = "t,i,j,k,v,q\n2022,156,699,999999,1.0,1.0\n"
    records = list(
        parse_baci_year_csv(
            _csv_stream(csv_text),
            vintage="V202601",
            hs_revision="22",
            year=2022,
            hs6_codes={"120791"},
        )
    )

    assert records == []


def test_parse_treats_a_blank_value_as_none_never_zero() -> None:
    """D2's "ZERO vs missing" discipline, extended to this source too."""
    csv_text = "t,i,j,k,v,q\n2022,156,699,120791,,8047.4\n"
    records = list(
        parse_baci_year_csv(
            _csv_stream(csv_text),
            vintage="V202601",
            hs_revision="22",
            year=2022,
            hs6_codes={"120791"},
        )
    )

    assert records[0].value_fob_usd is None
    assert records[0].quantity_kg == Decimal("8047400.0")


def test_parse_raises_on_missing_expected_columns() -> None:
    csv_text = "year,exporter,importer,product,value\n2022,156,699,120791,23.6\n"
    with pytest.raises(BaciParseError):
        list(
            parse_baci_year_csv(
                _csv_stream(csv_text),
                vintage="V202601",
                hs_revision="22",
                year=2022,
                hs6_codes={"120791"},
            )
        )


def test_parse_stores_the_real_verified_india_code() -> None:
    """India's BACI code is 699 - the same code Comtrade uses, verified
    live against the real downloaded country_codes CSV (an earlier,
    unverified guess in this project's own planning notes assumed a
    different UN M49 numeric scheme)."""
    assert INDIA_CODE == "699"
