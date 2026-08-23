"""Unit tests for `app.pipeline.msp` — real response-shape parsing (both
year-pair columns split into separate rows), never-a-float money parsing,
and the never-coerce-missing-to-zero discipline (D2).
"""

from __future__ import annotations

import pytest

from app.pipeline.msp import _parse_paise, _records_from_raw

pytestmark = pytest.mark.unit


def _real_raw_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "_sl__no_": 1,
        "crops": "Kharif Crops",
        "commodity": "Paddy (Vommon)",
        "_2017_18___cost": 1117,
        "_2017_18___msp": 1550,
        "_2022_23___cost": 1360,
        "_2022_23___msp": 2040,
    }
    base.update(overrides)
    return base


def test_parse_paise_converts_a_real_number_to_paise() -> None:
    assert _parse_paise(1550) == 155_000


def test_parse_paise_is_none_for_a_missing_field() -> None:
    """Never coerced to 0 - D2's "missing != zero" discipline."""
    assert _parse_paise(None) is None


def test_records_from_raw_splits_into_one_row_per_year_pair() -> None:
    records = _records_from_raw(_real_raw_row())

    assert len(records) == 2
    assert {r.year_label for r in records} == {"2017-18", "2022-23"}


def test_records_from_raw_maps_the_real_2017_18_values() -> None:
    records = _records_from_raw(_real_raw_row())

    row = next(r for r in records if r.year_label == "2017-18")
    assert row.crops == "Kharif Crops"
    assert row.commodity == "Paddy (Vommon)"
    assert row.cost_inr_paise_per_qtl == 111_700
    assert row.msp_inr_paise_per_qtl == 155_000


def test_records_from_raw_maps_the_real_2022_23_values() -> None:
    records = _records_from_raw(_real_raw_row())

    row = next(r for r in records if r.year_label == "2022-23")
    assert row.cost_inr_paise_per_qtl == 136_000
    assert row.msp_inr_paise_per_qtl == 204_000


def test_records_from_raw_returns_empty_for_a_malformed_row() -> None:
    assert _records_from_raw({"crops": "Kharif Crops"}) == []


def test_records_from_raw_preserves_a_missing_cost_as_none() -> None:
    records = _records_from_raw(_real_raw_row(_2017_18___cost=None))

    row = next(r for r in records if r.year_label == "2017-18")
    assert row.cost_inr_paise_per_qtl is None
    assert row.msp_inr_paise_per_qtl == 155_000  # unaffected
