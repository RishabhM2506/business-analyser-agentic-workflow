"""Unit tests for `app.report.monthly_current_year`'s pure helpers —
percentage-change math and the marker-to-status/is_provisional mapping.
`compute_monthly_current_year`/`upsert_monthly_current_year` themselves
are exercised against a real Postgres in
`tests/integration/test_monthly_current_year.py`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.report.monthly_current_year import _pct_change, _RawCell, _status_for_cell

pytestmark = pytest.mark.unit


def test_pct_change_positive_growth() -> None:
    assert _pct_change(previous=1000, current=1100) == Decimal("10")


def test_pct_change_negative_growth() -> None:
    assert _pct_change(previous=1000, current=900) == Decimal("-10")


def test_pct_change_is_none_for_a_zero_denominator() -> None:
    assert _pct_change(previous=0, current=1000) is None


def test_status_for_missing_cell_is_not_yet_published() -> None:
    status, detail, is_provisional = _status_for_cell(None)

    assert status == "NOT_YET_PUBLISHED"
    assert detail is not None
    assert is_provisional is False


def test_status_for_advance_marker_is_not_yet_published_regardless_of_value() -> None:
    """A real "(A)"-marked month showed value=0.00 for the specific
    commodity too - the marker is authoritative regardless."""
    cell = _RawCell(value_inr_paise=0, quantity_kg=None, marker="A")

    status, _detail, is_provisional = _status_for_cell(cell)

    assert status == "NOT_YET_PUBLISHED"
    assert is_provisional is False


def test_status_for_no_value_is_not_reported() -> None:
    cell = _RawCell(value_inr_paise=None, quantity_kg=None, marker="R")

    status, _detail, is_provisional = _status_for_cell(cell)

    assert status == "NOT_REPORTED"
    assert is_provisional is False


def test_status_for_zero_value_is_zero() -> None:
    cell = _RawCell(value_inr_paise=0, quantity_kg=None, marker="R")

    status, _detail, is_provisional = _status_for_cell(cell)

    assert status == "ZERO"
    assert is_provisional is False


def test_status_for_zero_value_with_flash_marker_is_zero_but_provisional() -> None:
    """status and is_provisional are orthogonal - a real, reported zero
    can still be subject to later revision."""
    cell = _RawCell(value_inr_paise=0, quantity_kg=None, marker="F")

    status, _detail, is_provisional = _status_for_cell(cell)

    assert status == "ZERO"
    assert is_provisional is True


def test_status_for_real_value_with_no_quantity_is_qty_missing() -> None:
    cell = _RawCell(value_inr_paise=1000, quantity_kg=None, marker="R")

    status, _detail, is_provisional = _status_for_cell(cell)

    assert status == "QTY_MISSING"
    assert is_provisional is False


def test_status_for_finalized_value_and_quantity_is_ok() -> None:
    cell = _RawCell(value_inr_paise=1000, quantity_kg=Decimal("50"), marker="R")

    status, _detail, is_provisional = _status_for_cell(cell)

    assert status == "OK"
    assert is_provisional is False


def test_status_for_flash_value_and_quantity_is_provisional() -> None:
    cell = _RawCell(value_inr_paise=1000, quantity_kg=Decimal("50"), marker="F")

    status, _detail, is_provisional = _status_for_cell(cell)

    assert status == "PROVISIONAL"
    assert is_provisional is True
