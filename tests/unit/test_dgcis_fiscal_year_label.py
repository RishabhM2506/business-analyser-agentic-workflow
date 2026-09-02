"""Unit tests for `app.pipeline.dgcis._fiscal_year_label_for_month` —
India's fiscal year runs April-March, so a calendar month in Jan/Feb/Mar
belongs to the fiscal year that *started* the previous calendar year.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.pipeline.dgcis import _fiscal_year_label_for_month

pytestmark = pytest.mark.unit


def test_april_starts_a_new_fiscal_year() -> None:
    assert _fiscal_year_label_for_month(date(2023, 4, 1)) == "2023 - 2024"


def test_december_is_still_within_the_fiscal_year_that_started_in_april() -> None:
    assert _fiscal_year_label_for_month(date(2023, 12, 1)) == "2023 - 2024"


def test_january_belongs_to_the_fiscal_year_that_started_the_previous_april() -> None:
    assert _fiscal_year_label_for_month(date(2024, 1, 1)) == "2023 - 2024"


def test_march_is_the_last_month_of_the_fiscal_year_that_started_the_previous_april() -> None:
    assert _fiscal_year_label_for_month(date(2024, 3, 1)) == "2023 - 2024"
