"""Unit tests for `app.report.facts`'s pure helpers — the worst-status
ordering and the real partner-name CSV lookup. `assemble_facts` itself is
exercised against a real Postgres in `tests/integration/test_facts.py`.
"""

from __future__ import annotations

import pytest

from app.report.facts import _display_name, _worst_status

pytestmark = pytest.mark.unit


def test_worst_status_ok_beats_nothing() -> None:
    assert _worst_status(["OK"]) == "OK"


def test_worst_status_fetch_failed_dominates_everything() -> None:
    assert _worst_status(["OK", "ZERO", "FETCH_FAILED", "QTY_MISSING"]) == "FETCH_FAILED"


def test_worst_status_qty_missing_beats_zero() -> None:
    assert _worst_status(["ZERO", "QTY_MISSING"]) == "QTY_MISSING"


def test_display_name_resolves_a_real_partner_code() -> None:
    """792 is Turkey - the real, live-verified canonical-scenario code
    (data/comtrade-partner-areas.csv)."""
    assert _display_name("792") == "Türkiye"


def test_display_name_passes_through_the_all_partners_sentinel() -> None:
    assert _display_name("ALL_PARTNERS") == "ALL_PARTNERS"


def test_display_name_surfaces_the_real_country_name_for_an_unmapped_code() -> None:
    assert _display_name("UNMAPPED:RURITANIA") == "RURITANIA (unmapped)"


def test_display_name_falls_back_to_the_code_for_an_unknown_code() -> None:
    assert _display_name("999999") == "999999"
