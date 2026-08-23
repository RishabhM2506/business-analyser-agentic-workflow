"""Unit tests for `app.report.mismatch`'s pure D9 logic — severity bands
(boundary-tested at 14.9/15.1/39.9/40.1%, per `docs/PLAN.md` §10's own
requirement that the bands are `<`/`>=`, not off-by-one), the signed-gap
formula, the check B "5-12% renders quiet" regression, and the
never-guess-a-missing-side rule. `compute_check_a`/`compute_check_b`
themselves are exercised against a real Postgres in
`tests/integration/test_mismatch.py`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.report.mismatch import (
    MismatchResult,
    SkippedCheck,
    _evaluate,
    _severity,
    _signed_gap_pct,
)

pytestmark = pytest.mark.unit


def test_severity_quiet_below_15_percent() -> None:
    assert _severity(Decimal("14.9"), direction_flip=False) == "quiet"


def test_severity_flag_at_exactly_15_percent() -> None:
    """The band is `<` 15, not `<=` - 15.0% itself is already 'flag'."""
    assert _severity(Decimal("15.0"), direction_flip=False) == "flag"


def test_severity_flag_just_above_15_percent() -> None:
    assert _severity(Decimal("15.1"), direction_flip=False) == "flag"


def test_severity_flag_just_below_40_percent() -> None:
    assert _severity(Decimal("39.9"), direction_flip=False) == "flag"


def test_severity_warning_at_exactly_40_percent() -> None:
    """The band is `>=` 40, not `>` - 40.0% itself is already 'warning'."""
    assert _severity(Decimal("40.0"), direction_flip=False) == "warning"


def test_severity_warning_just_above_40_percent() -> None:
    assert _severity(Decimal("40.1"), direction_flip=False) == "warning"


def test_check_b_5_to_12_percent_band_renders_quiet() -> None:
    """§10: check B's 5-12% band is asserted to render as 'quiet',
    explicitly - a regression test guards against someone "fixing" it
    into a flag later."""
    assert _severity(Decimal("5.0"), direction_flip=False) == "quiet"
    assert _severity(Decimal("8.5"), direction_flip=False) == "quiet"
    assert _severity(Decimal("12.0"), direction_flip=False) == "quiet"


def test_severity_direction_flip_overrides_magnitude_regardless_of_gap_size() -> None:
    """§10: "sign flips year-on-year -> ... untrustworthy (independent of
    gap size)" - even a tiny gap is untrustworthy if the sign flipped."""
    assert _severity(Decimal("0.5"), direction_flip=True) == "untrustworthy"
    assert _severity(Decimal("90.0"), direction_flip=True) == "untrustworthy"


def test_signed_gap_pct_positive_when_other_source_reports_higher() -> None:
    assert _signed_gap_pct(dgcis_value=1000, other_value=1100) == Decimal("10")


def test_signed_gap_pct_negative_when_other_source_reports_lower() -> None:
    assert _signed_gap_pct(dgcis_value=1000, other_value=900) == Decimal("-10")


def test_evaluate_skips_when_dgcis_value_is_none() -> None:
    outcome = _evaluate(
        check_name="A",
        hs6="120791",
        flow="import",
        year=2023,
        partner_country_code="ALL_PARTNERS",
        dgcis_value=None,
        other_value=1000,
        previous_signed_gap_pct=None,
    )
    assert isinstance(outcome, SkippedCheck)


def test_evaluate_skips_when_other_value_is_none() -> None:
    """A missing comtrade side is never treated as 0 (D2) - the check is
    un-computable, not silently "100% higher than nothing"."""
    outcome = _evaluate(
        check_name="A",
        hs6="120791",
        flow="import",
        year=2023,
        partner_country_code="ALL_PARTNERS",
        dgcis_value=1000,
        other_value=None,
        previous_signed_gap_pct=None,
    )
    assert isinstance(outcome, SkippedCheck)


def test_evaluate_skips_when_dgcis_value_is_zero() -> None:
    """A zero denominator is undefined, never approximated as an infinite
    or a zero gap."""
    outcome = _evaluate(
        check_name="A",
        hs6="120791",
        flow="import",
        year=2023,
        partner_country_code="ALL_PARTNERS",
        dgcis_value=0,
        other_value=500,
        previous_signed_gap_pct=None,
    )
    assert isinstance(outcome, SkippedCheck)


def test_evaluate_computes_a_real_result_with_no_prior_gap() -> None:
    outcome = _evaluate(
        check_name="A",
        hs6="120791",
        flow="import",
        year=2023,
        partner_country_code="ALL_PARTNERS",
        dgcis_value=1000,
        other_value=1100,
        previous_signed_gap_pct=None,
    )
    assert isinstance(outcome, MismatchResult)
    assert outcome.gap_pct == Decimal("10")
    assert outcome.severity == "quiet"
    assert outcome.direction_flip_yoy is False


def test_evaluate_detects_a_direction_flip_from_prior_year() -> None:
    """Prior year: other source was higher (+5%). This year: other source
    is lower (-3%) - a real sign flip, independent of either magnitude
    being small."""
    outcome = _evaluate(
        check_name="B",
        hs6="120791",
        flow="import",
        year=2024,
        partner_country_code="792",
        dgcis_value=1000,
        other_value=970,
        previous_signed_gap_pct=Decimal("5"),
    )
    assert isinstance(outcome, MismatchResult)
    assert outcome.direction_flip_yoy is True
    assert outcome.severity == "untrustworthy"


def test_evaluate_does_not_flag_a_flip_when_the_prior_gap_was_exactly_zero() -> None:
    """A prior gap of exactly 0 has no sign to flip from - not a flip."""
    outcome = _evaluate(
        check_name="B",
        hs6="120791",
        flow="import",
        year=2024,
        partner_country_code="792",
        dgcis_value=1000,
        other_value=1050,
        previous_signed_gap_pct=Decimal("0"),
    )
    assert isinstance(outcome, MismatchResult)
    assert outcome.direction_flip_yoy is False
