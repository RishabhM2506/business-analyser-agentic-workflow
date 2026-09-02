"""Unit tests for `app.report.rankings`'s pure `compute_hhi` logic.
`compute_partner_rankings`/`upsert_partner_rankings` themselves are
exercised against a real Postgres in `tests/integration/test_rankings.py`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.report.rankings import PartnerRanking, compute_hhi

pytestmark = pytest.mark.unit


def _ranking(
    *, partner: str, rank: int | None, value: int | None, status: str = "OK"
) -> PartnerRanking:
    return PartnerRanking(
        hs6="120791",
        flow="import",
        year=2023,
        partner_country_code=partner,
        rank=rank,
        value_inr_paise=value,
        status=status,
    )


def test_hhi_is_one_when_a_single_partner_holds_the_entire_market() -> None:
    rankings = [_ranking(partner="792", rank=1, value=1000)]

    assert compute_hhi(rankings) == Decimal("1")


def test_hhi_is_evenly_split_between_two_equal_partners() -> None:
    rankings = [
        _ranking(partner="792", rank=1, value=500),
        _ranking(partner="156", rank=2, value=500),
    ]

    assert compute_hhi(rankings) == Decimal("0.5")


def test_hhi_ignores_unranked_partners_with_no_value() -> None:
    rankings = [
        _ranking(partner="792", rank=1, value=1000),
        _ranking(partner="156", rank=None, value=None, status="NOT_REPORTED"),
    ]

    assert compute_hhi(rankings) == Decimal("1")


def test_hhi_is_none_when_every_partner_is_zero() -> None:
    """Never fabricated as 0.0 - that would falsely claim a perfectly
    unconcentrated market when there's actually no real trade to measure
    concentration over."""
    rankings = [_ranking(partner="792", rank=1, value=0, status="ZERO")]

    assert compute_hhi(rankings) is None


def test_hhi_is_none_for_an_empty_ranking_list() -> None:
    assert compute_hhi([]) is None
