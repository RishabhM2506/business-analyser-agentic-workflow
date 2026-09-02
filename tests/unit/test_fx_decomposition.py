"""Unit tests for `app.fx.decomposition.decompose` — the D8 three-way
qty/price/FX split (`docs/PLAN.md` §6, §11)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.fx.decomposition import decompose

_ONE = Decimal(1)


@pytest.mark.unit
def test_decompose_attributes_a_pure_quantity_change_entirely_to_qty() -> None:
    """Quantity doubles, price and FX unchanged -> ~69.3% (ln(2)*100) on
    qty, ~0% on price and fx, and the real (not log-approximated) value
    change is exactly 100%."""
    result = decompose(
        qty_start=Decimal(100),
        qty_end=Decimal(200),
        price_native_start=Decimal(10),
        price_native_end=Decimal(10),
        fx_start=_ONE,
        fx_end=_ONE,
    )

    assert result.delta_value_pct == Decimal(100)  # exact: value doubled
    assert result.delta_qty_pct is not None and round(result.delta_qty_pct, 1) == Decimal("69.3")
    assert result.delta_price_pct == Decimal(0)
    assert result.delta_fx_pct == Decimal(0)


@pytest.mark.unit
def test_decompose_dgcis_row_with_fx_fixed_at_one_contributes_zero_fx_delta() -> None:
    """docs/PLAN.md §6: DGCIS rows pass fx_start=fx_end=1 (never round-
    tripped through USD, D8) -> delta_fx_pct is exactly 0 by construction,
    regardless of the real rupee's movement that period."""
    result = decompose(
        qty_start=Decimal(100),
        qty_end=Decimal(110),
        price_native_start=Decimal(50),
        price_native_end=Decimal(55),
        fx_start=_ONE,
        fx_end=_ONE,
    )

    assert result.delta_fx_pct == Decimal(0)


@pytest.mark.unit
def test_decompose_a_pure_fx_move_is_attributed_entirely_to_fx() -> None:
    result = decompose(
        qty_start=Decimal(100),
        qty_end=Decimal(100),
        price_native_start=Decimal(10),
        price_native_end=Decimal(10),
        fx_start=Decimal(80),
        fx_end=Decimal(88),
    )

    assert result.delta_qty_pct == Decimal(0)
    assert result.delta_price_pct == Decimal(0)
    assert result.delta_fx_pct is not None and result.delta_fx_pct > Decimal(0)


@pytest.mark.unit
def test_decompose_returns_none_for_every_field_when_start_value_is_zero() -> None:
    """A genuinely ZERO-status starting year has no meaningful percent
    change - must return None (not crash, not silently report 0%)."""
    result = decompose(
        qty_start=Decimal(0),
        qty_end=Decimal(100),
        price_native_start=Decimal(10),
        price_native_end=Decimal(10),
        fx_start=_ONE,
        fx_end=_ONE,
    )

    assert result.delta_value_pct is None
    assert result.delta_qty_pct is None


@pytest.mark.unit
def test_decompose_returns_none_for_the_specific_term_that_goes_to_zero() -> None:
    """Only the term whose own start/end value is zero is undefined - a
    price genuinely going to zero shouldn't null out the quantity term,
    which is still well-defined."""
    result = decompose(
        qty_start=Decimal(100),
        qty_end=Decimal(120),
        price_native_start=Decimal(10),
        price_native_end=Decimal(0),
        fx_start=_ONE,
        fx_end=_ONE,
    )

    assert result.delta_qty_pct is not None
    assert result.delta_price_pct is None
    # value_start is non-zero (100*10*1), so delta_value_pct is still
    # computable even though price_end is zero.
    assert result.delta_value_pct == Decimal(-100)
