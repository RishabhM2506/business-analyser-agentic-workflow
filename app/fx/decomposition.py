"""FX / quantity / price three-way decomposition (`docs/PLAN.md` §6, §11 — D8's
non-negotiable "presenting INR growth as a single undifferentiated number is a
BLOCKER" rule).

First-order log-decomposition: `value = qty * price_native * fx`, so
`ln(value_end/value_start)` equals the sum of `ln(qty_end/qty_start)`,
`ln(price_end/price_start)`, and `ln(fx_end/fx_start)`, each expressed as
a percentage. `Decimal.ln()` is used throughout —
never a float round-trip (this repo's "money is never a float" discipline,
D8, extended here since these percentages are derived from money figures).

Callers, not this module, are responsible for substituting `fx = Decimal(1)`
for a DGCIS-sourced (native INR, `fx_rate_used IS NULL`) row before calling
`decompose()` — `docs/PLAN.md` §6: "fx(t) = 1 for DGCIS's native INR rows",
which makes `delta_fx_pct` fall out to exactly `0` by construction for a
series that was never round-tripped through USD (D8's other BLOCKER: "native
INR DGCIS data round-tripped through USD").
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_PERCENT = Decimal(100)


@dataclass(frozen=True)
class FxDecomposition:
    """Every field is `None` when the underlying ratio is undefined (a
    start or end value of zero — e.g. a genuinely `ZERO`-status year has no
    meaningful "percent change") rather than raising: a report encountering
    this must render "not computable," never crash or silently show `0%`
    (which would misreport a real quantity/price change as none)."""

    delta_value_pct: Decimal | None
    delta_qty_pct: Decimal | None
    delta_price_pct: Decimal | None
    delta_fx_pct: Decimal | None


def _log_pct(start: Decimal, end: Decimal) -> Decimal | None:
    if start == 0 or end == 0:
        return None
    try:
        return (end / start).ln() * _PERCENT
    except InvalidOperation:
        # ln() of a negative ratio (a sign flip) is undefined for a real
        # number — money/quantity/fx are never negative in this domain, so
        # this is defensive, not an expected path.
        return None


def decompose(
    *,
    qty_start: Decimal,
    qty_end: Decimal,
    price_native_start: Decimal,
    price_native_end: Decimal,
    fx_start: Decimal,
    fx_end: Decimal,
) -> FxDecomposition:
    """`qty`/`price_native`/`fx` are each a (start, end) pair for the two
    periods being compared. `delta_value_pct` is the real, exact percentage
    change in `qty * price_native * fx`; the qty/price/fx split is the
    first-order log-approximation of how that change decomposes (accurate
    for the moderate year-over-year changes this report deals in, not
    intended for extreme multi-order-of-magnitude swings)."""
    value_start = qty_start * price_native_start * fx_start
    value_end = qty_end * price_native_end * fx_end

    delta_value_pct = None if value_start == 0 else (value_end / value_start - 1) * _PERCENT

    return FxDecomposition(
        delta_value_pct=delta_value_pct,
        delta_qty_pct=_log_pct(qty_start, qty_end),
        delta_price_pct=_log_pct(price_native_start, price_native_end),
        delta_fx_pct=_log_pct(fx_start, fx_end),
    )
