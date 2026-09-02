"""Shared, dependency-free pure math over a plain `dict[int, float | None]`
year-series — no coupling to either `app.nodes.aggregate`'s Comtrade-only
`TradeTable`/`CountryRow` schemas or `app.report.facts`'s DGCIS-backed
`Facts` schema, so both pipelines can compute the same derived statistics
without importing each other (2026-09-02, Step 4 hardening: extracted out
of `app.nodes.aggregate`, where these were first built for Step 3's
rest-of-world/volatility/CAGR work, once `app.report.facts` needed the
identical math over its own differently-shaped, already-fetched data).

Every function here follows the same discipline: `None` (never a
fabricated/interpolated number) whenever there isn't honestly enough real
data to compute the figure — a missing year contributes nothing and is
never silently treated as 0 (master brief §2.2: "Missing data is shown as
missing").
"""

from __future__ import annotations

import statistics

# This module's own reasoned starting point for "volatile enough to flag,"
# not an empirically validated cutoff (no measured distribution of real
# per-partner CoV values exists yet) — flagged the same way
# `HIGH_CONFIDENCE_THRESHOLD` was in `app.search.rerank` before it was
# removed, and the same way `app.report.facts` flags its own invented
# thresholds: a real number that should be revisited once real usage data
# exists, not treated as if it were derived from one.
HIGH_VOLATILITY_COV_THRESHOLD = 1.0


def coefficient_of_variation(values_by_year: dict[int, float | None]) -> float | None:
    """Sample standard deviation / mean over the years we actually have a
    number for — a scale-independent volatility signal (a partner who
    traded $30M once and $0 every other year should read as volatile, not
    simply "large," which a raw cumulative-value ranking alone can't
    distinguish).

    `None` (never a fabricated/exploding ratio) when there are fewer than 2
    real data points to compute a spread from, or when the mean is exactly
    zero (an undefined denominator, not a "perfectly stable" 0 — the same
    "don't compute from an undefined denominator" discipline
    `app.report.mismatch`'s severity bands already use)."""
    real_values = [v for v in values_by_year.values() if v is not None]
    if len(real_values) < 2:
        return None
    mean = statistics.mean(real_values)
    if mean == 0:
        return None
    return statistics.stdev(real_values) / mean


def cagr(values_by_year: dict[int, float | None]) -> float | None:
    """Compound annual growth rate between this series' own earliest and
    latest *real* (non-`None`) years — deliberately not the declared
    year-range's own first/last year, since a series with a gap at either
    end must use its own real endpoints rather than silently treating a
    missing year as 0.

    `None` when there are fewer than 2 distinct real-valued years, or when
    the earliest real value isn't strictly positive (CAGR from a zero or
    negative base is mathematically undefined — never fabricated)."""
    real_years = sorted(year for year, value in values_by_year.items() if value is not None)
    if len(real_years) < 2:
        return None
    start_year, end_year = real_years[0], real_years[-1]
    start_value = values_by_year[start_year]
    end_value = values_by_year[end_year]
    assert start_value is not None and end_value is not None  # by construction of real_years
    if start_value <= 0:
        return None
    n_years = end_year - start_year
    return float((end_value / start_value) ** (1 / n_years) - 1)
