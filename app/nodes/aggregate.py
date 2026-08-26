"""`aggregate` node: pure functions — strip aggregate/"nes" partner codes,
rank top-10 by cumulative 5yr value, pivot into a 5-year table, flag year
completeness (docs/PLAN.md §2.2, §6). Deterministic Python only: the model
never sees, and therefore cannot mis-transcribe or invent, a ranking
decision (docs/PLAN.md §6).

This is the module docs/PLAN.md §7 calls out for exhaustive unit testing
against hand-computed fixtures — no I/O, no model, ever. `build_trade_table`
and its helpers below take/return plain `ComtradeRecord`/`TradeTable`
values; `tests/unit/test_aggregate.py` constructs `ComtradeRecord`s by hand
rather than depending on live/fixture data, so every edge case (ties,
missing years, more than 10 partners, all-aggregate input, a year with zero
data at all) is exactly controllable.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.schemas.response import CountryRow, TradeTable
from app.state import AnalysisState, FetchIssue, has_error
from app.tools.comtrade_client import (
    INDIA_REPORTER_CODE,
    ComtradeRecord,
    is_aggregate_partner_code,
)

TOP_N_PARTNERS = 10


def _is_excluded_partner_code(partner_code: str) -> bool:
    """True iff `partner_code` must be stripped before top-10 ranking:
    either a non-country aggregate/catch-all code (`is_aggregate_partner_code`
    — "World", an ", nes" regional bucket, "Bunkers", "Free Zones", "Special
    Categories") or the reporter's own code (finding M20/PBO-02).

    The fixed reporter, India (`INDIA_REPORTER_CODE`), can never
    legitimately be its own bilateral trading partner — but Comtrade
    sometimes carries reporter=partner rows anyway (e.g. re-imports or
    returned-goods categories; live-reproduced on HS 851713, where India
    appeared as one of its own top-10 import "partners," unexplained). Left
    unfiltered, that row is exactly the kind of unexplained, ownership-
    eroding artifact that makes a sharp user distrust every other number in
    the table on sight. `is_aggregate_partner_code` doesn't (and
    semantically shouldn't) cover this case on its own — India's code isn't
    a "non-country catch-all," it's a real country that just can't
    legitimately appear here — so this composes both checks rather than
    changing that function's own meaning."""
    return is_aggregate_partner_code(partner_code) or partner_code == INDIA_REPORTER_CODE


def strip_aggregate_partners(records: list[ComtradeRecord]) -> list[ComtradeRecord]:
    """Drop records whose `partner_code` must never appear as a ranked
    trading partner — a non-country catch-all ("World", an ", nes" regional
    bucket, "Bunkers", "Free Zones", "Special Categories" — see
    `app.tools.comtrade_client.is_aggregate_partner_code`) or the reporter's
    own code (finding M20/PBO-02) — before ranking. `docs/PLAN.md` §6: "the
    model never sees, and therefore cannot mis-transcribe or invent, a
    ranking decision" — this is also why it happens here, deterministically,
    rather than being left for a prompt instruction to the model."""
    return [r for r in records if not _is_excluded_partner_code(r.partner_code)]


def find_excluded_partner_codes(records: list[ComtradeRecord]) -> list[str]:
    """The distinct excluded partner codes (aggregate/catch-all, or the
    reporter's own — see `_is_excluded_partner_code`) actually present in
    `records` (before stripping) — transparency about what was excluded
    from *this* result specifically, not a generic static reference list
    (`TradeTable.excluded_partner_codes`, docs/PLAN.md §3.2)."""
    return sorted({r.partner_code for r in records if _is_excluded_partner_code(r.partner_code)})


def _cumulative_value(values_by_year: dict[int, float | None]) -> float:
    """Sum of the years we actually have a number for. This is plain
    aggregation-over-available-data (like SQL `SUM()` skipping `NULL`s), not
    interpolation: a missing year contributes nothing to the total and is
    still rendered as `None` in `values_by_year` — never guessed, never
    filled in (master brief §2.2)."""
    return sum(value for value in values_by_year.values() if value is not None)


def rank_top_partners(
    records: list[ComtradeRecord], *, years: list[int], top_n: int = TOP_N_PARTNERS
) -> list[CountryRow]:
    """Pivot `records` (already stripped of aggregate codes) into one row
    per partner country, ranked by 5-year cumulative value descending — an
    explicit Gate 0 decision (docs/PHASE0-FINDINGS.md §5): the most recent
    single year is frequently provisional/estimated, so cumulative value is
    the stable ranking basis, not latest-year value. Ties broken by partner
    country name (ascending) for fully deterministic output.

    Returns at most `top_n` rows — fewer if fewer than `top_n` partner
    countries have any data at all (no padding, no fabricated rows).
    """
    by_partner: dict[str, list[ComtradeRecord]] = defaultdict(list)
    for record in records:
        by_partner[record.partner_code].append(record)

    candidates: list[tuple[str, str, dict[int, float | None], float]] = []
    for partner_code, partner_records in by_partner.items():
        values_by_year: dict[int, float | None] = dict.fromkeys(years)
        for record in partner_records:
            if record.year in values_by_year:
                values_by_year[record.year] = record.value
        cumulative = _cumulative_value(values_by_year)
        partner_country = partner_records[0].partner_country
        candidates.append((partner_code, partner_country, values_by_year, cumulative))

    candidates.sort(key=lambda item: (-item[3], item[1]))

    return [
        CountryRow(
            partner_country=partner_country,
            partner_code=partner_code,
            values_by_year=values_by_year,
            cumulative_5yr=cumulative,
            rank=rank,
        )
        for rank, (partner_code, partner_country, values_by_year, cumulative) in enumerate(
            candidates[:top_n], start=1
        )
    ]


def flag_years_finalized(records: list[ComtradeRecord], *, years: list[int]) -> list[int]:
    """Which of `years` are NOT flagged provisional by Comtrade — the
    subset where every retained (non-aggregate-code) partner record for
    that year was genuinely reported, not modeled/estimated
    (`docs/PHASE0-FINDINGS.md` §4, `TradeTable.years_finalized`'s
    docstring). Deliberately computed over the *partner-level* records
    feeding the ranking (after aggregate-code stripping), which is a
    stricter/more precise signal than `app.cache.tool_cache`'s per-fetch TTL
    heuristic — that one conservatively includes the "World" aggregate row
    (fine for deciding how long to cache a raw fetch), this one answers the
    user-facing question "is the data behind the table I'm showing settled".

    A year with *zero* retained records is treated as NOT finalized —
    vacuous truth (`all([])  == True`) would otherwise mark a year with no
    data at all as "finalized", which is a meaningless/misleading claim.
    """
    by_year: dict[int, list[ComtradeRecord]] = defaultdict(list)
    for record in records:
        by_year[record.year].append(record)
    return [
        year
        for year in years
        if by_year[year] and all(not record.is_provisional for record in by_year[year])
    ]


def flag_years_no_data(
    records: list[ComtradeRecord],
    *,
    years: list[int],
    fetch_failed_years: frozenset[int] = frozenset(),
) -> list[int]:
    """Which of `years` have ZERO retained (non-aggregate-code) partner
    records at all — the subset of "not finalized" years that means
    something structurally different from "still settling" (finding
    M21/PBO-03, live-reproduced on HS 851713: every partner showed `null`
    for 2021, consistent with the HS 2022 nomenclature revision carving out
    that exact 6-digit code only from 2022 onward, not a reporting delay).

    `flag_years_finalized`'s own complement conflates two cases: a year
    with *some* records where at least one isn't finalized yet (genuinely
    provisional — "check back later" is honest advice) and a year with *no*
    records for any partner at all (commonly because this exact HS6
    classification code did not exist in that year's edition of the HS
    nomenclature; less commonly, because nothing has been reported for it
    yet). Calling the former "provisional — not yet finalized" is accurate;
    calling the latter that is actively misleading, since there is
    frequently nothing to finalize, ever. This function isolates that
    second case so the UI can use different, non-promissory language for
    it. Disjoint from `flag_years_finalized` by construction: a year here
    has zero records, a year there requires at least one.

    `fetch_failed_years` (2026-08-20, live user-reported finding) excludes
    a *third* case that would otherwise land here by the same zero-records
    logic: a year whose Comtrade fetch itself failed after every retry
    attempt (`app.nodes.fetch_trade.FetchIssue`). Unlike a genuine
    zero-records response, we never actually got an answer for that year —
    calling it "no data recorded" would assert something we don't know to
    be true. `TradeTable.fetch_issues` carries that case instead.
    """
    by_year: dict[int, list[ComtradeRecord]] = defaultdict(list)
    for record in records:
        by_year[record.year].append(record)
    return [year for year in years if not by_year[year] and year not in fetch_failed_years]


def _sort_fetch_issues(issues: list[FetchIssue]) -> list[FetchIssue]:
    """Stable, predictable order regardless of the (already-sequential, but
    defensively re-sorted here) order issues were appended in — shared by
    `TradeTable.fetch_issues`/`fetch_issue_years` so the two stay in
    lockstep by construction rather than being sorted independently."""
    return sorted(issues, key=lambda issue: issue.year)


def build_trade_table(
    records: list[ComtradeRecord],
    *,
    years: list[int],
    top_n: int = TOP_N_PARTNERS,
    fetch_issues: list[FetchIssue] | None = None,
) -> TradeTable:
    """Full pipeline for one flow direction (imports or exports): strip
    aggregate codes, rank top-N by 5yr cumulative value, flag year
    completeness, assemble the `TradeTable` the response envelope carries
    (docs/PLAN.md §3.2). `fetch_issues` (2026-08-20 roadmap decision):
    years `app.nodes.fetch_trade` couldn't retrieve after every retry —
    excluded from `years_no_data` (see that function's own docstring) and
    surfaced instead as `TradeTable.fetch_issues`."""
    sorted_issues = _sort_fetch_issues(fetch_issues or [])
    fetch_failed_years = frozenset(issue.year for issue in sorted_issues)
    excluded_partner_codes = find_excluded_partner_codes(records)
    country_records = strip_aggregate_partners(records)
    rows = rank_top_partners(country_records, years=years, top_n=top_n)
    years_finalized = flag_years_finalized(country_records, years=years)
    years_no_data = flag_years_no_data(
        country_records, years=years, fetch_failed_years=fetch_failed_years
    )
    return TradeTable(
        unit="USD",
        years=years,
        years_finalized=years_finalized,
        years_no_data=years_no_data,
        fetch_issues=[f"{issue.year}: {issue.reason}" for issue in sorted_issues],
        fetch_issue_years=[issue.year for issue in sorted_issues],
        excluded_partner_codes=excluded_partner_codes,
        rows=rows,
    )


def aggregate(state: AnalysisState) -> dict[str, Any]:
    """Turn `raw_imports`/`raw_exports` into `imports_table`/`exports_table`."""
    if has_error(state):
        return {}
    query = state.get("query")
    raw_imports = state.get("raw_imports")
    raw_exports = state.get("raw_exports")
    if query is None or raw_imports is None or raw_exports is None:
        return {}  # defensive: an upstream node should have set query/error
    if query.year_start is None or query.year_end is None:
        return {}  # defensive: validate_query always resolves these

    years = list(range(query.year_start, query.year_end + 1))
    return {
        "imports_table": build_trade_table(
            raw_imports,
            years=years,
            top_n=query.top_n,
            fetch_issues=state.get("import_fetch_issues"),
        ),
        "exports_table": build_trade_table(
            raw_exports,
            years=years,
            top_n=query.top_n,
            fetch_issues=state.get("export_fetch_issues"),
        ),
    }
