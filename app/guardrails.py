"""Input/output guardrails (docs/PLAN.md §6, master brief §8): controls
live in code, not prompts.

- Input: `hs_code` format + taxonomy-membership allowlist check, run in
  `validate_query` before any node reaches the network or a model.
- Output: every number appearing in `analytical_summary` must be a member
  of the flattened `imports`/`exports` table values before
  `assemble_response` runs — the concrete v1 instance of "the LLM never
  produces a number" (master brief §2.2), enforced structurally.

This is the single most load-bearing correctness module in the system
(master brief §2.2: "the single most important correctness property of the
whole system") — deliberately conservative and deterministic, no LLM
judge involved in either check.
"""

from __future__ import annotations

import contextlib
import re

from app.knowledge.provider import is_known_hs6_code
from app.schemas.response import TradeTable

# Matches either a comma-grouped number ("1,234,567.89") or a plain number
# ("2019", "42.5", "-3.1") — comma-grouped alternative listed first so a
# regex engine (which tries alternatives left-to-right and matches greedily)
# consumes "1,234" as one token rather than splitting at the comma. This
# also means a bare list of years like "2019, 2020, and 2021" is correctly
# read as three separate numbers, not merged: the comma-grouped alternative
# only matches when a comma is immediately followed by exactly three digits
# (proper thousands grouping), which "2019, 2020" (comma-space) never is.
#
# `(?<!\d)` guards both alternatives' leading `-?`: without it, a hyphen
# directly between two numbers — a plain-written year range like
# "2019-2023" (no spaces), not an adversarial input — misparses as "2019"
# followed by *negative* 2023 (M1/AWR-03, live-reproduced: a fully accurate
# summary using this ordinary formatting was rejected as if it had
# fabricated a number). The lookbehind means a `-` is only ever treated as
# a unary minus when it is NOT immediately preceded by a digit, which is
# exactly the case that distinguishes "a genuinely negative number" from
# "a hyphen separating two positive ones." Verified against the existing
# suite's cases plus the previously-failing one; no prior passing case
# regresses (see tests/unit/test_guardrails.py).
_NUMBER_PATTERN = re.compile(r"(?<!\d)-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<!\d)-?\d+(?:\.\d+)?")

# A number immediately adjacent to one of these is a definitionally
# *derived* figure — a percentage or a percentage-point delta — which
# prompts/summarize.md's own "Hard rules" already forbid the model from
# producing at all ("Do not compute or state percentages, growth rates,
# averages, or any other derived figure"). Rejected unconditionally,
# regardless of whether the bare numeral happens to collide with something
# in the grounded set (finding B7/AWR-02) — no accidental collision should
# ever save a number that is, by its own surrounding phrasing, self-evidently
# not a raw table cell.
_FORBIDDEN_SUFFIX_PATTERN = re.compile(r"\s*(%|percent\b|pct\.?|points?\b|pp\b)", re.IGNORECASE)
# Same reasoning, the other direction: a number immediately preceded by one
# of these growth/decline verbs + "by" ("grew BY 8", "declined BY 6") is
# the model asserting a computed delta, not transcribing a cell.
_FORBIDDEN_PRECEDING_PATTERN = re.compile(
    r"\b(?:grew|declined|increased|decreased|rose|fell)\s+by\s*$", re.IGNORECASE
)

# The narrow allowlist under which a bare *structural* number (a row's
# `rank`, or the fixed `len(years)`/`len(rows)` methodology constants — see
# `_structural_numbers` below) may be treated as grounded even though it
# isn't itself a monetary/year value: "#3", "top 3", "rank 3", "no. 3", or a
# number immediately followed by "years"/"partners"/"countries" ("5 years",
# "top 10 partners"). Deliberately narrow (finding B7/AWR-02): production
# tables always have up to 10 ranked rows and a 5-year window, so without
# this restriction every bare integer 1-10 in the prose is "grounded"
# purely because it happens to fall in that range — regardless of whether
# the prose is actually referring to a rank or a count at all. This was the
# exact exploit AWR-02 demonstrated with ordinary analyst prose ("grew 8%,
# declined 6%, ... 7 million versus 9 million") at the real production
# table shape. Anything outside this allowlist must match an actual table
# cell like any other number.
_STRUCTURAL_PRECEDING_PATTERN = re.compile(
    r"(?:#|\btop\b|\brank(?:s|ed|ing)?\b|\bno\.)\s*$", re.IGNORECASE
)
_STRUCTURAL_FOLLOWING_PATTERN = re.compile(
    r"^[\s-]*(?:years?|partners?|countries)\b", re.IGNORECASE
)


def check_hs_code_allowlisted(
    hs_code: str, *, taxonomy_path: str = "data/harmonized-system.csv"
) -> bool:
    """Return True iff `hs_code` exists in the checked-in HS6 taxonomy.

    Delegates to `app.knowledge.provider.is_known_hs6_code` — one CSV
    loader, shared by the allowlist check here and by `describe_item`'s
    source-text retrieval, rather than two independent parsers of the same
    file drifting apart.
    """
    return is_known_hs6_code(hs_code, taxonomy_path=taxonomy_path)


def extract_numbers(text: str) -> list[float]:
    """Extract every numeric token from `text`, as floats (thousands
    commas stripped). Public so `app.models.MockLLM` can reuse it to build
    canned summaries that are grounded by construction (see `app/models.py`),
    and so `app.nodes.describe_item` can assert *zero* numbers are present
    at all (finding M2/AWR-04)."""
    return [float(match.replace(",", "")) for match in _NUMBER_PATTERN.findall(text)]


def _flatten_value_numbers(*tables: TradeTable, hs_code: str | None = None) -> set[int]:
    """Every number that legitimately appears in prose as a *value* about
    `tables`, rounded to the nearest whole unit for tolerant comparison (a
    model copying "1,992,456" for a table value of 1992455.942 must not
    fail the check over sub-dollar rounding): a `years` entry, a row's
    `cumulative_5yr`, a per-year cell, and (if supplied) `hs_code` itself.

    `hs_code` is grounded the same way, for the same reason: `summarize.py`
    puts "HS code 010121" in the model's own prompt, and `extract_numbers`
    has no way to tell that numeral apart from a fabricated trade figure
    (leading zeros are stripped by float parsing, so "010121" reads back as
    the plain number 10121). Without this, a summary that legitimately
    references the code it's analyzing — "for HS code 010121, imports..." —
    would be rejected as ungrounded on every single run, mock or real.

    Deliberately excludes `row.rank` and the structural counts
    `len(table.years)`/`len(table.rows)` — see `_structural_numbers` and
    finding B7/AWR-02 for why those need a separate, context-gated check
    rather than blanket membership here.
    """
    grounded: set[int] = set()
    if hs_code is not None:
        with contextlib.suppress(ValueError):
            grounded.add(round(float(hs_code)))
    for table in tables:
        grounded.update(table.years)
        for row in table.rows:
            grounded.add(round(row.cumulative_5yr))
            for value in row.values_by_year.values():
                if value is not None:
                    grounded.add(round(value))
    return grounded


def _structural_numbers(*tables: TradeTable) -> set[int]:
    """Fixed methodology constants that are true by construction, not data
    the model could fabricate incorrectly — every row's `rank`, and the
    5-year-window/top-10-ranking sizes (`len(table.years)`,
    `len(table.rows)`, docs/PLAN.md §2.2) that any reasonable summary will
    mention as prose ("over the past 5 years", "the top 10 partners").

    Kept as a *separate* set from `_flatten_value_numbers` (finding
    B7/AWR-02): a number in this set is only actually treated as grounded
    when it also appears in prose adjacent to the narrow structural
    allowlist (`_STRUCTURAL_PRECEDING_PATTERN`/`_STRUCTURAL_FOLLOWING_PATTERN`)
    — never by blanket set membership alone, which previously made every
    integer 1-10 "grounded" regardless of context at the real production
    table shape (10 rows, 5 years).
    """
    structural: set[int] = set()
    for table in tables:
        structural.add(len(table.years))
        structural.add(len(table.rows))
        for row in table.rows:
            structural.add(row.rank)
    return structural


def _scan_numbers(
    prose: str, *tables: TradeTable, hs_code: str | None = None
) -> list[tuple[float, bool]]:
    """Every numeric token in `prose`, in order, paired with whether it's
    grounded. Shared implementation for `check_numbers_grounded` and
    `find_ungrounded_numbers` so the two can never drift against each other
    — one is always derived from the other rather than reimplementing the
    same scan twice.

    Three-step decision per number, in order (finding B7/AWR-02):
    1. Immediately adjacent to forbidden phrasing (a %/points suffix, or a
       growth/decline-verb-by prefix) -> always ungrounded, unconditionally.
    2. Otherwise, a real table/hs_code value -> grounded.
    3. Otherwise, a structural number (rank / year-count / row-count) *and*
       adjacent to the narrow structural-phrasing allowlist -> grounded.
       Anything else -> ungrounded.
    """
    value_grounded = _flatten_value_numbers(*tables, hs_code=hs_code)
    structural = _structural_numbers(*tables)

    results: list[tuple[float, bool]] = []
    for match in _NUMBER_PATTERN.finditer(prose):
        value = float(match.group().replace(",", ""))
        rounded = round(value)
        preceding = prose[: match.start()]
        following = prose[match.end() :]

        if _FORBIDDEN_SUFFIX_PATTERN.match(following) or _FORBIDDEN_PRECEDING_PATTERN.search(
            preceding
        ):
            results.append((value, False))
            continue

        if rounded in value_grounded:
            results.append((value, True))
            continue

        is_structural_context = _STRUCTURAL_PRECEDING_PATTERN.search(
            preceding
        ) or _STRUCTURAL_FOLLOWING_PATTERN.match(following)
        results.append((value, rounded in structural and bool(is_structural_context)))

    return results


def check_numbers_grounded(prose: str, *tables: TradeTable, hs_code: str | None = None) -> bool:
    """Return True iff every number in `prose` is present in `tables`
    (docs/PLAN.md §6, master brief §2.2/§8) — the deterministic,
    non-negotiable "the LLM never produces a number" check. No LLM judge:
    a plain extraction + set-membership comparison, tolerant only of
    whole-unit rounding (see `_scan_numbers`)."""
    return all(grounded for _, grounded in _scan_numbers(prose, *tables, hs_code=hs_code))


def find_ungrounded_numbers(
    prose: str, *tables: TradeTable, hs_code: str | None = None
) -> list[float]:
    """Like `check_numbers_grounded`, but returns the offending numbers
    instead of a bool — used for actionable error messages/logging when the
    guardrail trips (docs/PLAN.md §3.2: errors are user-safe but should
    still be diagnosable server-side)."""
    return [
        value for value, grounded in _scan_numbers(prose, *tables, hs_code=hs_code) if not grounded
    ]
