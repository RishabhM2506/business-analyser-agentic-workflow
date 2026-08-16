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
_NUMBER_PATTERN = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


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
    canned summaries that are grounded by construction (see `app/models.py`)."""
    return [float(match.replace(",", "")) for match in _NUMBER_PATTERN.findall(text)]


def _flatten_table_numbers(*tables: TradeTable) -> set[int]:
    """Every number that legitimately belongs in prose about `tables`,
    rounded to the nearest whole unit for tolerant comparison (a model
    copying "1,992,456" for a table value of 1992455.942 must not fail the
    check over sub-dollar rounding).

    Includes, beyond the raw trade-value cells: every `years` entry (a
    summary saying "in 2023" is referencing structure, not fabricating a
    figure) and every row's `rank` (same reasoning for "the #1 partner").
    Also includes `len(table.years)` and `len(table.rows)` — the fixed
    methodology constants (5-year window, top-10 ranking, docs/PLAN.md §2.2)
    that any reasonable summary will mention as prose ("over the past 5
    years", "the top 10 partners") — these are structural facts about how
    the table was built, true by construction, not data the model could
    fabricate incorrectly, so treating them as "not grounded" would make
    the guardrail reject essentially every legitimate summary.
    """
    grounded: set[int] = set()
    for table in tables:
        grounded.update(table.years)
        grounded.add(len(table.years))
        grounded.add(len(table.rows))
        for row in table.rows:
            grounded.add(row.rank)
            grounded.add(round(row.cumulative_5yr))
            for value in row.values_by_year.values():
                if value is not None:
                    grounded.add(round(value))
    return grounded


def check_numbers_grounded(prose: str, *tables: TradeTable) -> bool:
    """Return True iff every number in `prose` is present in `tables`
    (docs/PLAN.md §6, master brief §2.2/§8) — the deterministic,
    non-negotiable "the LLM never produces a number" check. No LLM judge:
    a plain extraction + set-membership comparison, tolerant only of
    whole-unit rounding (see `_flatten_table_numbers`)."""
    grounded = _flatten_table_numbers(*tables)
    return all(round(number) in grounded for number in extract_numbers(prose))


def find_ungrounded_numbers(prose: str, *tables: TradeTable) -> list[float]:
    """Like `check_numbers_grounded`, but returns the offending numbers
    instead of a bool — used for actionable error messages/logging when the
    guardrail trips (docs/PLAN.md §3.2: errors are user-safe but should
    still be diagnosable server-side)."""
    grounded = _flatten_table_numbers(*tables)
    return [number for number in extract_numbers(prose) if round(number) not in grounded]
