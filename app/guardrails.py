"""Input/output guardrails (docs/PLAN.md §6, master brief §8): controls
live in code, not prompts.

- Input: `hs_code` format + taxonomy-membership allowlist check, run in
  `validate_query` before any node reaches the network or a model.
- Output: every number appearing in `analytical_summary` must be a member
  of the flattened `imports`/`exports` table values before
  `assemble_response` runs — the concrete v1 instance of "the LLM never
  produces a number" (master brief §2.2), enforced structurally.

# TODO(Phase 3): implement `check_hs_code_allowlisted` against the taxonomy
# CSV and `check_numbers_grounded` as a deterministic number-extraction +
# set-membership check (docs/PLAN.md §7 — no LLM judge here).
"""

from __future__ import annotations

from app.schemas.response import TradeTable


def check_hs_code_allowlisted(
    hs_code: str, *, taxonomy_path: str = "data/harmonized-system.csv"
) -> bool:
    """Return True iff `hs_code` exists in the checked-in HS6 taxonomy."""
    raise NotImplementedError  # TODO(Phase 3): implement allowlist lookup.


def check_numbers_grounded(prose: str, *tables: TradeTable) -> bool:
    """Return True iff every number in `prose` is present in `tables`."""
    raise NotImplementedError  # TODO(Phase 3): implement extraction + membership check.
