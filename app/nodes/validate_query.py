"""`validate_query` node: parses the raw request body into `TradeQuery` and
runs the `hs_code`-against-taxonomy allowlist check (docs/PLAN.md §2.2,
§6). Runs before any node that could reach the network or a model —
PLAN.md §1.1's "nothing reaches a model before validation" ordering.

# TODO(Phase 3): implement — parse the raw request body into `TradeQuery`,
# call `app.guardrails.check_hs_code_allowlisted`, and write `query` or
# `error` into state accordingly.
"""

from __future__ import annotations

from typing import Any

from app.state import AnalysisState


def validate_query(state: AnalysisState) -> dict[str, Any]:
    """Validate the incoming query; writes `query` on success or `error` on
    failure. No model, no I/O — pure validation + allowlist lookup."""
    raise NotImplementedError  # TODO(Phase 3): implement as described above.
