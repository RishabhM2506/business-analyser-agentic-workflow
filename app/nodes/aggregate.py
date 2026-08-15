"""`aggregate` node: pure functions — strip aggregate/"nes" partner codes,
rank top-10 by cumulative 5yr value, pivot into a 5-year table, flag year
completeness (docs/PLAN.md §2.2, §6). Deterministic Python only: the model
never sees, and therefore cannot mis-transcribe or invent, a ranking
decision (docs/PLAN.md §6).

This is the module docs/PLAN.md §7 calls out for exhaustive unit testing
against hand-computed fixtures — no I/O, no model, ever.

# TODO(Phase 3): implement the top-10/5yr-pivot/completeness-flag logic
# described above.
"""

from __future__ import annotations

from typing import Any

from app.state import AnalysisState


def aggregate(state: AnalysisState) -> dict[str, Any]:
    """Turn `raw_imports`/`raw_exports` into `imports_table`/`exports_table`."""
    raise NotImplementedError  # TODO(Phase 3): implement.
