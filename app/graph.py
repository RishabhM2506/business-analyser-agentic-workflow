"""StateGraph assembly (docs/PLAN.md §2.2): wires the nodes in `app/nodes/`
into the fixed v1 pipeline (`validate_query` -> `fetch_imports`/
`fetch_exports` -> `aggregate` -> `retrieve_description` -> `describe_item`
-> `summarize` -> `assemble_response`) and compiles with a checkpointer
(`SqliteSaver` locally, `PostgresSaver` in any deployed environment).

`assemble_response` has no dedicated file under `app/nodes/` (docs/PLAN.md
§4.1's tree lists it only in the node table, §2.2) — it's simple enough to
live alongside the graph assembly that calls it.

# TODO(Phase 3): implement `build_graph()` — add all nodes/edges per the
# fixed pipeline above, and `assemble_response`'s body.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph

from app.state import AnalysisState


def assemble_response(state: AnalysisState) -> dict[str, Any]:
    """Final node: assemble `TradeAnalysisResponse` from everything upstream."""
    raise NotImplementedError  # TODO(Phase 3): implement.


def build_graph() -> StateGraph[AnalysisState]:
    """Assemble and return the (uncompiled) v1 `StateGraph`.

    Compilation (`.compile(checkpointer=...)`) is deliberately left to the
    caller (docs/PLAN.md §2.2: `SqliteSaver` locally, `PostgresSaver` in any
    deployed environment — "one line" to swap, per the guide's Ch.18 claim
    cited in PLAN.md §2.1).
    """
    raise NotImplementedError  # TODO(Phase 3): implement per docs/PLAN.md §2.2's fixed pipeline.
