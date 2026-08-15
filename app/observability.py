"""LangSmith tracing wiring (docs/PLAN.md §4.1, §5.6): attaches trace
metadata — `tenant_id`, `prompt_version`, release SHA — to every graph run,
so a prompt edit or a deploy is visible in trace history without cross-
referencing a deploy log.

Note: this module is about *LangGraph run* tracing metadata specifically.
The generic structured JSON request/response logging (with request-ID
propagation) used by every HTTP request is implemented directly in
`app/main.py` — it's a cross-cutting HTTP-layer concern independent of
whether the graph exists yet, whereas this module's job only makes sense
once `app/graph.py` has real invocations to attach metadata to.

# TODO(Phase 3): implement `build_trace_metadata` and pass it into
# `graph.py`'s `.invoke(..., config={"metadata": ...})` call.
"""

from __future__ import annotations

from typing import TypedDict


class TraceMetadata(TypedDict):
    """Metadata attached to every LangSmith trace for a graph invocation."""

    tenant_id: str
    user_id: str
    prompt_version: str
    release_sha: str


def build_trace_metadata(
    *, tenant_id: str, user_id: str, prompt_version: str, release_sha: str
) -> TraceMetadata:
    """Assemble the LangSmith trace metadata dict attached to a graph run."""
    raise NotImplementedError  # TODO(Phase 3): implement once graph.py has real invocations.
