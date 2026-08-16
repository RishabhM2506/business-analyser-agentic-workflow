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

Two responsibilities, deliberately kept separate:

1. `build_trace_metadata` — a pure function building the `metadata` dict
   passed into `config={"metadata": ...}` on every `.ainvoke()` call
   (`app/main.py`). This dict is attached to a run's config regardless of
   whether tracing is actually turned on; it costs nothing when it isn't.
2. `configure_langsmith_tracing` — called once at app startup
   (`app/main.py`'s `lifespan`) to turn tracing *on* at all, by propagating
   `Settings`' typed fields into the real process-environment variables the
   installed `langsmith` SDK reads. Verified directly against the installed
   `langsmith==0.11.0` package (`langsmith.utils.get_env_var`,
   `langsmith.utils.tracing_is_enabled`): it checks `LANGSMITH_TRACING_V2` →
   `LANGSMITH_TRACING` → `LANGCHAIN_TRACING_V2` → `LANGCHAIN_TRACING` (first
   non-empty wins), and separately reads `LANGSMITH_API_KEY`/
   `LANGSMITH_PROJECT` (also with a `LANGCHAIN_*` fallback) when actually
   submitting a trace. Setting `LANGSMITH_TRACING=true` here (not `_V2`)
   relies on that documented fallback, so the SDK's own already-fully-general
   env-var precedence is authoritative — this module does not reimplement
   or second-guess it, only feeds it. When tracing is off (the default:
   `settings.langsmith_tracing_enabled=False` or no API key configured),
   this function does nothing — no env vars are set, no client is
   constructed, no network call is attempted, so nothing to no-op around:
   the installed SDK's own `tracing_is_enabled()` already returns `False`
   for "no relevant env var set," which is exactly the state this function
   leaves the process in.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from app.settings import Settings


class TraceMetadata(TypedDict):
    """Metadata attached to every LangSmith trace for a graph invocation."""

    tenant_id: str
    user_id: str
    prompt_version: str
    release_sha: str


def build_trace_metadata(
    *, tenant_id: str, user_id: str, prompt_version: str, release_sha: str
) -> TraceMetadata:
    """Assemble the LangSmith trace metadata dict attached to a graph run.

    Pure and side-effect-free: safe to call unconditionally on every
    request regardless of whether tracing is enabled (see module
    docstring) — `app/main.py` does exactly that.
    """
    return TraceMetadata(
        tenant_id=tenant_id,
        user_id=user_id,
        prompt_version=prompt_version,
        release_sha=release_sha,
    )


def configure_langsmith_tracing(settings: Settings) -> None:
    """Turn on LangSmith tracing for the rest of this process, if
    configured to. Called once at app startup (`app/main.py`'s `lifespan`).

    No-ops cleanly or when `langsmith_tracing_enabled` is `False` or
    `langsmith_api_key` is unset (both true by default,
    `app/settings.py`) — no environment variables are touched, so the
    installed `langsmith`/`langchain-core` tracing machinery stays exactly
    as inert as if this module didn't exist. Never raises: this is
    best-effort observability wiring, not a correctness dependency, and
    must not be able to take the app down at startup.
    """
    if not settings.langsmith_tracing_enabled or not settings.langsmith_api_key:
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
