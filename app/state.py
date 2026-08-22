"""`AnalysisState` — the LangGraph state schema for the v1 workflow.

Field set mirrors the per-node input/output columns in docs/PLAN.md §2.2
exactly, so each node's signature (`app/nodes/*.py`) can be read directly
against that table.
"""

from __future__ import annotations

import uuid
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict

from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.schemas.response import TradeAnalysisResponse, TradeTable
from app.tools.comtrade_client import ComtradeRecord


class FetchIssue(BaseModel):
    """One (year, reason) pair where `app.nodes.fetch_trade` ultimately
    failed to fetch that year after every retry attempt — `reason` is the
    real caught exception's own message (`str(exc)`), never a paraphrase.

    A Pydantic model, not a plain `@dataclass` (live-reproduced 2026-08-21:
    a plain dataclass here made `GET /threads/{id}` silently fail to
    deserialize this field back out of the checkpointer —
    `langgraph.checkpoint.serde.jsonplus`'s msgpack serde only reconstructs
    a custom type via an `allowed_msgpack_modules` allowlist arbitrary
    dataclasses aren't on; Pydantic models get native, no-allowlist-needed
    support). Matches `app.tools.comtrade_client.ComtradeRecord` — the same
    "small typed value stored in `AnalysisState` and round-tripped through
    the checkpointer" shape, solved the same way there already.

    Defined here (not in `app.nodes.fetch_trade`, where it's constructed)
    so both that module and `app.nodes.aggregate` can import it from this
    shared, dependency-free module without either importing the other —
    `app.nodes.fetch_trade` already imports `AnalysisState`/`has_error`
    from this module, so the reverse import would be circular."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year: int
    reason: str


def _keep_first_error(
    existing: ErrorResponse | None, new: ErrorResponse | None
) -> ErrorResponse | None:
    """Reducer for the `error` channel (see its `Annotated[...]` below for
    why one is needed at all): when `fetch_imports` and `fetch_exports`
    — a genuine parallel fan-out, docs/PLAN.md §2.2 — both fail in the
    same superstep (a systemic Comtrade outage fails both at once, not
    just one flow), LangGraph's *default* per-key channel (`LastValue`)
    rejects the second concurrent write to the same key outright
    (`InvalidUpdateError: Can receive only one value per step`) — this was
    reproduced live via a full graph invocation against an always-timing-out
    transport, not a hypothetical. Marking `error` with this reducer instead
    makes it a `BinaryOperatorAggregate` channel, which folds concurrent
    writes through this function two at a time instead of rejecting them.

    Keeping whichever real error was written first is an arbitrary but
    deterministic-enough choice: only one `ErrorResponse` can ever reach the
    user regardless (`app/main.py` returns exactly one), and every node
    after the point of failure already no-ops via `has_error()` no matter
    which upstream error specifically "won" the merge.

    A `new` value of `None` is a deliberate **reset** signal, not "no
    update," and is honored unconditionally, even over a real `existing`
    error (PBO-01/QA-01, finding B3): `app/main.py`'s `post_message` seeds
    `initial_state["error"] = None` on *every* invocation, specifically so a
    fresh request's first superstep can clear whatever `error` was
    checkpointed by a *previous* run on this same `thread_id`. Without this,
    a `BinaryOperatorAggregate` channel simply keeps whatever was last
    written forever — nothing in the fixed v1 pipeline otherwise ever writes
    `error` on a run that succeeds, so the *old* run's error would silently
    outlive it, and every node would no-op via `has_error()` on every
    subsequent message regardless of that message's own `hs_code` (live-
    reproduced: identical `trace_id`, 27ms response = no real call made).
    Safe to honor unconditionally because a `None` write can only ever be
    this one seed value at superstep 0 — no node in this codebase ever
    returns `{"error": None}` — so it can never race against or clobber a
    real failure reported later in the *same* run (verified directly
    against the installed `langgraph`'s `BinaryOperatorAggregate.update`:
    the very first write to a never-before-set channel bypasses this
    reducer entirely, and every write after that goes through it in order,
    so a `None` seed followed by a real node error in a later superstep of
    the same run still correctly ends up with that real error, not `None`).
    """
    if new is None:
        return None
    if existing is None:
        return new
    return existing


class AnalysisState(TypedDict, total=False):
    """Shared state threaded through every node of the v1 workflow.

    `total=False`: a given superstep only ever writes a subset of these
    keys (docs/PLAN.md §2.2) — LangGraph merges partial node returns into
    this dict across supersteps.
    """

    # Seeded by the caller (`app/main.py`) at `.ainvoke()` time from the same
    # UUID4 already bound as the HTTP request's `X-Request-ID`
    # (`request_id_middleware`) — not written by any node, only read, so any
    # `ErrorResponse` a node constructs correlates with the request's own
    # structured logs rather than minting an independent, uncorrelated id.
    trace_id: str

    # Also seeded by the caller, same "written once, read-only" contract as
    # `trace_id` above: `thread_id` is the `POST /threads/{id}/messages` path
    # parameter (the LangGraph checkpointer's own thread key, so it's already
    # available via `config["configurable"]["thread_id"]` too — duplicated
    # into state so every node can read it the same plain way as every other
    # input rather than half the nodes taking a second `RunnableConfig`
    # argument); `message_id` is a fresh id minted per `POST .../messages`
    # call. Both are read by `describe_item`/`summarize` (per-thread budget
    # accounting, `app/budget.py`) and by `assemble_response` (the response
    # envelope's `thread_id`/`message_id` fields, docs/PLAN.md §3.2).
    thread_id: str
    message_id: str

    query: TradeQuery
    # `ErrorResponse | None` (not just `ErrorResponse`): `app/main.py` seeds
    # this explicitly to `None` on every invocation as a reset signal (see
    # `_keep_first_error`'s docstring above, finding B3) — the type must
    # actually admit that value, not just tolerate it as an absent key.
    error: Annotated[ErrorResponse | None, _keep_first_error]

    raw_imports: list[ComtradeRecord]
    raw_exports: list[ComtradeRecord]
    # Per-year fetch failures that survived every retry attempt (2026-08-20
    # roadmap decision: graceful degradation over a hard request failure —
    # see `app.nodes.fetch_trade`'s module docstring). Always written
    # alongside its sibling `raw_*` key, empty in the common case.
    import_fetch_issues: list[FetchIssue]
    export_fetch_issues: list[FetchIssue]

    imports_table: TradeTable
    exports_table: TradeTable

    taxonomy_text: str

    item_description: str
    analytical_summary: str

    response: TradeAnalysisResponse


def get_or_mint_trace_id(state: AnalysisState) -> str:
    """`state["trace_id"]` if the caller seeded one (the normal case — see
    the field's docstring above), else a freshly-minted UUID4 — shared by
    every node that constructs an `ErrorResponse`, so there's exactly one
    fallback policy instead of five slightly-different copies."""
    return state.get("trace_id") or str(uuid.uuid4())


def has_error(state: AnalysisState) -> bool:
    """True iff an earlier node already wrote `error` — every node after
    `validate_query` checks this first and no-ops if so (docs/PLAN.md §2.2:
    v1 has no conditional edges, so every node in the fixed pipeline runs
    regardless; a failed validation is a no-op for the rest of the
    pipeline, not a graph-level branch)."""
    return state.get("error") is not None
