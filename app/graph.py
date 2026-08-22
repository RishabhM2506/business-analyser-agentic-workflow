"""StateGraph assembly (docs/PLAN.md §2.2): wires the nodes in `app/nodes/`
into the fixed v1 pipeline (`validate_query` -> `fetch_imports`/
`fetch_exports` -> `aggregate` -> `retrieve_description` -> `describe_item`
-> `summarize` -> `assemble_response`) and compiles with a checkpointer
(`SqliteSaver`-family locally, `PostgresSaver`-family in any deployed
environment).

`assemble_response` has no dedicated file under `app/nodes/` (docs/PLAN.md
§4.1's tree lists it only in the node table, §2.2) — it's simple enough to
live alongside the graph assembly that calls it.

**LangGraph API verified directly against the installed packages** (not
assumed from memory/docs), the same way `app/models.py` verified
`langchain-google-genai` — versions and verification method:

- `langgraph==1.2.11`, `langgraph-checkpoint==4.2.0`,
  `langgraph-checkpoint-sqlite==3.1.1`, `langgraph-checkpoint-postgres==3.1.2`
  (all pinned in `pyproject.toml`).
- `StateGraph.__init__`/`.add_node`/`.add_edge`/`.compile` signatures read
  directly via `inspect.signature`/`inspect.getsource` from the installed
  package. Confirmed: `StateGraph[AnalysisState]` is a valid subscript
  (`ContextT`/`InputT`/`OutputT` all carry PEP-696 `default=` type-var
  defaults in `langgraph.typing`, so a single type argument is enough).
  Confirmed via `add_node`'s docstring + a live smoke test: a bare callable
  passed as `node` (no explicit name) has its node-graph key inferred from
  the callable itself, matching every node function's own name
  (`validate_query`, `fetch_imports`, ... — see `app/nodes/*.py`).
- `add_edge(list[str], str)` (fan-in — "the graph will wait for ALL of the
  start nodes to complete before executing the end node") confirmed both by
  reading `StateGraph.add_edge`'s source and by a live smoke test mixing a
  sync + an async node fanning into one downstream node under `.ainvoke()`.
- `recursion_limit` confirmed as a `RunnableConfig` key (`langchain_core`'s
  `RunnableConfig` TypedDict — `typing.get_type_hints` lists it directly)
  rather than a separate keyword argument to `.ainvoke()` — it belongs in
  `config={"recursion_limit": ...}`, alongside `configurable` and
  `metadata`.
- `AsyncSqliteSaver`/`AsyncPostgresSaver` (`langgraph.checkpoint.sqlite.aio`
  / `langgraph.checkpoint.postgres.aio`): both `.from_conn_string(...)` are
  `@asynccontextmanager`s meant to be held open for the process's lifetime,
  not opened per call — confirmed by reading their source directly.
  `AsyncSqliteSaver`'s own CRUD methods (`aget_tuple`/`aput`/...) call
  `self.setup()` internally (idempotent — guarded by an `is_setup` flag), so
  it self-heals; `AsyncPostgresSaver`'s methods do NOT call `self.setup()`
  internally (grepped the installed source for `self.setup(` inside
  `aio.py` — zero matches, unlike sqlite's five), so `.setup()` must be
  called explicitly once before first use. `build_checkpointer` below calls
  it unconditionally for both, which is a no-op on sqlite and required on
  postgres.
- **Live-verified via a standalone smoke test** (mixed sync/async nodes,
  real fan-out/fan-in, a real in-memory `AsyncSqliteSaver`, `recursion_limit`
  + `metadata` in `config`, and `.aget_state()` on both a populated and a
  never-invoked `thread_id`): `.aget_state()` on a thread that has never run
  raises nothing and returns a `StateSnapshot` with `.values == {}` — this
  is what `GET /threads/{id}` (`app/main.py`) relies on to distinguish "no
  such thread" from "thread has a completed run."
- **Checkpoint serialization of our own Pydantic types is explicitly
  allowlisted**, for a concrete reason found by running the graph for real
  (not by reading docs): `AnalysisState` stores several `app.*` Pydantic
  models directly (`TradeQuery`, `ErrorResponse`, `ComtradeRecord`,
  `TradeTable`, `CountryRow`, `Provenance`, `TradeAnalysisResponse`), and
  the checkpointer's default `JsonPlusSerializer` logs, once per type,
  `"Deserializing unregistered type ... This will be blocked in a future
  version"` the first time it round-trips one through a real checkpoint —
  confirmed by an end-to-end smoke run against a real (in-memory)
  `AsyncSqliteSaver`. Today this is a warning, not a failure (the installed
  `langgraph-checkpoint==4.2.0`'s default is permissive: "all types allowed
  with a warning" unless `LANGGRAPH_STRICT_MSGPACK=true`), but the message
  is explicit that permissiveness is not guaranteed to stay the default.
  `_checkpoint_serde()` below passes every such type to
  `JsonPlusSerializer(allowed_msgpack_modules=[...])` explicitly (verified:
  the installed serializer's own `_normalize_module_keys` accepts live type
  objects directly — `(cls.__module__, cls.__name__)` is derived
  automatically, not hand-typed as brittle dotted strings), which silences
  the warning now and keeps working if the default ever flips.
- **`AsyncPostgresSaver` is deliberately imported lazily** (inside
  `build_checkpointer`, not at module top) for a concrete, verified reason:
  it imports `psycopg`, and a bare `import psycopg` (the exact dependency
  `langgraph-checkpoint-postgres` pulls in — no `[c]`/`[binary]` extra is
  pinned) **fails at import time** on a plain Debian slim image with no
  system `libpq` installed — reproduced directly against
  `python:3.12-slim` (this project's own Docker base image, see
  `Dockerfile`) with `pip install psycopg==3.3.4 && python -c "import
  psycopg"`: `ImportError: no pq wrapper available ... libpq library not
  found`. Installing the small `libpq5` runtime package (no compiler, no
  `-dev` headers) fixes it — verified the same way — and the `Dockerfile`'s
  runtime stage now installs it so a real `DATABASE_URL=postgresql://...`
  deployment works. But a lazy import means a sqlite-only deployment (the
  default, and every test in this repo) never needs `libpq` present at all,
  which is the safer default given most of this repo's environments
  (dev, CI, `docker build .`'s own verification) never touch Postgres.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy.engine import make_url

from app.nodes.aggregate import aggregate
from app.nodes.describe_item import PROMPT_VERSION as DESCRIBE_ITEM_PROMPT_VERSION
from app.nodes.describe_item import describe_item
from app.nodes.fetch_trade import fetch_exports, fetch_imports
from app.nodes.retrieve_description import retrieve_description
from app.nodes.summarize import PROMPT_VERSION as SUMMARIZE_PROMPT_VERSION
from app.nodes.summarize import summarize
from app.nodes.validate_query import validate_query
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.schemas.response import CountryRow, Provenance, TradeAnalysisResponse, TradeTable
from app.state import AnalysisState, FetchIssue, get_or_mint_trace_id, has_error
from app.tools.comtrade_client import ComtradeRecord

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# A response's `provenance.prompt_version` is a single string but two
# separately-versioned prompts feed one response (docs/PLAN.md §5.1: call 1
# is `describe_item`, call 2 is `summarize`) — combined so that bumping
# *either* prompt changes this string, which is what makes it usable as a
# response-cache-key component and a trace-metadata field that actually
# reflects "which exact prompts produced this" (docs/PLAN.md §5.4.2, §5.6).
COMBINED_PROMPT_VERSION = f"{DESCRIBE_ITEM_PROMPT_VERSION}+{SUMMARIZE_PROMPT_VERSION}"

# Every `app.*` Pydantic type that can appear as a value in `AnalysisState`
# and therefore gets checkpointed — see this module's docstring for why
# this list exists at all (a real warning found via a live smoke run, not a
# hypothetical). Kept as actual classes, not hand-typed dotted strings: the
# installed serializer derives `(module, qualname)` from the class object
# itself, so a future `git mv`/refactor that moves one of these classes
# can't silently desync this list from reality.
_CHECKPOINTED_APP_TYPES: list[type] = [
    TradeQuery,
    ErrorResponse,
    ComtradeRecord,
    TradeTable,
    CountryRow,
    Provenance,
    TradeAnalysisResponse,
    # 2026-08-20 roadmap decision (per-year graceful fetch degradation,
    # `app.nodes.fetch_trade`): stored directly in `AnalysisState`'s
    # `import_fetch_issues`/`export_fetch_issues`, so it needs the same
    # allowlisting as every other `app.*` Pydantic type above — same real,
    # live-reproduced warning this list already exists to silence.
    FetchIssue,
]


def _checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINTED_APP_TYPES)


def assemble_response(state: AnalysisState) -> dict[str, Any]:
    """Final node: assemble `TradeAnalysisResponse` from everything upstream.

    If `state["error"]` is already set (any earlier node failed), no-ops —
    consistent with every other node's `has_error(state)` short-circuit
    (docs/PLAN.md §2.2: v1's fixed pipeline has no conditional edges, so a
    failed node upstream doesn't stop later nodes from *running*, only from
    doing anything).

    Otherwise, if any required upstream field is still missing despite no
    `error` being set, this deliberately does NOT no-op the same way
    intermediate nodes do. Every intermediate node's defensive `return {}`
    is safe only because a *later* node re-checks the same precondition and
    no-ops identically — the no-op just propagates forward. This is the
    terminal node: propagating a silent `{}` here would leave
    `app/main.py` with neither `response` nor `error` to return, which is
    exactly the "silent partial render" docs/PLAN.md §3.2 forbids. So this
    branch synthesizes an explicit `ErrorResponse` instead — the one place
    in the pipeline that turns "this should never happen" into a real,
    user-safe error rather than a silent no-op.
    """
    if has_error(state):
        return {}

    query = state.get("query")
    item_description = state.get("item_description")
    analytical_summary = state.get("analytical_summary")
    imports_table = state.get("imports_table")
    exports_table = state.get("exports_table")
    thread_id = state.get("thread_id")
    message_id = state.get("message_id")

    if (
        query is None
        or item_description is None
        or analytical_summary is None
        or imports_table is None
        or exports_table is None
        or thread_id is None
        or message_id is None
    ):
        return {
            "error": ErrorResponse(
                error_code="INCOMPLETE_STATE",
                message="The analysis could not be completed due to an internal error.",
                retryable=True,
                trace_id=get_or_mint_trace_id(state),
            )
        }

    response = TradeAnalysisResponse(
        thread_id=thread_id,
        message_id=message_id,
        hs_code=query.hs_code,
        item_description=item_description,
        imports=imports_table,
        exports=exports_table,
        analytical_summary=analytical_summary,
        provenance=Provenance(
            source="UN Comtrade (comtradeapi.un.org)",
            retrieved_at=datetime.now(UTC),
            period_type="calendar_year",
            currency="USD",
            prompt_version=COMBINED_PROMPT_VERSION,
            # Finding M22/PBO-04: matches the fixed reporter every fetch in
            # this system actually uses (`INDIA_REPORTER_CODE`,
            # app/tools/comtrade_client.py) — stated explicitly rather than
            # left for the user to infer.
            reporter_country="India",
        ),
    )
    return {"response": response}


def build_graph() -> StateGraph[AnalysisState]:
    """Assemble and return the (uncompiled) v1 `StateGraph`.

    Compilation (`.compile(checkpointer=...)`) is deliberately left to the
    caller (docs/PLAN.md §2.2: `SqliteSaver`-family locally,
    `PostgresSaver`-family in any deployed environment — "one line" to
    swap, per the guide's Ch.18 claim cited in PLAN.md §2.1).

    Every node function is passed bare (no explicit name string): each
    one's own `__name__` already matches the node-table name in
    docs/PLAN.md §2.2 exactly, and `StateGraph.add_node` infers the graph
    key from the callable when `node` isn't a string (verified — see this
    module's docstring). `assemble_response` is the one exception (defined
    above, not imported from `app/nodes/`), given an explicit name for the
    same reason it would get one implicitly: `assemble_response.__name__`
    already equals the desired key, so passing it explicitly is purely
    for symmetry/readability with the fan-out/fan-in edges below, not a
    functional necessity.
    """
    graph: StateGraph[AnalysisState] = StateGraph(AnalysisState)

    graph.add_node(validate_query)
    graph.add_node(fetch_imports)
    graph.add_node(fetch_exports)
    graph.add_node(aggregate)
    graph.add_node(retrieve_description)
    graph.add_node(describe_item)
    graph.add_node(summarize)
    graph.add_node("assemble_response", assemble_response)

    graph.add_edge(START, "validate_query")
    # Fan-out: both fetch nodes depend only on `validate_query` completing,
    # and have no data dependency on each other (docs/PLAN.md §2.2) — two
    # separate edges from the same source run as a parallel superstep, not
    # sequentially.
    graph.add_edge("validate_query", "fetch_imports")
    graph.add_edge("validate_query", "fetch_exports")
    # Fan-in: `aggregate` waits for BOTH fetch nodes (a list `start_key`,
    # not two separate `add_edge` calls into the same destination).
    graph.add_edge(["fetch_imports", "fetch_exports"], "aggregate")
    graph.add_edge("aggregate", "retrieve_description")
    graph.add_edge("retrieve_description", "describe_item")
    graph.add_edge("describe_item", "summarize")
    graph.add_edge("summarize", "assemble_response")
    graph.add_edge("assemble_response", END)

    return graph


def _checkpointer_conn_info(database_url: str) -> tuple[str, str]:
    """Translate `settings.database_url` (a SQLAlchemy-style URL, e.g.
    `sqlite+aiosqlite:///./local.db` or
    `postgresql+asyncpg://user:pass@host:5432/db` — see `app/settings.py`
    and `.env.example`) into the plain, driver-specific connection string
    LangGraph's checkpointer savers expect.

    This translation is needed because the checkpointer packages do NOT use
    SQLAlchemy at all: `AsyncSqliteSaver` wraps `aiosqlite.connect(...)`
    directly (which wants a bare file path or `:memory:`, not a SQLAlchemy
    URL), and `AsyncPostgresSaver` wraps `psycopg` (not `asyncpg` — a
    different driver from the one `app/main.py`'s own `check_database`
    healthcheck uses via SQLAlchemy's `create_async_engine`). This is an
    inherent consequence of using LangGraph's official Postgres checkpointer
    package as-is rather than hand-rolling one against SQLAlchemy, not an
    oversight — both drivers end up installed regardless
    (`langgraph-checkpoint-postgres` pulls in `psycopg`; `asyncpg` is
    already a direct dependency for the healthcheck).

    Returns `(backend, conn_info)` where `backend` is `"sqlite"` or
    `"postgres"`. Raises `ValueError` for anything else — fails loudly
    (matching `app/settings.py`'s own stated philosophy) rather than
    silently defaulting to one backend for an unrecognized URL.
    """
    url = make_url(database_url)
    backend = url.get_backend_name()
    if backend == "sqlite":
        return "sqlite", url.database or ":memory:"
    if backend == "postgresql":
        # Reconstruct a plain `postgresql://` conninfo string via
        # SQLAlchemy's own URL renderer (handles special characters in the
        # password correctly) rather than hand-formatting one — `psycopg`
        # doesn't understand the `+asyncpg` driver suffix in
        # `settings.database_url`, only the bare `postgresql://` scheme.
        pg_url = url.set(drivername="postgresql")
        return "postgres", pg_url.render_as_string(hide_password=False)
    raise ValueError(
        f"Unsupported database_url backend for the checkpointer: {backend!r} "
        "(expected a sqlite or postgresql URL)"
    )


@asynccontextmanager
async def build_checkpointer(
    database_url: str,
) -> AsyncIterator[AsyncSqliteSaver | AsyncPostgresSaver]:
    """Open the checkpointer appropriate for `database_url` and hold it open
    for the lifetime of the `async with` block — intended to be entered
    once, at app startup (`app/main.py`'s `lifespan`), and held for the
    life of the process, not opened per-request (see this module's
    docstring for why: both savers' `.from_conn_string` are themselves
    long-lived async context managers, not per-call helpers).
    """
    backend, conn_info = _checkpointer_conn_info(database_url)
    if backend == "sqlite":
        # `AsyncSqliteSaver.from_conn_string` (unlike the Postgres saver's)
        # doesn't accept a `serde=` override — verified by reading its
        # signature directly — so this replicates its own implementation
        # (`async with aiosqlite.connect(conn_string) as conn: yield
        # cls(conn)`, confirmed from source) with our custom serializer
        # passed through the one extra constructor kwarg it exposes.
        async with aiosqlite.connect(conn_info) as conn:
            saver = AsyncSqliteSaver(conn, serde=_checkpoint_serde())
            await saver.setup()
            yield saver
        return

    # Lazy import: see this module's docstring for the concrete, verified
    # reason (bare `psycopg` fails to import on a slim image with no system
    # `libpq`) — a sqlite-only process must never pay that cost.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(
        conn_info, serde=_checkpoint_serde()
    ) as pg_saver:
        await pg_saver.setup()
        yield pg_saver
