"""FastAPI application entrypoint.

Phase 3 scope: `/`, `/healthz`, and the full thread/message API
(docs/PLAN.md §3.3) — `POST /threads`, `GET /threads/{id}`,
`POST /threads/{id}/messages` — are all real. The latter three invoke the
LangGraph workflow assembled in `app/graph.py`.

`/healthz` genuinely verifies the configured database is reachable — it
does not return an unconditional 200 (master brief §2 backend specifics,
docs/PLAN.md §3.3).

Guardrail ordering (docs/PLAN.md §1.1's Gateway -> Guard -> Graph, master
brief's "nothing reaches a model before validation"):
1. FastAPI parses the request body into `TradeQuery` (or a malformed body
   is turned into an `ErrorResponse` by `handle_validation_error` below,
   never FastAPI's default `{"detail": [...]}` shape — docs/PLAN.md §3.2:
   *every* response, success or error, is schema-validated).
2. `check_hs_code_allowlisted` runs again here, before the graph is even
   invoked — belt-and-suspenders on top of `validate_query`'s identical
   in-graph check (`app/nodes/validate_query.py`), not a replacement for
   it: this is cheap (an in-process, `lru_cache`d CSV lookup) and means an
   adversarial/invalid code is rejected before a checkpoint write or any
   graph superstep, not just before a model call.
3. The model-call budget is checked inside the two model-call nodes
   themselves (`describe_item`/`summarize`, `app/budget.py`), immediately
   before each call — not redundantly pre-checked here. A pre-check here
   would have to either skip incrementing (pointless) or increment for a
   call that hasn't happened yet (double-counts against calls actually
   made inside the two nodes) — checking inside each node is what actually
   guarantees "no model spend past the ceiling" (docs/PLAN.md §5.5), and
   the free, deterministic nodes ahead of them still run either way.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.graph import COMBINED_PROMPT_VERSION, build_checkpointer, build_graph
from app.guardrails import check_hs_code_allowlisted
from app.observability import build_trace_metadata, configure_langsmith_tracing
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.schemas.response import TradeAnalysisResponse
from app.settings import Settings, get_settings
from app.state import AnalysisState

REQUEST_ID_HEADER = "X-Request-ID"

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

# error_code -> HTTP status. Every code any node/endpoint can currently
# produce is listed explicitly; an unrecognized code (should not happen —
# every `ErrorResponse` in this codebase is constructed with a code from
# this set) falls back to 500 rather than guessing.
_ERROR_STATUS_CODES: dict[str, int] = {
    "INVALID_QUERY": 400,
    "INVALID_HS_CODE": 400,
    "UPSTREAM_TIMEOUT": 504,
    "UPSTREAM_UNAVAILABLE": 503,
    "UPSTREAM_SCHEMA_INVALID": 502,
    "UPSTREAM_ERROR": 502,
    "UNGROUNDED_SUMMARY": 502,
    "BUDGET_EXCEEDED": 429,
    "INCOMPLETE_STATE": 500,
    "SCHEMA_VALIDATION_FAILED": 500,
    "INTERNAL_ERROR": 500,
    "THREAD_NOT_FOUND": 404,
}


def _status_code_for_error(error_code: str) -> int:
    return _ERROR_STATUS_CODES.get(error_code, 500)


def _model_response(
    model: TradeAnalysisResponse | ErrorResponse, *, status_code: int
) -> JSONResponse:
    """Every success/error response is sent via this one path so schema
    validation (already guaranteed by `model` being a real Pydantic
    instance, not a hand-built dict) and the status-code mapping can never
    drift apart across the three thread endpoints (docs/PLAN.md §3.2: never
    a silent partial render).

    Wrapped in the `{"type": "final", "data": ...}` envelope docs/PLAN.md
    §3.3 documents ("the wire format is an envelope with a `type: 'final'`
    discriminator so a future `type: 'delta'` streaming chunk is additive,
    not breaking") — ARCH-01/B1: this was specified but never implemented,
    which broke every real analysis request against the frontend's own
    (already-built, streaming-ready) envelope consumer. `data` carries
    exactly what used to be the bare top-level body, so `TradeAnalysisResponse`
    and `ErrorResponse` themselves are unchanged."""
    return JSONResponse(
        status_code=status_code,
        content={"type": "final", "data": model.model_dump(mode="json")},
    )


def _current_trace_id() -> str:
    """The request-scoped id already bound by `request_id_middleware`
    (falls back to a fresh UUID4 if called outside a request, e.g. a
    process-level error path) — one id per request, echoed on the
    `X-Request-ID` header and threaded through every `ErrorResponse`."""
    return structlog.contextvars.get_contextvars().get("request_id") or str(uuid.uuid4())


def configure_logging(*, json_logs: bool, log_level: str) -> None:
    """Configure stdlib logging + structlog so every log line is structured
    JSON (or, outside `json_logs`, a human-readable console renderer) and
    automatically carries whatever `structlog.contextvars` has bound —
    notably the per-request `request_id` set by `request_id_middleware`.

    We use structlog (rather than a hand-rolled `logging.Formatter`
    subclass) because its contextvars integration is exactly what's needed
    to thread a request ID through every log line emitted during a request
    without passing a logger instance down every call chain. Documented in
    README.md's "Logging" section.
    """
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Generate (or propagate an inbound) UUID4 request ID into every log
    line emitted while handling this request, and echo it back on the
    `X-Request-ID` response header."""
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    start = time.perf_counter()
    logger.info("request.started", method=request.method, path=request.url.path)
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers[REQUEST_ID_HEADER] = request_id
    logger.info(
        "request.finished",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


async def check_database(database_url: str) -> None:
    """Open a real connection against `database_url` and run a trivial
    query. Raises on any failure — callers decide what that means for their
    response status."""
    engine = create_async_engine(database_url, pool_pre_ping=True)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    finally:
        await engine.dispose()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Read from `app.state.settings`, not the process-wide `get_settings()`
    # singleton: `create_app(settings=...)` accepts a settings override
    # specifically so tests can run against isolated config (e.g. an
    # in-memory sqlite `database_url`) without mutating process environment
    # variables (see `create_app`'s own docstring) — every route already
    # closes over `resolved_settings` for exactly this reason, so `lifespan`
    # must use the same instance, not silently fall back to the real
    # process settings underneath a test that thought it had overridden them.
    settings: Settings = app.state.settings
    configure_logging(json_logs=settings.log_json, log_level=settings.log_level)
    configure_langsmith_tracing(settings)
    logger.info(
        "app.startup",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
    )
    # Held open for the process's lifetime (docs/PLAN.md §2.2) — see
    # `app.graph.build_checkpointer`'s docstring for why this must be
    # entered once here rather than per-request.
    async with build_checkpointer(settings.database_url) as checkpointer:
        app.state.compiled_graph = build_graph().compile(checkpointer=checkpointer)
        yield
    logger.info("app.shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application. `settings` is accepted as an override
    so tests can inject settings without mutating process environment
    variables."""
    resolved_settings = settings or get_settings()

    app = FastAPI(
        title="Business Analyser — Agentic Workflow API",
        summary="Grounded import/export trade-data analysis for a selected HS code.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings

    app.middleware("http")(request_id_middleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """A shape-invalid request body (e.g. bad `hs_code` pattern, an
        extra field rejected by `TradeQuery`'s `extra="forbid"`) must still
        come back as our own `ErrorResponse` schema, not FastAPI's default
        `{"detail": [...]}` — docs/PLAN.md §3.2: *every* response, success
        or error, is schema-validated on the way out."""
        return _model_response(
            ErrorResponse(
                error_code="INVALID_QUERY",
                message="The request body did not match the expected shape.",
                retryable=False,
                trace_id=_current_trace_id(),
            ),
            status_code=400,
        )

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "service": "business-analyser-agentic-workflow",
            "status": "ok",
            "phase": "3-implementation",
        }

    @app.get("/healthz")
    async def healthz(response: Response) -> dict[str, object]:
        checks: dict[str, str] = {}
        healthy = True
        try:
            await check_database(resolved_settings.database_url)
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = "unreachable"
            healthy = False
            logger.warning("healthz.database_check_failed", error=str(exc))

        response.status_code = 200 if healthy else 503
        return {"status": "ok" if healthy else "unhealthy", "checks": checks}

    @app.post("/threads", status_code=201)
    async def create_thread() -> dict[str, str]:
        """Creates a thread (docs/PLAN.md §3.3: "on 'Start my process'").
        Nothing is persisted yet — the checkpointer lazily creates real
        state for a `thread_id` the first time `POST /threads/{id}/messages`
        actually invokes the graph on it (verified: `.aget_state()` on a
        never-invoked `thread_id` returns an empty snapshot rather than
        raising — see `app/graph.py`'s module docstring). This endpoint
        exists so the frontend has an id to address before the first
        message, not to register it in a separate durable store — there
        isn't one in v1 (no accounts, no auth, docs/PLAN.md §1.2)."""
        return {"thread_id": str(uuid.uuid4())}

    @app.get("/threads/{thread_id}")
    async def get_thread(thread_id: str, request: Request) -> Response:
        """Returns thread state for resume-after-refresh (docs/PLAN.md
        §3.3): the most recently completed analysis or error on this
        thread, or 404 if the thread has no completed run yet."""
        compiled_graph = request.app.state.compiled_graph
        snapshot = await compiled_graph.aget_state({"configurable": {"thread_id": thread_id}})
        values: AnalysisState = snapshot.values

        if not values:
            return _model_response(
                ErrorResponse(
                    error_code="THREAD_NOT_FOUND",
                    message=f"No thread found for id {thread_id!r}.",
                    retryable=False,
                    trace_id=_current_trace_id(),
                ),
                status_code=404,
            )

        error = values.get("error")
        if error is not None:
            return _model_response(error, status_code=_status_code_for_error(error.error_code))

        response_payload = values.get("response")
        if response_payload is not None:
            return _model_response(response_payload, status_code=200)

        # The thread exists (some state was checkpointed) but the pipeline
        # hasn't reached `assemble_response` yet — e.g. this deployment
        # restarted mid-run, or the ceiling was hit inline. Not a "success"
        # to resume into, but also not "no such thread": distinguished from
        # the truly-unknown-thread case above via `THREAD_INCOMPLETE`, not
        # reused as the same 404 code, so a client/observer can tell them
        # apart in logs/metrics even though both currently render as 404.
        return _model_response(
            ErrorResponse(
                error_code="THREAD_INCOMPLETE",
                message="This thread has no completed analysis yet.",
                retryable=True,
                trace_id=_current_trace_id(),
            ),
            status_code=404,
        )

    @app.post("/threads/{thread_id}/messages")
    async def post_message(thread_id: str, query: TradeQuery, request: Request) -> Response:
        """Invokes the graph on `thread_id` with a `TradeQuery`-shaped
        selection (docs/PLAN.md §3.3). Returns `TradeAnalysisResponse` on
        success or `ErrorResponse` on failure — always one or the other,
        schema-validated (see `_model_response`)."""
        settings: Settings = request.app.state.settings
        trace_id = _current_trace_id()

        if not check_hs_code_allowlisted(query.hs_code):
            return _model_response(
                ErrorResponse(
                    error_code="INVALID_HS_CODE",
                    message=f"{query.hs_code!r} is not a recognized HS6 code.",
                    retryable=False,
                    trace_id=trace_id,
                ),
                status_code=400,
            )

        compiled_graph = request.app.state.compiled_graph
        initial_state: AnalysisState = {
            "trace_id": trace_id,
            "thread_id": thread_id,
            "message_id": str(uuid.uuid4()),
            # Deliberately the raw, not-yet-year-resolved query: the
            # in-graph `validate_query` node resolves `year_start`/
            # `year_end` and re-checks the allowlist (see that node's own
            # docstring for exactly why that split exists).
            "query": query,
            # Explicit reset, every invocation (PBO-01/QA-01, finding B3):
            # without this, a `thread_id` that failed once stays poisoned
            # forever — `error`'s `_keep_first_error` reducer (app/state.py)
            # otherwise has no way to distinguish "no error yet" from "the
            # previous, unrelated message on this thread failed," and every
            # node no-ops via `has_error()` regardless of *this* request's
            # own `hs_code`. See `_keep_first_error`'s docstring for exactly
            # why `None` is safe to honor unconditionally here.
            "error": None,
        }
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.recursion_limit,
            "metadata": build_trace_metadata(
                tenant_id=query.tenant_id,
                user_id=query.user_id,
                prompt_version=COMBINED_PROMPT_VERSION,
                release_sha=settings.release_sha,
            ),
        }

        try:
            final_state: AnalysisState = await compiled_graph.ainvoke(initial_state, config=config)
        except Exception as exc:
            is_schema_failure = isinstance(exc, ValidationError)
            error_code = "SCHEMA_VALIDATION_FAILED" if is_schema_failure else "INTERNAL_ERROR"
            logger.exception(
                "thread.message.graph_invoke_failed", thread_id=thread_id, error_code=error_code
            )
            return _model_response(
                ErrorResponse(
                    error_code=error_code,
                    message="The analysis could not be completed due to an internal error.",
                    retryable=not is_schema_failure,
                    trace_id=trace_id,
                ),
                status_code=_status_code_for_error(error_code),
            )

        error = final_state.get("error")
        if error is not None:
            return _model_response(error, status_code=_status_code_for_error(error.error_code))

        response_payload = final_state.get("response")
        if response_payload is not None:
            return _model_response(response_payload, status_code=200)

        # Defensive: `assemble_response` (app/graph.py) always writes either
        # `response` or `error`. Reaching here would mean it didn't run at
        # all (e.g. a `recursion_limit` cutoff before the graph reached the
        # end) — still never a silent partial render.
        logger.error(
            "thread.message.graph_produced_neither_response_nor_error", thread_id=thread_id
        )
        return _model_response(
            ErrorResponse(
                error_code="SCHEMA_VALIDATION_FAILED",
                message="The analysis could not be completed due to an internal error.",
                retryable=True,
                trace_id=trace_id,
            ),
            status_code=500,
        )

    return app


app = create_app()
