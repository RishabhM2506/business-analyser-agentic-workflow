"""FastAPI application entrypoint.

Phase 2 scope: only `/` and `/healthz` are wired up and real. The
thread/message API described in docs/PLAN.md §3.3
(`POST /threads`, `GET /threads/{id}`, `POST /threads/{id}/messages`) is
Phase 3 work — it invokes the LangGraph workflow in `app/graph.py`, which
is not implemented yet (see the `# TODO(Phase 3)` markers there and in
`app/nodes/`). Rather than register placeholder routes that always 501, we
have chosen not to register them at all yet; there is nothing meaningful to
501 against until the request/response shapes for that path are wired to a
real (even if stubbed) graph invocation. This is documented in README.md.

`/healthz` genuinely verifies the configured database is reachable — it
does not return an unconditional 200 (master brief §2 backend specifics,
docs/PLAN.md §3.3).
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.settings import Settings, get_settings

REQUEST_ID_HEADER = "X-Request-ID"

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")


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
    settings = get_settings()
    configure_logging(json_logs=settings.log_json, log_level=settings.log_level)
    logger.info(
        "app.startup",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
    )
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

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "service": "business-analyser-agentic-workflow",
            "status": "ok",
            "phase": "2-scaffolding",
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

    return app


app = create_app()
