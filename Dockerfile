# syntax=docker/dockerfile:1

# Multi-stage build: dependencies are resolved/compiled in `builder`, and
# only the resulting virtualenv + application source are copied into the
# final `runtime` image. No build toolchain, no uv, no dev-dependency
# group, and no secrets ever land in the image that ships (docs/PLAN.md
# §4.1, master brief §8: "never baked into images").

# --- builder -----------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first (layer-cacheable independent of source changes).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now add the application source and finish the sync.
COPY app/ ./app/
COPY prompts/ ./prompts/
COPY data/ ./data/
# migrations/ + alembic.ini: the trade pipeline's schema (docs/PLAN.md §4)
# needs `alembic upgrade head` runnable inside the deployed container, not
# just from a developer's local checkout.
COPY migrations/ ./migrations/
COPY alembic.ini ./alembic.ini
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- runtime -------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libpq5: the runtime shared library `psycopg` (pulled in transitively by
# `langgraph-checkpoint-postgres`, app/graph.py's PostgresSaver path) needs
# to actually open a Postgres connection. Verified directly against this
# same base image (`python:3.12-slim`): a bare `pip install psycopg &&
# python -c "import psycopg"` fails with "libpq library not found" without
# it. `libpq5` (not `libpq-dev`) is the small, runtime-only package — no
# compiler, no headers, consistent with "slim base" (docs/PLAN.md §4.1).
# app/graph.py imports `psycopg` lazily specifically so a sqlite-only
# deployment (the default) never needs this at all; this line is what makes
# the documented Postgres-in-production path (docs/PLAN.md §2.2) actually
# work when DATABASE_URL is switched to a postgresql:// URL.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --no-create-home app

WORKDIR /app

# WORKDIR creates /app owned by root; the app user needs to write here too
# (e.g. the default sqlite+aiosqlite DATABASE_URL creates its file directly
# under /app at runtime) — chown it before switching users below.
RUN chown app:app /app

COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/app ./app
COPY --from=builder --chown=app:app /app/prompts ./prompts
COPY --from=builder --chown=app:app /app/data ./data
COPY --from=builder --chown=app:app /app/migrations ./migrations
COPY --from=builder --chown=app:app /app/alembic.ini ./alembic.ini

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
