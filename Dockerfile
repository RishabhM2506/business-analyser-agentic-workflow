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
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- runtime -------------------------------------------------------------------
FROM python:3.12-slim AS runtime

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

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
