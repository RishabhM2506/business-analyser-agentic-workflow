# business-analyser-agentic-workflow

Agentic backend for the **Business Analyser** trade-data assistant: a user picks an HS
(Harmonized System) trade-code item and gets back a grounded 5-year import/export table for
that item's top trading partners, with an LLM-written description and summary — **never
LLM-written numbers**.

Stack: Python 3.12, FastAPI, Pydantic v2, LangGraph (workflow runtime, not an agent loop —
see the architecture doc), `uv` for dependency management.

**Architecture, data contracts, cost model, security model, testing strategy**: see
`docs/PLAN.md` in the sibling `BusinessAnalysingAgent/` directory (i.e.
`../BusinessAnalysingAgent/docs/PLAN.md` relative to this repo, or wherever you've cloned
this repo alongside the planning repo — it isn't duplicated here so there's exactly one
source of truth).

## Status: Phase 3 (implementation)

The full v1 request path is real, end to end: HS-code validation, cached Comtrade fetches,
deterministic aggregation, the two LLM nodes, the output guardrail, and the thread/message API,
all wired together as a compiled LangGraph `StateGraph` with a real checkpointer.

- `app/schemas/*.py` — the full set of data contracts (`TradeQuery`, `TradeAnalysisResponse`,
  `TradeTable`, `CountryRow`, `Provenance`, `ErrorResponse`), matching `docs/PLAN.md` §3
  exactly.
- `app/settings.py` — typed config (Pydantic Settings), fails loudly at startup if a
  required field is missing.
- `app/tools/comtrade_client.py`, `app/cache/*.py`, `app/knowledge/provider.py` — the Comtrade
  client (timeout, bounded retry+jitter, circuit breaker) and its two cache layers, and the
  static HS-taxonomy knowledge provider.
- `app/nodes/*.py` — every node in `docs/PLAN.md` §2.2's table: `validate_query`,
  `fetch_imports`/`fetch_exports`, `aggregate`, `retrieve_description`, `describe_item`,
  `summarize`.
- `app/graph.py` — `build_graph()` assembles the fixed pipeline; `build_checkpointer()` picks
  `AsyncSqliteSaver` or `AsyncPostgresSaver` from `DATABASE_URL`'s scheme.
- `app/main.py` — `GET /`, `GET /healthz`, `POST /threads`, `GET /threads/{id}`,
  `POST /threads/{id}/messages` (`docs/PLAN.md` §3.3) — all real.
- `app/guardrails.py`, `app/budget.py`, `app/observability.py` — the input/output guardrails,
  per-thread/per-day model-call ceilings, and LangSmith trace metadata wiring.
- `evals/dataset.jsonl` + `evals/run_evals.py` — a 20-row seed eval set spanning 20 of the 21
  official WCO HS sections, scored in CI (`eval-gate` job): number-grounding is blocking,
  a taxonomy-text sanity check is warn-only. See `evals/run_evals.py`'s module docstring.

### Thread/message API

```
POST /threads                    -> {"thread_id": "<uuid4>"}
GET  /threads/{id}               -> TradeAnalysisResponse or ErrorResponse (resume-after-refresh)
POST /threads/{id}/messages      -> body: TradeQuery-shaped selection
                                     -> TradeAnalysisResponse (200) or ErrorResponse (4xx/5xx)
```

Every response, success or error, is a schema-validated Pydantic model serialized straight to
JSON — never FastAPI's default validation-error shape (see `app/main.py`'s
`handle_validation_error`) and never a partial render.

## Running locally

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12 (uv will fetch 3.12 automatically
per `.python-version` if it's not already installed).

```bash
cp .env.example .env
# Edit .env: LLM_PROVIDER=mock needs no real API keys, but COMTRADE_API_KEY and
# GEMINI_API_KEY must still be *set to something* (Settings validates presence, not
# correctness, when nothing calls out to those APIs yet).

uv sync                       # installs runtime + dev dependency groups, creates .venv
uv run uvicorn app.main:app --reload
```

Then:

```bash
curl -s localhost:8000/          # service info
curl -s localhost:8000/healthz   # real DB connectivity check — 200 if reachable, 503 if not
```

`DATABASE_URL` defaults to a local SQLite file (`sqlite+aiosqlite:///./local.db`), created on
first connection — no external datastore needed to run `/healthz` successfully out of the box.

### Running with `LLM_PROVIDER=mock`

This is already the default in `.env.example`. Every test, and all of CI, runs against
`app/models.py`'s `MockLLM` — zero token spend, zero API keys required to be *valid* (they
still need to be present as strings; see above). `LLM_PROVIDER=gemini` switches to the real
`langchain-google-genai`-backed adapter (`GeminiModelClient`) for real calls — needs a real
`GEMINI_API_KEY`.

## Testing

```bash
uv run pytest                      # everything: unit + integration + llm
uv run pytest -m unit              # pure-function / no-I/O tests only
uv run pytest -m integration       # FastAPI app + full-graph tests (mocked Comtrade, no live network)
uv run pytest -m llm               # cassette-replay tests for describe_item/summarize (no live call)
uv run python evals/run_evals.py   # eval gate: number-grounding (blocking) + taxonomy sanity (warn)
```

Markers (`unit`, `integration`, `llm`) are registered in `pyproject.toml` under
`[tool.pytest.ini_options]` with `--strict-markers`, so a typo'd or unregistered marker fails
the run rather than silently doing nothing.

## Linting, formatting, type-checking

```bash
uv run ruff check .        # lint
uv run black --check .     # format check (uv run black . to auto-format)
uv run mypy app            # strict type-check
```

`pre-commit` runs the fast subset of these (ruff + black --check) plus Conventional Commits
enforcement on every commit — see `CONTRIBUTING.md` for setup.

## Structured logging

Uses [`structlog`](https://www.structlog.org/) rather than a hand-rolled
`logging.Formatter` subclass, specifically for its `contextvars` integration: `app/main.py`'s
`request_id_middleware` binds a per-request `request_id` (a UUID4, generated or propagated
from an inbound `X-Request-ID` header) into `structlog.contextvars` once per request, and
every log line emitted anywhere during that request — from any module, without passing a
logger instance down the call chain — automatically carries it. The same ID is echoed back on
the `X-Request-ID` response header. Logs render as JSON by default (`LOG_JSON=true`); set
`LOG_JSON=false` locally for a human-readable console renderer.

## HS taxonomy data

`data/harmonized-system.csv` is the real file, fetched from
`https://raw.githubusercontent.com/datasets/harmonized-system/main/data/harmonized-system.csv`
(ODC-PDDL public domain, per Gate 0 findings) — ~6,900 rows, checked in as-is. It's static and
versioned; nothing in this repo regenerates it. `app/knowledge/provider.py`'s
`StaticKnowledgeProvider` reads it directly (`app/guardrails.py`'s `hs_code` allowlist check
delegates to the same loader).

## Coverage

The `test` CI job reports coverage (`--cov=app --cov-report=term-missing`) but does not gate
on a threshold yet.

## Data retention

Per `docs/PLAN.md` §6: checkpointed conversation state is intended to be retained for a
rolling 90 days (`CHECKPOINT_RETENTION_DAYS`) — no accounts, no PII, only HS-code selections
and generated prose. The setting exists and is documented; an automated pruning job that
actually enforces it against the checkpointer's store is not yet implemented — tracked as
follow-up work, not silently assumed to already be running.

## Free-tier data policy

If you set `LLM_PROVIDER=gemini`: Google's free tier may use submitted content to improve its
products. Do not send confidential or personal data while on the free tier. This is a
documented production blocker to revisit before any real launch (`docs/PLAN.md` §6).

## Docker

```bash
docker build -t business-analyser-agentic-workflow .
docker run --rm -p 8000:8000 --env-file .env business-analyser-agentic-workflow
```

Multi-stage build (`Dockerfile`): dependencies resolve in a `builder` stage; the `runtime`
stage ships only the resulting virtualenv + application source, running as a non-root user
(`app`, uid 1000) on `python:3.12-slim`, plus the `libpq5` runtime library (needed only if
`DATABASE_URL` is switched to a `postgresql://` URL — see `app/graph.py`'s module docstring).
No secrets are baked into any layer.

## Checkpointer: SQLite vs Postgres

`DATABASE_URL`'s scheme picks the checkpointer (`app.graph.build_checkpointer`):
`sqlite+aiosqlite:///...` (the default) uses `AsyncSqliteSaver`; `postgresql+asyncpg://...`
uses `AsyncPostgresSaver`. Both are held open for the process's lifetime (opened once in
`app/main.py`'s `lifespan`, not per-request).
