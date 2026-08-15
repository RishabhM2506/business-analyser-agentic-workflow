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

## Status: Phase 2 (scaffolding)

This repo currently has **zero business logic**. What's real:

- `app/schemas/*.py` — the full set of data contracts (`TradeQuery`, `TradeAnalysisResponse`,
  `TradeTable`, `CountryRow`, `Provenance`, `ErrorResponse`), matching `docs/PLAN.md` §3
  exactly.
- `app/settings.py` — typed config (Pydantic Settings), fails loudly at startup if a
  required field is missing.
- `app/main.py` — a real FastAPI app with two working routes: `GET /` and `GET /healthz`.
- `data/harmonized-system.csv` — the real, checked-in HS6 taxonomy (see "HS taxonomy data"
  below).
- Everything else (`app/graph.py`, `app/nodes/*.py`, `app/tools/comtrade_client.py`,
  `app/knowledge/provider.py`, `app/cache/*.py`, `app/guardrails.py`, `app/budget.py`,
  `app/observability.py`, `app/models.py`) exists with correct imports, correct type
  signatures (mypy-strict-clean), and a body that's either `raise NotImplementedError` or a
  minimal placeholder — enough for the rest of the codebase to import and type-check against,
  nothing pretending to be real behavior. Each is marked `# TODO(Phase 3): ...`.

### Why `/threads` and `/threads/{id}/messages` aren't registered yet

`docs/PLAN.md` §3.3 defines a thread/message API. Those routes invoke the LangGraph workflow
in `app/graph.py`, which doesn't exist yet — `build_graph()` is a `NotImplementedError` stub.
Rather than register routes that would always return a placeholder 501, we chose **not to
register them at all** until Phase 3 gives them something real (even provisionally real) to
invoke. `GET /` and `GET /healthz` are registered and fully working now.

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
still need to be present as strings; see above). `LLM_PROVIDER=gemini` is the switch for real
calls, once Phase 3 implements the Gemini adapter.

## Testing

```bash
uv run pytest                      # everything: unit + integration + llm
uv run pytest -m unit              # pure-function / no-I/O tests only
uv run pytest -m integration       # FastAPI app tests (httpx ASGI transport, no live network)
uv run pytest -m llm               # proves the `llm` marker/CI-job wiring; trivial until Phase 3
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
`StaticKnowledgeProvider` will read it directly once implemented in Phase 3.

## Coverage

The `test` CI job reports coverage (`--cov=app --cov-report=term-missing`) but does not gate
on a threshold yet. A meaningful floor is hard to set honestly while most of `app/` is
intentionally unreachable `NotImplementedError` stubs (see "Status" above) — several modules
are never imported by any test at all. Revisit once Phase 3 fills those bodies in.

## Data retention

Per `docs/PLAN.md` §6: checkpointed conversation state (once the checkpointer exists, Phase
3) is retained for a rolling 90 days (`CHECKPOINT_RETENTION_DAYS`), enforced at that point —
no accounts, no PII, only HS-code selections and generated prose.

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
(`app`, uid 1000) on `python:3.12-slim`. No secrets are baked into any layer.
