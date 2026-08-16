# Contributing

This repository is public on GitHub (GitHub blocks branch protection on private repos without a
paid plan, and this project uses branch protection — a deliberate, accepted trade-off, not an
oversight). It is not open-source: see `LICENSE` — all rights reserved, no license is granted to
external contributors. No secrets are ever committed regardless of visibility (enforced by
`.gitignore`/`.dockerignore` and secret-scanning CI on every push). This document covers the
mechanics of contributing, for whoever has access.

## Commit convention

[Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <description>`.

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`,
`chore`, `revert`. Enforced on every commit by the `conventional-pre-commit` hook (see below)
and expected to match on PR titles.

Examples:

```
feat(schemas): add TradeQuery and response envelope models
fix(healthz): use pool_pre_ping to avoid stale connection false-negatives
ci: pin actions to commit SHAs
```

## Pre-commit hooks

Pre-commit runs the **fast subset** of what CI runs — ruff (lint) and `black --check`
(format), plus Conventional Commits enforcement on the commit message — so obvious problems
are caught before a push, not after a CI round-trip. Type-checking, the full test suite,
dependency audit, secret scanning, and the Docker build stay CI-only; they're too slow for a
pre-commit hook and don't need to gate every local commit.

Setup (one-time, after `uv sync --group dev`):

```bash
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type commit-msg
```

Run manually against the whole tree:

```bash
uv run pre-commit run --all-files
```

## Branch / PR workflow

1. Branch off `main`.
2. Commit using the convention above.
3. Open a PR into `main`. All CI checks (`.github/workflows/ci.yml`) must pass:
   `lint`, `format-check`, `typecheck`, `test` (unit + integration), `llm`, `dep-audit`,
   `secret-scan`, `docker-build`.
4. `CODEOWNERS` requires review from `@RishabhM2506` before merge.
5. Squash or rebase-merge — keep `main`'s history readable.

Branch protection on `main` (PR required, checks green, no force-push) is configured
separately in the repo settings, not by anything in this repo's code.

## Code style

- Ruff for linting, Black for formatting (100-char line length, both configured in
  `pyproject.toml`) — run `uv run ruff check . --fix` and `uv run black .` before committing
  if you didn't use the pre-commit hook.
- mypy strict mode. Every new function needs full type annotations; this is enforced in CI
  (`typecheck` job), not just a suggestion.
- New data contracts go in `app/schemas/`; never pass loosely-typed dicts between nodes —
  see `docs/PLAN.md` §3 and the master brief's "never positional args" rule.
