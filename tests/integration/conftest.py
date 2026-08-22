"""Fixtures for tests that need a real Postgres connection to the trade
pipeline's warehouse tables (`app.warehouse.schema`) — `docs/PLAN.md`
Testing standards: "Integration tests run against a real Postgres... not
mocks." Skips (not fails) when `DATABASE_URL` isn't a real Postgres URL,
so the rest of this repo's test suite (sqlite by default,
`tests/conftest.py`) is unaffected and a local run without
`docker compose up postgres` still passes everything except the tests
that specifically need this fixture.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.settings import get_settings


@pytest.fixture
async def warehouse_engine() -> AsyncIterator[AsyncEngine]:
    """A real `AsyncEngine` against the warehouse tables. Not wrapped in a
    single shared transaction (the code under test — e.g.
    `ManualDutySource` — opens its own connections per call, matching real
    production use): tests are responsible for using a unique key (e.g. a
    test-only `hs8` value) and cleaning up their own inserted rows."""
    database_url = get_settings().database_url
    if not database_url.startswith("postgresql"):
        pytest.skip(
            f"warehouse_engine requires a real Postgres DATABASE_URL (got {database_url!r}) — "
            f"run with DATABASE_URL=postgresql+asyncpg://... pointed at a real instance with "
            f"migrations applied (e.g. the docker-compose postgres service)."
        )

    engine = create_async_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()
