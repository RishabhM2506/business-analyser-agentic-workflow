"""Integration tests for the India trade-report endpoint
(`POST /threads/{thread_id}/trade-report`, `app.report.facts`/
`app.report.narrative`). Full HTTP stack (FastAPI + ASGI transport),
`LLM_PROVIDER=mock` (zero token spend, matching
`tests/integration/test_search_endpoint.py`'s established pattern) —
seeds the *real* warehouse tables with test-only data via the same
`warehouse_engine` fixture every other integration test in this pipeline
uses (`app.warehouse.db.get_engine` is a process-wide singleton resolved
from the real `DATABASE_URL` env var, not swappable per-app, so this test
relies on that same real Postgres rather than an isolated sqlite DB).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, insert
from sqlalchemy.ext.asyncio import AsyncEngine

import app.main as main_module
from app.budget import BudgetTracker
from app.main import create_app
from app.settings import Settings
from app.warehouse.db import get_engine as get_warehouse_engine
from app.warehouse.schema import analytics_partner_rankings, normalized_trade_flows

pytestmark = pytest.mark.integration

_TEST_HS6 = "010121"  # a real, taxonomy-allowlisted HS6 code (live horses; pure-bred breeding)


def _isolated_settings(**overrides: object) -> Settings:
    return Settings.model_validate({"database_url": "sqlite+aiosqlite:///:memory:", **overrides})


@asynccontextmanager
async def _client_for(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture(autouse=True)
async def _fresh_warehouse_engine() -> AsyncIterator[None]:
    """`app.warehouse.db.get_engine` is `@lru_cache`d for production's one
    long-lived event loop — correct there, but pytest-asyncio (`asyncio_mode
    = "auto"`, function-scoped loops by default) gives every test its own
    fresh loop, so a cached engine from an earlier test's now-closed loop
    would leak into this one (confirmed live: exactly this caused a real
    `RuntimeError: Event loop is closed` when this file's tests ran
    together, though each passed in isolation). Clearing the cache before
    and after every test forces a fresh engine bound to *this* test's own
    loop, the same problem `test_post_trade_report_returns_budget_exceeded_
    when_calls_are_exhausted` solves for `get_budget_tracker`'s identical
    singleton shape."""
    get_warehouse_engine.cache_clear()
    yield
    get_warehouse_engine.cache_clear()


@pytest.fixture(autouse=True)
async def _cleanup(warehouse_engine: AsyncEngine) -> None:
    yield
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            delete(normalized_trade_flows).where(normalized_trade_flows.c.hs6 == _TEST_HS6)
        )
        await conn.execute(
            delete(analytics_partner_rankings).where(analytics_partner_rankings.c.hs6 == _TEST_HS6)
        )


async def test_post_trade_report_returns_a_real_facts_document_and_narrative(
    warehouse_engine: AsyncEngine,
) -> None:
    year = date.today().year - 1
    async with warehouse_engine.begin() as conn:
        await conn.execute(
            insert(normalized_trade_flows).values(
                source="dgcis",
                hs6=_TEST_HS6,
                hs8=_TEST_HS6 + "00",
                hs_revision="ITC-HS",
                flow="import",
                period_month=date(year, 1, 1),
                calendar="FY",
                partner_country_code="792",
                basis="CIF",
                currency="INR",
                universe="india-customs",
                dataset_version="dgcis-annual-v1",
                is_provisional=False,
                status="OK",
                status_detail=None,
                value_inr_paise=1_000_000_000,
                value_original_currency_paise=1_000_000_000,
                fx_rate_used=None,
                fx_rate_date=None,
                quantity_kg=100,
            )
        )
        await conn.execute(
            insert(analytics_partner_rankings).values(
                hs6=_TEST_HS6,
                flow="import",
                year=year,
                partner_country_code="792",
                rank=1,
                value_inr_paise=1_000_000_000,
                status="OK",
            )
        )

    thread_id = str(uuid.uuid4())
    async with _client_for(_isolated_settings()) as client:
        response = await client.post(
            f"/threads/{thread_id}/trade-report",
            json={"hs_code": _TEST_HS6, "flow": "import", "years": 1},
        )

    assert response.status_code == 200
    body = response.json()
    assert "type" not in body  # bare response, not {type, data}-enveloped
    assert body["thread_id"] == thread_id
    assert body["facts"]["hs6"] == _TEST_HS6
    assert body["facts"]["flow"] == "import"
    assert body["facts"]["annual_series"][0]["total_inr_paise"] == 1_000_000_000
    assert body["facts"]["annual_series"][0]["partners"][0]["country"] == "Türkiye"
    assert body["narrative"]
    assert body["narrative_source"] in ("model", "model_retry", "template_fallback")


async def test_post_trade_report_rejects_an_unrecognized_hs_code() -> None:
    thread_id = str(uuid.uuid4())
    async with _client_for(_isolated_settings()) as client:
        response = await client.post(
            f"/threads/{thread_id}/trade-report",
            # 000000 is genuinely absent from the taxonomy (999999 is not -
            # it's a real reserved code, verified live before picking this
            # one).
            json={"hs_code": "000000", "flow": "import"},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "INVALID_HS_CODE"


async def test_post_trade_report_returns_budget_exceeded_when_calls_are_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # get_budget_tracker() is a process-wide singleton resolved from the
    # global Settings, not swappable via create_app(settings=...) - the
    # established pattern (tests/integration/test_search_endpoint.py) is
    # to monkeypatch the imported name directly in app.main's namespace.
    monkeypatch.setattr(
        main_module,
        "get_budget_tracker",
        lambda: BudgetTracker(max_calls_per_thread=0, max_calls_per_day=100),
    )
    thread_id = str(uuid.uuid4())
    async with _client_for(_isolated_settings()) as client:
        response = await client.post(
            f"/threads/{thread_id}/trade-report",
            json={"hs_code": _TEST_HS6, "flow": "import"},
        )

    assert response.status_code == 429
    body = response.json()
    assert body["error_code"] == "BUDGET_EXCEEDED"


async def test_post_trade_report_rejects_years_outside_the_d14_bounds() -> None:
    """This app's global `RequestValidationError` handler (`app/main.py`)
    turns every shape-invalid body - including a `TradeReportQuery` field
    bound violation - into a 400 `INVALID_QUERY`, not FastAPI's default
    422 (docs/PLAN.md §3.2: every response is our own schema, not
    FastAPI's `{"detail": [...]}`)."""
    thread_id = str(uuid.uuid4())
    async with _client_for(_isolated_settings()) as client:
        response = await client.post(
            f"/threads/{thread_id}/trade-report",
            json={"hs_code": _TEST_HS6, "flow": "import", "years": 9},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["data"]["error_code"] == "INVALID_QUERY"  # enveloped: the shared handler's shape
