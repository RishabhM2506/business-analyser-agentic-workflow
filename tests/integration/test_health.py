"""Integration tests for the two routes that are real in Phase 2: `/` and
`/healthz`. `/healthz` must genuinely check database connectivity, not
return an unconditional 200 (master brief §2 backend specifics) — tested
here on both the healthy and unhealthy paths.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import REQUEST_ID_HEADER, create_app
from app.settings import Settings


@pytest.mark.integration
async def test_root_returns_service_info() -> None:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "business-analyser-agentic-workflow"
    assert body["status"] == "ok"


@pytest.mark.integration
async def test_healthz_reports_healthy_when_database_reachable() -> None:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"


@pytest.mark.integration
async def test_healthz_reports_unhealthy_when_database_unreachable() -> None:
    broken_settings = Settings(
        comtrade_api_key="k-comtrade",
        gemini_api_key="k-gemini",
        database_url="sqlite+aiosqlite:///./this/directory/does/not/exist/db.sqlite3",
    )
    app = create_app(settings=broken_settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["checks"]["database"] == "unreachable"


@pytest.mark.integration
async def test_request_id_is_generated_and_echoed() -> None:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert REQUEST_ID_HEADER in response.headers
    assert len(response.headers[REQUEST_ID_HEADER]) == 36  # UUID4 string length


@pytest.mark.integration
async def test_request_id_is_propagated_when_supplied() -> None:
    app = create_app()
    transport = ASGITransport(app=app)
    supplied_id = "11111111-1111-4111-8111-111111111111"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/", headers={REQUEST_ID_HEADER: supplied_id})
    assert response.headers[REQUEST_ID_HEADER] == supplied_id
