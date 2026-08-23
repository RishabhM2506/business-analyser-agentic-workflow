"""Unit tests for `app.pipeline.dgcis.get_dgcis_countries` and
`fetch_all_countries_annual` — the real, checked-in country reference and
the per-country loop, exercised with `httpx.MockTransport` fakes, never a
live call."""

from __future__ import annotations

import httpx
import pytest

from app.pipeline.dgcis import (
    ANNUAL_IMPORT_PATH,
    DgcisAnnualRecord,
    DgcisClient,
    DgcisCountry,
    DgcisFetchFailure,
    fetch_all_countries_annual,
    get_dgcis_countries,
)

pytestmark = pytest.mark.unit

_TOKEN_PAGE = '<html><body><input type="hidden" name="_token" value="tok"></body></html>'


def _table_page(country: str) -> str:
    return f"""
    <html><body>
    <table id="example">
      <tr><td>Country: <span>{country}</span> HSCODE: <b>12079100</b></td></tr>
      <tr><th>S.No.</th><th>Year</th><th>2024 - 2025</th></tr>
      <tr><td>1</td><td>Values in ₹ Crore</td><td>1.00</td></tr>
    </table>
    </body></html>
    """


def test_get_dgcis_countries_loads_the_real_checked_in_reference() -> None:
    countries = get_dgcis_countries()

    assert len(countries) == 251
    assert DgcisCountry(code="409", name="TURKEY") in countries
    assert DgcisCountry(code="1", name="AFGHANISTAN") in countries


async def test_fetch_all_countries_annual_yields_one_record_per_country_with_no_gaps() -> None:
    countries = [DgcisCountry(code="1", name="ALPHA"), DgcisCountry(code="2", name="BETA")]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_TOKEN_PAGE)
        body = request.content.decode()
        country = "ALPHA" if "ContEidbi=1" in body else "BETA"
        return httpx.Response(200, text=_table_page(country))

    client = DgcisClient(transport=httpx.MockTransport(handler))
    results = [
        r
        async for r in fetch_all_countries_annual(
            client,
            path=ANNUAL_IMPORT_PATH,
            hs8="12079100",
            year="2024",
            countries=countries,
            delay_seconds=0,
        )
    ]

    assert len(results) == 2
    assert all(isinstance(r, DgcisAnnualRecord) for r in results)
    assert {r.country for r in results if isinstance(r, DgcisAnnualRecord)} == {"ALPHA", "BETA"}


async def test_fetch_all_countries_annual_yields_a_failure_and_continues() -> None:
    """docs/PLAN.md §7: 'job continues to next country... rather than
    aborting the batch' - one country's failure must not stop the loop."""
    countries = [DgcisCountry(code="1", name="ALPHA"), DgcisCountry(code="2", name="BETA")]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_TOKEN_PAGE)
        body = request.content.decode()
        if "ContEidbi=1" in body:
            return httpx.Response(500, text="server error")
        return httpx.Response(200, text=_table_page("BETA"))

    client = DgcisClient(transport=httpx.MockTransport(handler))
    results = [
        r
        async for r in fetch_all_countries_annual(
            client,
            path=ANNUAL_IMPORT_PATH,
            hs8="12079100",
            year="2024",
            countries=countries,
            delay_seconds=0,
        )
    ]

    assert len(results) == 2
    failures = [r for r in results if isinstance(r, DgcisFetchFailure)]
    successes = [r for r in results if isinstance(r, DgcisAnnualRecord)]
    assert len(failures) == 1 and failures[0].country.name == "ALPHA"
    assert len(successes) == 1 and successes[0].country == "BETA"


async def test_fetch_all_countries_annual_skips_a_response_with_no_table_without_failing() -> None:
    countries = [DgcisCountry(code="1", name="ALPHA")]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_TOKEN_PAGE)
        return httpx.Response(200, text="<html><body>no data</body></html>")

    client = DgcisClient(transport=httpx.MockTransport(handler))
    results = [
        r
        async for r in fetch_all_countries_annual(
            client,
            path=ANNUAL_IMPORT_PATH,
            hs8="12079100",
            year="2024",
            countries=countries,
            delay_seconds=0,
        )
    ]

    assert results == []  # no record, no failure - just nothing to report
