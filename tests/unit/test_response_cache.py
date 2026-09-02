"""Unit tests for `app.cache.response_cache` — `filter_hash` and
`ResponseCache` get/set/TTL semantics. Pure in-memory state, no I/O, same
classification as `tests/unit/test_budget.py` for the structurally
identical `BudgetTracker` (finding B8/AWR-01: this module had zero test
coverage of any kind before this fix).
"""

from __future__ import annotations

import pytest

from app.cache.response_cache import ResponseCache, filter_hash, get_response_cache
from app.schemas.query import TradeQuery
from app.schemas.response import Provenance, TradeAnalysisResponse, TradeBalance, TradeTable

_YEARS = (2019, 2023)


def _response(*, thread_id: str = "t-1", message_id: str = "m-1") -> TradeAnalysisResponse:
    table = TradeTable(
        unit="USD", years=[2019], years_finalized=[2019], excluded_partner_codes=[], rows=[]
    )
    return TradeAnalysisResponse(
        thread_id=thread_id,
        message_id=message_id,
        hs_code="010121",
        item_description="A description.",
        imports=table,
        exports=table,
        trade_balance=TradeBalance(by_year={}, cumulative=None),
        analytical_summary="A summary.",
        provenance=Provenance(
            source="UN Comtrade (comtradeapi.un.org)",
            retrieved_at="2026-01-01T00:00:00Z",
            period_type="calendar_year",
            currency="USD",
            prompt_version="v1",
            reporter_country="India",
        ),
    )


# --- filter_hash --------------------------------------------------------------


@pytest.mark.unit
def test_filter_hash_is_deterministic_for_identical_queries() -> None:
    a = TradeQuery(hs_code="010121", flow="import", partner_region="APAC")
    b = TradeQuery(hs_code="010121", flow="import", partner_region="APAC")
    assert filter_hash(a) == filter_hash(b)


@pytest.mark.unit
def test_filter_hash_ignores_identity_and_already_separate_key_fields() -> None:
    # tenant_id/user_id (identity) and hs_code/year_start/year_end (already
    # separate cache-key components) must not affect the hash at all.
    base = TradeQuery(hs_code="010121", year_start=2019, year_end=2023, tenant_id="default")
    varied = TradeQuery(
        hs_code="160100", year_start=1990, year_end=1995, tenant_id="acme", user_id="u-1"
    )
    assert filter_hash(base) == filter_hash(varied)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flow", "import"),
        ("partner_region", "APAC"),
        ("value_or_volume", "volume"),
    ],
)
def test_filter_hash_changes_when_a_relevant_field_changes(field: str, value: object) -> None:
    base = TradeQuery(hs_code="010121")
    varied = TradeQuery(hs_code="010121", **{field: value})  # type: ignore[arg-type]
    assert filter_hash(base) != filter_hash(varied)


@pytest.mark.unit
def test_filter_hash_distinguishes_partner_region_values_containing_a_delimiter_char() -> None:
    # A naive "|".join(...) implementation could let a delimiter character
    # inside a free-form field (partner_region is the only one - flow/
    # value_or_volume are closed Literals and can't contain one) produce
    # ambiguous input to the hash - json.dumps(..., sort_keys=True) encodes
    # the field boundary unambiguously regardless of the value's own content.
    a = TradeQuery(hs_code="010121", partner_region="APAC|EU")
    b = TradeQuery(hs_code="010121", partner_region="APAC")
    assert filter_hash(a) != filter_hash(b)


# --- ResponseCache --------------------------------------------------------------


@pytest.mark.unit
async def test_get_returns_none_for_a_never_set_key() -> None:
    cache = ResponseCache()
    result = await cache.get(
        hs_code="010121", year_range=_YEARS, filter_hash="h1", prompt_version="v1"
    )
    assert result is None


@pytest.mark.unit
async def test_set_then_get_returns_the_same_response() -> None:
    cache = ResponseCache()
    response = _response()
    await cache.set(
        hs_code="010121",
        year_range=_YEARS,
        filter_hash="h1",
        prompt_version="v1",
        response=response,
    )
    result = await cache.get(
        hs_code="010121", year_range=_YEARS, filter_hash="h1", prompt_version="v1"
    )
    assert result == response


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"hs_code": "160100"},
        {"year_range": (2020, 2024)},
        {"filter_hash": "different"},
        {"prompt_version": "v2"},
    ],
)
async def test_get_misses_when_any_key_component_differs(overrides: dict[str, object]) -> None:
    cache = ResponseCache()
    await cache.set(
        hs_code="010121",
        year_range=_YEARS,
        filter_hash="h1",
        prompt_version="v1",
        response=_response(),
    )
    lookup = {
        "hs_code": "010121",
        "year_range": _YEARS,
        "filter_hash": "h1",
        "prompt_version": "v1",
        **overrides,
    }
    assert await cache.get(**lookup) is None  # type: ignore[arg-type]


@pytest.mark.unit
async def test_entry_expires_after_ttl() -> None:
    clock = iter([0.0, 1000.0])  # set at t=0, get at t=1000, ttl=10
    cache = ResponseCache(ttl_seconds=10, now_fn=lambda: next(clock))
    await cache.set(
        hs_code="010121",
        year_range=_YEARS,
        filter_hash="h1",
        prompt_version="v1",
        response=_response(),
    )
    result = await cache.get(
        hs_code="010121", year_range=_YEARS, filter_hash="h1", prompt_version="v1"
    )
    assert result is None


@pytest.mark.unit
def test_get_response_cache_returns_a_process_wide_singleton() -> None:
    assert get_response_cache() is get_response_cache()
