"""Integration tests for the free-text product search endpoint
(`POST /threads/{thread_id}/search`, docs/PLAN.md's 2026-08-20 roadmap
decision). Full HTTP stack (FastAPI + ASGI transport), `LLM_PROVIDER=mock`
(zero token spend, master brief §6) — but the *real*, checked-in taxonomy
CSV and the *real*, offline-generated corpus embeddings
(`data/hs_taxonomy_embeddings.*`, `scripts/embed_taxonomy.py`) are used
unmocked, matching this repo's existing testing philosophy
(`tests/integration/test_threads_api.py` does the same for the real
taxonomy CSV against the graph). Only the network-calling model/query-
embedding calls are mocked.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypeVar

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

import app.main as main_module
from app.budget import BudgetTracker
from app.main import REQUEST_ID_HEADER, create_app
from app.settings import Settings

T = TypeVar("T", bound=BaseModel)


def _isolated_settings(**overrides: object) -> Settings:
    return Settings.model_validate({"database_url": "sqlite+aiosqlite:///:memory:", **overrides})


@asynccontextmanager
async def _client_for(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):  # actually runs app/main.py's lifespan
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


class _FixedScoreModelClient:
    """Test double that ranks the real candidates it was handed
    (extracted from `user_content`, same 6-digit-extraction trick
    `app.models.MockLLM` itself uses) at a caller-chosen fixed score --
    lets tests deterministically force a high- or low-confidence rerank
    result without depending on MockLLM's own fixed 0.5 constant."""

    def __init__(self, *, score: float) -> None:
        self._score = score

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        codes = list(dict.fromkeys(re.findall(r"\b\d{6}\b", user_content)))
        ranked = [{"hs_code": code, "relevance_score": self._score} for code in codes]
        return schema.model_validate({"ranked_candidates": ranked})


class _InventedCodeModelClient:
    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        payload = {"ranked_candidates": [{"hs_code": "999999", "relevance_score": 0.9}]}
        return schema.model_validate(payload)


@pytest.mark.integration
async def test_post_search_coffee_query_returns_disambiguate_under_mock_llm() -> None:
    """Under `LLM_PROVIDER=mock`, `MockLLM` assigns every candidate the
    same fixed 0.5 `relevance_score` (app/models.py's `_MOCK_FLOAT_VALUE`)
    -- so the mock path is deterministically `disambiguate`."""
    thread_id = str(uuid.uuid4())
    async with _client_for(_isolated_settings()) as client:
        response = await client.post(f"/threads/{thread_id}/search", json={"query_text": "coffee"})

    assert response.status_code == 200
    body = response.json()
    # Bare response, not {"type": "final", "data": ...}-enveloped.
    assert "type" not in body
    assert body["thread_id"] == thread_id
    assert body["query_text"] == "coffee"
    assert body["outcome"] == "disambiguate"
    assert "selected_hs_code" not in body  # removed field - never auto-selects, see below
    assert len(body["candidates"]) > 0
    hs_codes = [c["hs_code"] for c in body["candidates"]]
    assert any(code.startswith("0901") for code in hs_codes)  # coffee family present somewhere
    assert REQUEST_ID_HEADER in response.headers


@pytest.mark.integration
async def test_post_search_high_confidence_result_still_returns_disambiguate_not_auto_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-09-02 product decision: a search never auto-navigates on the
    user's behalf, however confident - `outcome` has exactly two values
    now (`disambiguate` / `no_candidates_found`), `auto_selected` was
    removed entirely. This is the regression test for that: a rerank
    result that would previously have cleared the old 0.75 auto-select
    threshold must still come back as `disambiguate`, with the
    high-scoring candidate simply first in the list, not silently acted
    on."""
    monkeypatch.setattr(
        main_module, "get_model_for_role", lambda role, provider: _FixedScoreModelClient(score=0.95)
    )
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(f"/threads/{thread_id}/search", json={"query_text": "coffee"})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "disambiguate"
    assert "selected_hs_code" not in body
    assert len(body["candidates"]) > 0
    assert all(c["relevance_score"] == 0.95 for c in body["candidates"])


@pytest.mark.integration
async def test_post_search_caps_candidates_at_five_even_when_more_qualify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`app.search.service.MAX_DISAMBIGUATE_CANDIDATES` (5): a query whose
    real retrieval + fusion step surfaces more than 5 candidates, all
    scored well above the confidence floor, must still come back capped
    at exactly 5 - never all of them. "coffee" against the real taxonomy
    genuinely fuses more than 5 real HS6 candidates (the whole coffee/tea/
    mate family plus fusion noise), so this exercises the real pipeline,
    not a synthetic fixture."""
    monkeypatch.setattr(
        main_module, "get_model_for_role", lambda role, provider: _FixedScoreModelClient(score=0.9)
    )
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(f"/threads/{thread_id}/search", json={"query_text": "coffee"})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "disambiguate"
    assert len(body["candidates"]) == 5


@pytest.mark.integration
async def test_post_search_nonsense_query_returns_no_candidates_found_after_one_normalize_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-08-21 update: a nonsense query no longer skips the budget check
    entirely. `app.search.service.search_products` now gates a
    query-normalization retry on "BM25 found zero lexical overlap" (the
    verified fix for the "posta dana" cross-lingual bug -
    `tests/integration/test_search_service.py`'s regression test) - true
    for gibberish exactly as much as for real vernacular terms, since
    gibberish also has no lexical overlap with the taxonomy. Under
    `LLM_PROVIDER=mock`, normalization is a deterministic passthrough
    (`app.models._mock_text_for`'s `normalized_query` branch), so the
    retry search never actually re-runs (`normalized == query_text`) and
    the outcome is still `no_candidates_found` - just at the cost of
    exactly one budget-checked model call instead of zero. This is the
    one deliberate, flagged cost trade-off of that fix (see its plan's
    "must-verify" section): "nonsense is free" no longer holds for the
    BM25-empty subset specifically."""
    calls = {"count": 0}
    real_tracker = BudgetTracker(max_calls_per_thread=100, max_calls_per_day=100)
    real_check = real_tracker.check_and_increment

    async def _counting_check(*, thread_id: str, tenant_id: str) -> None:
        calls["count"] += 1
        await real_check(thread_id=thread_id, tenant_id=tenant_id)

    monkeypatch.setattr(real_tracker, "check_and_increment", _counting_check)
    monkeypatch.setattr(main_module, "get_budget_tracker", lambda: real_tracker)
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(
            f"/threads/{thread_id}/search", json={"query_text": "zzzqqqxxx nonsense gibberish"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "no_candidates_found"
    assert "selected_hs_code" not in body
    assert body["candidates"] == []
    assert calls["count"] == 1  # one budget-checked normalization call, never rerank


@pytest.mark.integration
async def test_post_search_uniformly_low_reranked_scores_also_return_no_candidates_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real live-testing finding (2026-08-20, real
    embeddings corpus + real Gemini): a genuinely nonsense query wasn't
    always caught by the pre-reranker floor (real embedding-space geometry
    can put an unrelated query's raw similarity above that floor), reached
    the reranker, and got back a uniformly-near-zero-relevance candidate
    list that `search_products` used to return as `disambiguate` — a list
    of options the model itself rated as not actually matching anything.
    Forces that exact shape here via a fixed-low-score model client on a
    query BM25 *does* match (`"coffee"`), so this exercises
    `app.search.rerank.LOW_CONFIDENCE_FLOOR`, not the pre-reranker floor
    (which this query doesn't even reach)."""
    monkeypatch.setattr(
        main_module, "get_model_for_role", lambda role, provider: _FixedScoreModelClient(score=0.1)
    )
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(f"/threads/{thread_id}/search", json={"query_text": "coffee"})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "no_candidates_found"
    assert "selected_hs_code" not in body
    assert body["candidates"] == []


@pytest.mark.integration
async def test_post_search_budget_exceeded_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main_module,
        "get_budget_tracker",
        lambda: BudgetTracker(max_calls_per_thread=0, max_calls_per_day=100),
    )
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(f"/threads/{thread_id}/search", json={"query_text": "coffee"})

    assert response.status_code == 429
    body = response.json()
    assert body["error_code"] == "BUDGET_EXCEEDED"
    assert "type" not in body  # still bare, even on this error path


@pytest.mark.integration
async def test_post_search_invented_code_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main_module, "get_model_for_role", lambda role, provider: _InventedCodeModelClient()
    )
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(f"/threads/{thread_id}/search", json={"query_text": "coffee"})

    assert response.status_code == 502
    body = response.json()
    assert body["error_code"] == "RERANK_INVALID_CANDIDATE"
    assert "type" not in body


@pytest.mark.integration
async def test_post_search_unexpected_failure_returns_schema_valid_500_not_a_raw_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: a real `docker build`/`docker run` smoke test
    (2026-08-20) hit this exact path live — before `post_search` had a
    catch-all, a missing `data/hs_taxonomy_embeddings.npy` (any deployment
    that hasn't yet run `scripts/embed_taxonomy.py`) raised a bare
    `FileNotFoundError` all the way out to FastAPI's own default handler,
    returning plain-text "Internal Server Error" instead of this project's
    `ErrorResponse` contract (docs/PLAN.md §3.2: *every* response is
    schema-validated, never a raw stack trace). Simulates that class of
    failure generically (any unexpected exception from `find_candidates`),
    not the specific `FileNotFoundError`, since the fix is a blanket
    catch-all, not a narrow one."""

    class _ExplodingSearchProvider:
        async def find_candidates(self, query_text: str, *, top_k: int = 8) -> list[object]:
            raise FileNotFoundError("simulated missing embeddings corpus")

    monkeypatch.setattr(
        main_module, "HybridSearchProvider", lambda **kwargs: _ExplodingSearchProvider()
    )
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(f"/threads/{thread_id}/search", json={"query_text": "coffee"})

    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "INTERNAL_ERROR"
    assert body["retryable"] is False
    assert "type" not in body  # still bare, even on this error path
    assert len(body["trace_id"]) > 0


@pytest.mark.integration
async def test_post_search_malformed_body_returns_existing_enveloped_400() -> None:
    """docs/PLAN.md's roadmap plan explicitly reuses the existing global
    `handle_validation_error` for a malformed body on this endpoint too
    (rather than a bespoke bare-shaped 400) -- this pins that deliberate
    choice: a shape-invalid `/search` body comes back enveloped, unlike
    every other response this endpoint sends, and the frontend's envelope
    handling already tolerates either shape."""
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(f"/threads/{thread_id}/search", json={"query_text": ""})

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "final"
    assert body["data"]["error_code"] == "INVALID_QUERY"


@pytest.mark.integration
async def test_post_search_binds_tenant_and_user_id_into_log_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors `test_threads_api.py`'s identical check for `post_message`
    (finding M6/ARCH-05) -- `structlog.contextvars.bind_contextvars` must
    run before any real work starts, so it's captured from inside
    `get_embeddings_client`, the first thing `post_search` calls after
    binding."""
    import structlog

    from app.search.embeddings import EmbeddingsClient, MockEmbeddingsClient

    captured: dict[str, object] = {}

    def _capturing_get_embeddings_client(*, provider: str) -> EmbeddingsClient:
        captured.update(structlog.contextvars.get_contextvars())
        return MockEmbeddingsClient()  # this test's isolated settings always use provider="mock"

    monkeypatch.setattr(main_module, "get_embeddings_client", _capturing_get_embeddings_client)
    thread_id = str(uuid.uuid4())

    async with _client_for(_isolated_settings()) as client:
        response = await client.post(
            f"/threads/{thread_id}/search",
            json={"query_text": "coffee", "tenant_id": "acme", "user_id": "u-42"},
        )

    assert response.status_code == 200
    assert captured["tenant_id"] == "acme"
    assert captured["user_id"] == "u-42"
