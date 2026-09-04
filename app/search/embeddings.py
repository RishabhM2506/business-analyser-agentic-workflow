"""`EmbeddingsClient` Protocol + adapters: `GeminiEmbeddingsClient` (real) and
`MockEmbeddingsClient` (deterministic, zero-cost). Mirrors `app.models`'s
`ModelClient`/`get_model_for_role` factory shape, but is a distinct Protocol
— embedding a string isn't a chat/structured-output call, so it doesn't fit
`ModelClient.generate_structured`'s shape. Used both at request time (one
query embedding per search, via `app.search.vector_index`) and offline by
`scripts/embed_taxonomy.py` (5,613 corpus document embeddings, once).

Verified against the installed `langchain-google-genai==4.3.4` directly:
`GoogleGenerativeAIEmbeddings.aembed_query`/`.aembed_documents` are real,
async, and a `task_type=None` argument resolves internally to
`"RETRIEVAL_QUERY"`/no fixed default respectively — passed explicitly here
rather than relied on, matching `app/models.py`'s own "be explicit" habit.
Real live call against the user's own key confirmed `gemini-embedding-2-preview`
returns 3072-dim float vectors. `Settings.model_embedding` switched to the GA
`gemini-embedding-2` 2026-09-04, live-verified to produce identical vectors
(cosine similarity `1.000000` across 5 real HS6 taxonomy texts, vs. a `0.71`
baseline between two different texts under the same model) — same
underlying model under two names, not a corpus-invalidating change; see
that setting's own docstring for the full comparison.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol

import numpy as np

_RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
_RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
_EMBED_BATCH_SIZE = 100
# Live-reproduced 2026-08-21 (`scripts/embed_taxonomy.py`, a real overnight
# run): with no timeout configured, a single embed call can hang on an
# already-accepted TCP connection indefinitely — confirmed via `lsof` on the
# stuck process showing one ESTABLISHED connection to a Google IP that never
# completed or errored for 30+ minutes, no retry ever triggered because
# `tenacity`'s retry loop never got control back from the `await`. Unlike
# `app/models.py`'s `GeminiModelClient`, this adapter previously set no
# timeout at all. `client_args={"timeout": ...}` is the verified-correct
# knob (not `request_options`, which this docstring-documents but the
# installed `langchain-google-genai==4.3.4` never actually wires up —
# confirmed by reading `GoogleGenerativeAIEmbeddings._build_config`'s and
# `_initialize_client`'s source directly): it flows straight into
# `google.genai`'s own `HttpOptions.async_client_args`, which the SDK spreads
# as literal kwargs into `httpx.AsyncClient(**async_client_args)` — verified
# by reading `google.genai._api_client` directly, not assumed. A generous
# ceiling relative to `app/models.py`'s 20s chat-call timeout: a real batch
# of 100 texts is a heavier single call than one chat completion.
_EMBED_TIMEOUT_SECONDS = 60.0

# Real, live-verified dimensionality of `gemini-embedding-2-preview`
# (2026-08-20, real call against the user's own key) — used as
# `MockEmbeddingsClient`'s default dimensionality so mock vectors are
# shape-compatible with the real precomputed corpus by default; tests with
# a smaller synthetic corpus pass a smaller value explicitly.
GEMINI_EMBEDDING_DIMENSIONS = 3072


class EmbeddingsClient(Protocol):
    """Minimal interface every embeddings-provider adapter (real or mock)
    must satisfy. Two methods, not one, because Gemini's own API
    distinguishes `RETRIEVAL_QUERY` (asymmetric, optimized for a short
    search string) from `RETRIEVAL_DOCUMENT` (optimized for the longer text
    being searched over) — collapsing them into a single method would lose
    that distinction for adapters where it matters."""

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class GeminiEmbeddingsClient:
    """Real `langchain-google-genai`-backed adapter."""

    def __init__(self, *, model: str, api_key: str) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from pydantic import SecretStr

        # `google_api_key=` (the field's real name, wrapped in the field's
        # real declared type), not the `api_key=` alias with a plain `str`:
        # unlike `ChatGoogleGenerativeAI` (`app/models.py`), mypy's pydantic
        # plugin does not resolve this class's alias/coercion the same way
        # (verified directly — `api_key=api_key` as a bare `str` type-checks
        # clean on `ChatGoogleGenerativeAI` but not here), so both the alias
        # form and an un-wrapped `str` are real errors under this project's
        # `mypy --strict` gate even though every spelling works identically
        # at runtime (pydantic coerces a plain `str` into `SecretStr` either
        # way).
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=SecretStr(api_key),
            client_args={"timeout": _EMBED_TIMEOUT_SECONDS},
        )

    async def embed_query(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text, task_type=_RETRIEVAL_QUERY)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(
            texts, task_type=_RETRIEVAL_DOCUMENT, batch_size=_EMBED_BATCH_SIZE
        )


class MockEmbeddingsClient:
    """Deterministic, zero-token-spend embeddings used whenever
    `LLM_PROVIDER=mock` (shares that single switch with `app.models.MockLLM`
    — there is no separate embeddings-only provider setting; see
    `app/settings.py`'s `model_embedding` docstring). Each text's vector is
    seeded from `hashlib.sha256(text)` so it is stable across processes and
    runs (unlike `numpy`'s unseeded global RNG), then L2-normalized to match
    the real corpus's own normalization convention
    (`scripts/embed_taxonomy.py`). Explicitly *not* semantically meaningful
    — two similar-meaning strings get unrelated mock vectors — same
    "deterministic, not realistic" stance as `MockLLM`'s own placeholder
    text."""

    def __init__(self, *, dimensions: int = GEMINI_EMBEDDING_DIMENSIONS) -> None:
        self._dimensions = dimensions

    def _vector_for(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self._dimensions)
        norm = np.linalg.norm(vector)
        normalized = vector / norm if norm > 0 else vector
        return [float(x) for x in normalized]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(text) for text in texts]


def get_embeddings_client(*, provider: Literal["gemini", "mock"]) -> EmbeddingsClient:
    """Return the configured embeddings client. Mirrors
    `app.models.get_model_for_role`'s exact factory pattern — `Settings` is
    only read in the real-provider branch, same reasoning as that function
    (mock must work with zero configured API keys, e.g. in CI)."""
    if provider == "mock":
        return MockEmbeddingsClient()

    from app.settings import get_settings

    settings = get_settings()
    return GeminiEmbeddingsClient(model=settings.model_embedding, api_key=settings.gemini_api_key)
