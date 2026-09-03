"""Node-role -> model instance mapping (master brief §3: "A model-selection
layer (`models.py`) that maps node role to model, so swapping providers is
one file"). Also owns the `MockLLM` provider switch
(`LLM_PROVIDER=mock`, master brief §6) — mandatory for CI/tests/frontend
dev so zero token spend is possible end to end.

Verified against the installed `langchain-google-genai==4.3.4` (pinned in
`pyproject.toml`) directly — `ChatGoogleGenerativeAI`'s constructor kwargs
(`model`, `api_key` — an alias for `google_api_key` — `temperature`,
`max_output_tokens`, `timeout`, `max_retries`) and
`.with_structured_output(schema, method="json_schema")` returning a
`Runnable` whose `.ainvoke()` yields a validated instance of `schema` were
all introspected from the real installed package, not assumed from
documentation alone (`docs.langchain.com`'s current integration page for
`google_generative_ai` was also checked and agrees).

**`generate_grounded` (2026-09-02, Step 4 hardening, `llm_datapoints`)
— real live-spike findings, not assumed:** binding the `google_search`
grounding tool (`.bind_tools([{"google_search": {}}])`) and then calling
`.with_structured_output(schema, method="json_schema")` in the *same*
call does not raise — the API accepts the combined request — but a real,
live successful call this way came back with `response_metadata`
containing no `grounding_metadata` key at all, meaning either the model
silently chose not to invoke the search tool, or the citation channel
does not survive being combined with a schema constraint on this SDK
version. Confirmed separately, with the raw `google-genai` SDK directly
(bypassing langchain): the exact same API key succeeds on a bare
ungrounded call and fails immediately with `429 RESOURCE_EXHAUSTED` the
moment the `google_search` tool is added — Search grounding has its own,
separate, far more easily exhausted quota from ordinary generation calls,
a real operational fact for `run_llm_datapoint_search.py` regardless of
call shape.

Given this, `generate_grounded` below uses a **two-call** design: call 1
is a plain grounded free-text call (no schema constraint, so grounding
metadata isn't competing with anything), call 2 is an ordinary
`generate_structured`-shaped extraction over that grounded text (no
search tool, so the schema constraint applies cleanly, exactly like every
other structured call in this file). **Not yet independently confirmed**:
that `grounding_metadata` actually populates on the plain call 1 once
real search does happen (today's quota exhaustion above prevented
completing that specific check) — `_extract_citations` below is written
defensively against Google's documented `groundingChunks[].web.{uri,
title}` shape and fails closed (raises, extracts nothing) rather than
guessing at an unconfirmed shape; this should be re-verified live once
quota resets, per this file's own "verified against the real thing, not
assumed" standard the rest of this docstring holds itself to.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, get_args, get_origin

import structlog
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from app.guardrails import extract_numbers
from app.nodes.aggregate import TOP_N_PARTNERS

logger: structlog.stdlib.BoundLogger = structlog.get_logger("app")

NodeRole = Literal["utility", "analysis"]

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class GroundingCitation:
    """One real, checkable source a `generate_grounded` call's answer was
    (claimed to be) drawn from — `source_url` is the one field every
    caller actually needs (a citation with no URL isn't checkable); `title`
    is best-effort, `None` when the grounding response didn't carry one."""

    source_url: str
    title: str | None = None


class UngroundedSearchError(Exception):
    """Raised by `generate_grounded` when a grounded search call comes back
    with zero real citations to extract — never silently falls through to
    treating the model's bare answer as if it were cited (2026-09-02, Step
    4 hardening: the same fail-closed discipline `check_hs_codes_grounded`/
    `check_numbers_grounded` already apply to a narrated number applies
    here to a *sourced* one — an uncited `llm_datapoints` entry is exactly
    the fabrication risk this whole feature exists to avoid)."""


@dataclass(frozen=True)
class GroundedResult[U: BaseModel]:
    """Result of `generate_grounded` — a validated `schema` instance plus
    the real citations the grounded search step actually returned. `citations`
    is deliberately required, not defaulted to `[]` — every real construction
    site explicitly has a non-empty list in hand already (`generate_grounded`
    raises `UngroundedSearchError` instead of ever constructing one with
    none), so a silently-available empty default would invite a future
    caller to construct a "grounded" result with nothing actually grounding
    it."""

    value: U
    citations: list[GroundingCitation]


# Bounded retries only (master brief §7.9) — the langchain-google-genai
# default (6) is too generous against a finite per-thread call budget
# (docs/PLAN.md §5.5: max_model_calls_per_thread, a per-session, not
# per-call, ceiling); a stuck call should fail fast into the
# budget/guardrail error path, not silently multiply spend retrying
# internally.
_GEMINI_MAX_RETRIES = 2
_GEMINI_TIMEOUT_SECONDS = 20.0


class ModelClient(Protocol):
    """Minimal interface every model-provider adapter (real or mock) must
    satisfy — deliberately narrow so swapping Gemini for another provider
    is a new adapter, not a new interface. Structured (not free-text)
    output is part of the interface itself: every v1 LLM node needs
    schema-constrained output (docs/PLAN.md §2.2), so there is no
    "generate free text and regex-parse it later" path to accidentally
    reach for."""

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T: ...

    async def generate_grounded(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> GroundedResult[T]:
        """Sibling to `generate_structured`, not a replacement — every
        existing caller is unaffected. For a datapoint that must carry a
        real, checkable source (2026-09-02, Step 4 hardening,
        `llm_datapoints`) rather than the model's own unsourced answer.
        Raises `UngroundedSearchError` when no real citation comes back —
        never returns a `GroundedResult` with empty `citations`."""
        ...


class GeminiModelClient:
    """Real `langchain-google-genai`-backed adapter."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        max_retries: int = _GEMINI_MAX_RETRIES,
        timeout: float = _GEMINI_TIMEOUT_SECONDS,
    ) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._chat = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=0.2,  # low but nonzero: consistent prose, not maximally deterministic
            max_retries=max_retries,
            timeout=timeout,
        )

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        from langchain_core.messages import HumanMessage, SystemMessage

        structured = self._chat.with_structured_output(schema, method="json_schema")
        result = await structured.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        )
        if not isinstance(result, schema):
            # with_structured_output(schema) (without include_raw=True)
            # contractually returns a `schema` instance; this is defensive,
            # not an expected runtime path.
            raise TypeError(f"expected {schema.__name__}, got {type(result).__name__}")
        return result

    async def generate_grounded(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> GroundedResult[T]:
        """Two-call design — see this module's own docstring for the real,
        live spike finding that motivates it (combining the search tool
        with a schema-constrained call silently drops grounding metadata,
        not an explicit incompatibility)."""
        from langchain_core.messages import HumanMessage, SystemMessage

        grounded_chat = self._chat.bind_tools([{"google_search": {}}])
        grounded_response = await grounded_chat.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        )
        citations = _extract_citations(grounded_response.response_metadata)
        if not citations:
            raise UngroundedSearchError(
                "generate_grounded: the search call returned no real citations "
                "to extract from — refusing to treat an uncited answer as sourced"
            )

        extraction_prompt = (
            "Extract the answer to the original question from the real, "
            "already-researched text below into the requested schema. Do not "
            "add any information that isn't present in the text.\n\n"
            f"Original question:\n{user_content}\n\n"
            f"Researched text:\n{grounded_response.content}"
        )
        value = await self.generate_structured(
            system_prompt=system_prompt, user_content=extraction_prompt, schema=schema
        )
        return GroundedResult(value=value, citations=citations)


def _extract_citations(response_metadata: dict[str, Any]) -> list[GroundingCitation]:
    """Pulls real source URLs out of a grounded response's
    `grounding_metadata` — written defensively against Google's documented
    `groundingChunks[].web.{uri,title}` shape (`google-genai==2.18.1`'s own
    `GroundingChunk`/`GroundingChunkWeb` types), **not yet independently
    confirmed against a real populated response** (this module's own
    docstring explains why — real Search-grounding quota was exhausted
    before that specific check could complete). Returns `[]` (never raises)
    for any shape that doesn't match what's expected — `generate_grounded`
    treats an empty result as "nothing to safely extract," the same
    fail-closed outcome as a citation genuinely not existing, rather than
    this function guessing at an unconfirmed shape."""
    grounding_metadata = response_metadata.get("grounding_metadata")
    if not grounding_metadata:
        return []
    chunks = grounding_metadata.get("grounding_chunks") or grounding_metadata.get("groundingChunks")
    if not chunks:
        return []
    citations: list[GroundingCitation] = []
    for chunk in chunks:
        web = chunk.get("web") if isinstance(chunk, dict) else None
        if not web:
            continue
        uri = web.get("uri")
        if not uri:
            continue
        citations.append(GroundingCitation(source_url=uri, title=web.get("title")))
    return citations


_WORD_WITH_DIGIT_PATTERN = re.compile(r"\d")


def _strip_numeric_words(text: str) -> str:
    """`text` with every whitespace-delimited token containing a digit
    removed entirely (not just reformatted) — used to build mock output
    that is provably number-free by construction, not just "unlikely to
    contain one" (finding M2/AWR-04 — `describe_item`'s output guardrail
    now rejects *any* number at all)."""
    return " ".join(word for word in text.split() if not _WORD_WITH_DIGIT_PATTERN.search(word))


def _mock_text_for(user_content: str, *, field_name: str) -> str:
    """Deterministic, schema-agnostic placeholder text for `MockLLM`.

    `field_name` determines whether numeric echoing is safe to do at all:

    - `describe_item`'s `description` field is now guardrail-checked for
      *any* number, full stop (finding M2/AWR-04, `app/nodes/describe_item.py`)
      — its `user_content` always contains the hs_code's own digits (`f"HS
      code: {hs_code}..."`), so this branch must guarantee zero digits reach
      the output, not just "usually" avoid them. `_strip_numeric_words`
      removes every digit-containing token outright rather than trying to
      reformat around individual numbers.
    - `app.search.normalize.NormalizedQuery`'s `normalized_query` field is a
      deterministic passthrough of `user_content` (the raw query text)
      unchanged — under `LLM_PROVIDER=mock`, normalization is always a
      no-op, so `app.search.service.search_products`'s BM25-empty retry
      path never fires in any mock-based test/CI run; it's exercised only
      by tests using a fake `ModelClient` that returns a genuinely
      different string (mirroring `test_describe_item.py`'s
      `_NumberInventingModelClient` pattern).
    - Every other field (currently just `summarize`'s `analytical_summary`)
      keeps the original intent: if `user_content` contains any number (for
      `summarize`, the compact rendered trade table — `app/nodes/summarize.py`),
      a few are echoed verbatim via `app.guardrails.extract_numbers` — the
      exact same extraction the output guardrail uses — so the mock output
      is grounded *by construction* and a full graph run under
      `LLM_PROVIDER=mock` exercises that guardrail meaningfully instead of
      trivially passing it by never mentioning a number at all. Numbers in
      the narrow `1..TOP_N_PARTNERS` range are skipped when choosing which
      ones to echo (falling back to them only if nothing else is available):
      finding B7/AWR-02 narrowed when a bare small integer counts as
      grounded to a narrow allowlist of *structural* phrasing ("top 3",
      "rank 3", "3 years") — this mock's own generic "including values: X,
      Y, Z" framing doesn't supply any of that context, so a coincidentally
      small echoed number (e.g. equal to a table's row count — which can
      legitimately be 0 for an empty result — or a row's rank, both always
      in `0..TOP_N_PARTNERS`) is no longer reliably grounded the way an
      echoed year or dollar value always unconditionally is.

    Driven by the output schema's own field name (see `_build_mock_instance`
    below) rather than importing `describe_item`/`summarize` directly, which
    would create a circular import (both import `app.models`).
    """
    if field_name == "description":
        excerpt = " ".join(_strip_numeric_words(user_content).split()[:20])
        body = excerpt or "a general trade classification"
        return f"[MockLLM deterministic output] {body}".strip()

    if field_name == "normalized_query":
        return user_content.strip()

    numbers = extract_numbers(user_content)
    if numbers:
        unambiguous = [n for n in numbers if not (n == round(n) and 0 <= n <= TOP_N_PARTNERS)]
        sample = ", ".join(str(n) for n in (unambiguous or numbers)[:3])
        return f"[MockLLM deterministic output] Reflects provided data, including values: {sample}."
    excerpt = " ".join(user_content.split()[:20])
    return f"[MockLLM deterministic output] {excerpt}".strip()


_HS6_PATTERN = re.compile(r"\b\d{6}\b")

# Not load-bearing for MockLLM's job (no existing field guardrail-checks a
# bare float the way `describe_item`'s numbers are checked) — a fixed,
# schema-valid constant is enough for `app.search.rerank.RankedCandidate.
# relevance_score` and any future plain-`float` structured-output field.
_MOCK_FLOAT_VALUE = 0.5

# `app.report.source_relevance.AgricultureRelevanceCheck.is_agricultural`
# is the first plain-`bool` structured-output field — under
# `LLM_PROVIDER=mock` this call never fires for real (mock mode has no
# real judgment to make), so a fixed, schema-valid constant is enough,
# same reasoning as `_MOCK_FLOAT_VALUE`.
_MOCK_BOOL_VALUE = True

# `generate_grounded`'s mock counterpart (2026-09-02, Step 4 hardening) —
# a deliberately obviously-fake URL/title, same "fixed, schema-valid
# constant, never mistaken for the real thing" reasoning as
# `_MOCK_FLOAT_VALUE`/`_MOCK_BOOL_VALUE` above.
_MOCK_CITATION = GroundingCitation(
    source_url="https://mock.example/citation", title="Mock citation"
)


def _list_item_model(annotation: Any) -> type[BaseModel] | None:
    """If `annotation` is exactly `list[SomeModel]`, return `SomeModel`;
    otherwise `None`. Used to detect the one new list shape `MockLLM`
    supports (`app.search.rerank.RerankOutput.ranked_candidates`) without
    handling `list[...]` in general."""
    if get_origin(annotation) is not list:
        return None
    args = get_args(annotation)
    if len(args) != 1 or not (isinstance(args[0], type) and issubclass(args[0], BaseModel)):
        return None
    return args[0]


def _list_max_length(field: FieldInfo) -> int | None:
    """`Field(max_length=...)` on a list field lands in `FieldInfo.metadata`
    as an `annotated_types.MaxLen`, not a plain attribute (verified against
    the installed `pydantic` directly) — read it back the same way."""
    for constraint in field.metadata:
        max_length = getattr(constraint, "max_length", None)
        if isinstance(max_length, int):
            return max_length
    return None


def _mock_nested_instance(item_model: type[BaseModel], *, hs_code: str, user_content: str) -> Any:
    """Build one `item_model` instance for `_mock_hs_code_list` below.
    `hs_code` is supplied directly (never routed through `_mock_text_for`,
    which returns non-numeric prose that would fail that field's own
    `pattern=r"^\\d{6}$"` constraint) — every other field falls back to the
    same `str`/`float` handling `_build_mock_instance` itself supports."""
    values: dict[str, Any] = {}
    for name, field in item_model.model_fields.items():
        if name == "hs_code":
            values[name] = hs_code
        elif field.annotation is str:
            values[name] = _mock_text_for(user_content, field_name=name)
        elif field.annotation is float:
            values[name] = _MOCK_FLOAT_VALUE
        elif field.annotation is bool:
            values[name] = _MOCK_BOOL_VALUE
        else:
            raise NotImplementedError(
                f"MockLLM has no generic mock strategy for {item_model.__name__}.{name}: "
                f"{field.annotation!r} (only plain `str`/`float`/`bool` fields are supported)"
            )
    return item_model.model_validate(values)


def _mock_hs_code_list(
    item_model: type[BaseModel], *, field: FieldInfo, user_content: str
) -> list[Any]:
    """Every distinct 6-digit HS code mentioned in `user_content`, each
    turned into one `item_model` instance — the real rerank prompt
    (`app.search.rerank._build_user_content`) always lists candidate codes
    as plain text, the same trick `_mock_text_for` already uses for
    `summarize`'s numbers. Grounded by construction: every code this
    produces necessarily came from the prompt itself, so a full graph run
    under `LLM_PROVIDER=mock` exercises `app.guardrails.check_hs_codes_grounded`
    meaningfully instead of trivially passing it. Zero codes found is a
    caller bug (a rerank call with no candidates), not a case to paper over
    with a fabricated placeholder — raises rather than returning an empty
    or invented list."""
    codes = list(dict.fromkeys(_HS6_PATTERN.findall(user_content)))
    if not codes:
        raise ValueError(
            f"MockLLM found no 6-digit HS codes in user_content to build "
            f"{item_model.__name__} instances from"
        )
    max_items = _list_max_length(field)
    if max_items is not None:
        codes = codes[:max_items]
    return [
        _mock_nested_instance(item_model, hs_code=code, user_content=user_content) for code in codes
    ]


def _build_mock_instance[U: BaseModel](schema: type[U], *, user_content: str) -> U:
    """Build a schema-valid canned instance generically from `schema`'s own
    fields — works for any structured-output schema built from `str`/
    `float` fields, or a `list[NestedModel]` field where `NestedModel` has
    an `hs_code` field, without `MockLLM` needing to import
    `describe_item`'s/`summarize`'s/`app.search.rerank`'s specific output
    types (which would create a circular import: those modules import
    `app.models`)."""
    field_values: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        item_model = _list_item_model(field.annotation)
        if field.annotation is str:
            field_values[name] = _mock_text_for(user_content, field_name=name)
        elif field.annotation is float:
            field_values[name] = _MOCK_FLOAT_VALUE
        elif field.annotation is bool:
            field_values[name] = _MOCK_BOOL_VALUE
        elif item_model is not None and "hs_code" in item_model.model_fields:
            field_values[name] = _mock_hs_code_list(
                item_model, field=field, user_content=user_content
            )
        else:
            raise NotImplementedError(
                f"MockLLM has no generic mock strategy for {schema.__name__}.{name}: "
                f"{field.annotation!r} (only plain `str`/`float`/`bool` fields and "
                f"`list[NestedModel]` fields where `NestedModel` has an `hs_code` "
                f"field are supported)"
            )
    return schema.model_validate(field_values)


class MockLLM:
    """Deterministic, zero-token-spend model used whenever
    `LLM_PROVIDER=mock` (master brief §6). All CI, all unit tests, and all
    frontend development run against this."""

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        return _build_mock_instance(schema, user_content=user_content)

    async def generate_grounded(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> GroundedResult[T]:
        return GroundedResult(
            value=_build_mock_instance(schema, user_content=user_content),
            citations=[_MOCK_CITATION],
        )


# One round-robin fairness counter per role, persisted at module scope (not
# owned by any single `GeminiScheduler` instance) — `get_model_for_role` is
# called fresh on every request (every real call site: app/main.py's route
# handlers, app/nodes/describe_item.py, app/nodes/summarize.py all call it
# once per request/graph-run, matching this file's pre-existing per-request
# `GeminiModelClient` construction pattern), so without this, each
# request's own instance-owned counter would reset to 0 and every request's
# first attempt would always favor the same credential — see
# `app.gemini_scheduler.scheduler.GeminiScheduler.__init__`'s own docstring
# for the full finding (originally established for `LoadBalancedGeminiModelClient`,
# the class this scheduler replaced, 2026-09-04).
_role_fairness_counters: dict[NodeRole, itertools.count[int]] = {}


def get_model_for_role(role: NodeRole, *, provider: Literal["gemini", "mock"]) -> ModelClient:
    """Return the configured model client for a given node role.

    Real-provider branch returns a `GeminiScheduler` (`app.gemini_scheduler`)
    when more than one credential is configured (`Settings.
    gemini_credentials`, or the legacy `gemini_key_pool` fallback when that's
    empty), otherwise a plain single-key `GeminiModelClient` — byte-for-byte
    the same object every existing deployment already got when only one key
    is configured. Every call site already treats the return value as an
    opaque `ModelClient`, so this branch is invisible to callers either way
    (docs/PLAN.md §3's "swapping providers is one file").
    """
    if provider == "mock":
        return MockLLM()

    from app.gemini_scheduler.concurrency import get_concurrency
    from app.gemini_scheduler.credentials import build_credential_pool
    from app.gemini_scheduler.health import get_health_store
    from app.gemini_scheduler.scheduler import GeminiScheduler
    from app.settings import get_settings

    settings = get_settings()
    model_name = settings.model_utility if role == "utility" else settings.model_analysis
    credentials = build_credential_pool(settings)
    if len(credentials) > 1:
        counter = _role_fairness_counters.setdefault(role, itertools.count())
        return GeminiScheduler(
            model=model_name,
            credentials=credentials,
            health_store=get_health_store(),
            concurrency=get_concurrency(),
            fairness_counter=counter,
        )
    return GeminiModelClient(model=model_name, api_key=credentials[0].api_key)
