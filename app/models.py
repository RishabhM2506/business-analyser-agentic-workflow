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
"""

from __future__ import annotations

import re
from typing import Any, Literal, Protocol, TypeVar, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from app.guardrails import extract_numbers
from app.nodes.aggregate import TOP_N_PARTNERS

NodeRole = Literal["utility", "analysis"]

T = TypeVar("T", bound=BaseModel)

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


class GeminiModelClient:
    """Real `langchain-google-genai`-backed adapter."""

    def __init__(self, *, model: str, api_key: str) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._chat = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=0.2,  # low but nonzero: consistent prose, not maximally deterministic
            max_retries=_GEMINI_MAX_RETRIES,
            timeout=_GEMINI_TIMEOUT_SECONDS,
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
        else:
            raise NotImplementedError(
                f"MockLLM has no generic mock strategy for {item_model.__name__}.{name}: "
                f"{field.annotation!r} (only plain `str`/`float` fields are supported)"
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
        elif item_model is not None and "hs_code" in item_model.model_fields:
            field_values[name] = _mock_hs_code_list(
                item_model, field=field, user_content=user_content
            )
        else:
            raise NotImplementedError(
                f"MockLLM has no generic mock strategy for {schema.__name__}.{name}: "
                f"{field.annotation!r} (only plain `str`/`float` fields and "
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


def get_model_for_role(role: NodeRole, *, provider: Literal["gemini", "mock"]) -> ModelClient:
    """Return the configured model client for a given node role."""
    if provider == "mock":
        return MockLLM()

    from app.settings import get_settings

    settings = get_settings()
    model_name = settings.model_utility if role == "utility" else settings.model_analysis
    return GeminiModelClient(model=model_name, api_key=settings.gemini_api_key)
