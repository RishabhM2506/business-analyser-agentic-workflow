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

from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from app.guardrails import extract_numbers

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


def _mock_text_for(user_content: str) -> str:
    """Deterministic, schema-agnostic placeholder text for `MockLLM`.

    Never invents a number: if `user_content` contains any (which, for
    `summarize`, is the compact rendered trade table — see
    `app/nodes/summarize.py`), a few are echoed verbatim via
    `app.guardrails.extract_numbers` — the exact same extraction the output
    guardrail uses — so the mock output is grounded *by construction* and a
    full graph run under `LLM_PROVIDER=mock` exercises the guardrail
    meaningfully instead of trivially passing it by never mentioning a
    number at all. `describe_item`'s mock output isn't guardrail-checked,
    so this same helper works for both roles without needing to know which
    one called it (no import of either node module -> no circular import).
    """
    numbers = extract_numbers(user_content)
    if numbers:
        sample = ", ".join(str(n) for n in numbers[:3])
        return f"[MockLLM deterministic output] Reflects provided data, including values: {sample}."
    excerpt = " ".join(user_content.split()[:20])
    return f"[MockLLM deterministic output] {excerpt}".strip()


def _build_mock_instance[U: BaseModel](schema: type[U], *, user_content: str) -> U:
    """Build a schema-valid canned instance generically from `schema`'s own
    fields — works for any single-or-multi-`str`-field structured-output
    schema without `MockLLM` needing to import `describe_item`'s or
    `summarize`'s specific output types (which would create a circular
    import: those modules import `app.models`)."""
    field_values: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if field.annotation is str:
            field_values[name] = _mock_text_for(user_content)
        else:
            raise NotImplementedError(
                f"MockLLM has no generic mock strategy for {schema.__name__}.{name}: "
                f"{field.annotation!r} (only plain `str` fields are supported)"
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
