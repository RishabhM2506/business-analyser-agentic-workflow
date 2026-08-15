"""Node-role -> model instance mapping (master brief §3: "A model-selection
layer (`models.py`) that maps node role to model, so swapping providers is
one file"). Also owns the `MockLLM` provider switch
(`LLM_PROVIDER=mock`, master brief §6) — mandatory for CI/tests/frontend
dev so zero token spend is possible end to end.

# TODO(Phase 3): implement `get_model_for_role` returning a real
# langchain-google-genai chat model when `settings.llm_provider == "gemini"`,
# or `MockLLM` (deterministic, schema-valid canned output) when
# `settings.llm_provider == "mock"`.
"""

from __future__ import annotations

from typing import Literal, Protocol

NodeRole = Literal["utility", "analysis"]


class ModelClient(Protocol):
    """Minimal interface every model-provider adapter (real or mock) must
    satisfy — deliberately narrow so swapping Gemini for another provider
    is a new adapter, not a new interface."""

    async def generate(self, *, system_prompt: str, user_content: str) -> str: ...


class MockLLM:
    """Deterministic, zero-token-spend model used whenever
    `LLM_PROVIDER=mock` (master brief §6). All CI, all unit tests, and all
    frontend development run against this."""

    async def generate(self, *, system_prompt: str, user_content: str) -> str:
        raise NotImplementedError  # TODO(Phase 3): return deterministic schema-valid canned output.


def get_model_for_role(role: NodeRole, *, provider: Literal["gemini", "mock"]) -> ModelClient:
    """Return the configured model client for a given node role."""
    raise NotImplementedError  # TODO(Phase 3): implement provider switch (gemini vs MockLLM).
