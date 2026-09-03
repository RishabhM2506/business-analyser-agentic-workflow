"""Credential/project pool for the Gemini Provider Scheduler (Phase 2).

Gemini quotas are primarily associated with the Google Cloud *project*, not
the individual API key -- two credentials belonging to the same project
share one quota pool and must not be treated as independent capacity. This
module is the one place that grouping happens; everything downstream
(`health.py`, `concurrency.py`, `scheduler.py`) keys its state off
`GeminiCredential.project_id`, never off the credential/key itself.

Never holds a raw key value in anything other than `GeminiCredential.
api_key` -- logging call sites elsewhere in this package pass `credential.id`
and `credential.project_id`, never `credential.api_key`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.settings import GeminiCredentialConfig, Settings


@dataclass(frozen=True)
class GeminiCredential:
    """One real, usable Gemini credential -- `project_id` is the quota-
    grouping key every other module in this package uses; `id` is a stable,
    loggable, non-secret identifier for this specific credential within
    its project (multiple credentials can share a `project_id`)."""

    id: str
    project_id: str
    api_key: str
    account_label: str | None = None
    models: frozenset[str] | None = None  # None = every configured model
    enabled: bool = True

    def supports(self, model: str) -> bool:
        return self.enabled and (self.models is None or model in self.models)


def build_credential_pool(
    settings: Settings, *, env: Mapping[str, str] | None = None
) -> list[GeminiCredential]:
    """`settings.gemini_credentials` (structured, project-grouped config) if
    configured; otherwise falls back to today's flat `settings.
    gemini_key_pool` (`gemini_api_key` + `gemini_api_keys_extra`), each key
    given its own distinct synthetic `project_id` -- i.e. **no quota
    consolidation is ever assumed** for a deployment that hasn't explicitly
    declared which keys share a project. Every current deployment (empty
    `gemini_credentials`) keeps working with zero `.env` changes."""
    if settings.gemini_credentials:
        resolved_env = env if env is not None else os.environ
        return [_resolve(config, env=resolved_env) for config in settings.gemini_credentials]
    return [
        GeminiCredential(id=f"legacy-{i}", project_id=f"legacy-{i}", api_key=key)
        for i, key in enumerate(settings.gemini_key_pool)
    ]


def _resolve(config: GeminiCredentialConfig, *, env: Mapping[str, str]) -> GeminiCredential:
    """A *disabled* credential's env var is never required to be set
    (lets an operator pre-declare a credential ahead of having the real
    secret) -- an *enabled* one that can't actually authenticate is a real
    misconfiguration and fails loudly at startup, matching `Settings`'
    own "fails loudly" contract (`app/settings.py`'s module docstring)."""
    if not config.enabled:
        return GeminiCredential(
            id=config.id,
            project_id=config.project_id,
            api_key="",
            account_label=config.account_label,
            models=frozenset(config.models) if config.models else None,
            enabled=False,
        )
    api_key = env.get(config.env_var, "")
    if not api_key.strip():
        raise ValueError(
            f"gemini_credentials entry {config.id!r} is enabled and names env var "
            f"{config.env_var!r}, but that variable is unset or empty"
        )
    return GeminiCredential(
        id=config.id,
        project_id=config.project_id,
        api_key=api_key,
        account_label=config.account_label,
        models=frozenset(config.models) if config.models else None,
        enabled=True,
    )
