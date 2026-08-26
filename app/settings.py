"""Typed application configuration (Pydantic Settings).

Loaded from process environment variables and, if present, a local `.env`
file (never committed — see `.env.example` for the documented field set).
Fails loudly at startup: a `Settings()` construction raises
`pydantic.ValidationError` immediately if a required field (an API key,
for instance) is missing, rather than the app starting in a half-configured
state and failing confusingly on first use.

This module is config, not business logic — every field here is named and
typed directly from docs/PLAN.md §5 (cost model / ceilings) and §6
(security model), plus the master brief's model-routing and multi-tenancy
seams (§3, §6).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Runtime ------------------------------------------------------------
    environment: Literal["development", "test", "production"] = "development"

    # --- External API credentials (docs/PLAN.md §6: env-injected, never
    # committed, separate keys per environment) ------------------------------
    comtrade_api_key: str
    gemini_api_key: str
    # Additional Gemini API keys for the round-robin/failover load balancer
    # (2026-08-26 addition, `app.models.LoadBalancedGeminiModelClient`) —
    # JSON array, same convention as `cors_allowed_origins` below. `[]`
    # (the default) means "just gemini_api_key, no pooling," so every
    # existing deployment that hasn't set this keeps its current
    # single-key `GeminiModelClient` behavior unchanged.
    # `gemini_api_key` is always the pool's first key — never duplicated
    # in this list, see `Settings.gemini_key_pool`.
    gemini_api_keys_extra: list[str] = Field(default_factory=list)
    # data.gov.in resource key for the Agmarknet daily mandi-price feed
    # (app/pipeline/agmarknet.py). Required even when the Agmarknet job
    # isn't running, matching this file's existing "fails loudly at
    # startup" contract for every other credential.
    agmarknet_api_key: str
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None
    langsmith_tracing_enabled: bool = False
    # Deploy identifier attached to every trace (docs/PLAN.md §4.1's
    # observability.py file-tree note: "trace metadata (tenant_id,
    # prompt_version, release SHA)"). Typically set by CI/the deploy
    # pipeline to the built commit SHA; "unknown" is a safe default for
    # local dev, where no such pipeline sets it, rather than a required
    # field that would fail Settings construction for every developer who
    # hasn't wired one up.
    release_sha: str = "unknown"

    # --- Datastore (checkpointer + cache tables; docs/PLAN.md §2.2, §9) -----
    # sqlite locally by default; a postgresql+asyncpg:// URL in any deployed
    # environment (docs/PLAN.md §2.2's "SqliteSaver locally, PostgresSaver in
    # any deployed environment").
    database_url: str = "sqlite+aiosqlite:///./local.db"
    checkpoint_retention_days: int = 90  # docs/PLAN.md §6: explicit 90-day rolling retention
    # Trade pipeline's FX cache only (app/fx/cache.py) — nothing else in this
    # app uses Redis (the existing response/tool caches are deliberately
    # in-process, see app/cache/tool_cache.py's own docstring). Default
    # matches the new `redis` service in docker-compose.yml.
    redis_url: str = "redis://localhost:6379/0"

    # --- Model routing (master brief §3: node-role -> model, config-driven,
    # never hard-coded; docs/PLAN.md §5.1) -----------------------------------
    llm_provider: Literal["gemini", "mock"] = "mock"
    model_utility: str = "gemini-flash-lite-latest"
    model_analysis: str = "gemini-flash-latest"
    # Product-search embeddings (app/search/embeddings.py, 2026-08-20 roadmap
    # decision) — verified live against the real model, 3072-dim output.
    # Shares `llm_provider`'s "gemini"/"mock" switch rather than a second
    # setting: there is no scenario where chat is mocked but embeddings are
    # real, or vice versa.
    model_embedding: str = "gemini-embedding-2-preview"

    # --- Cost & recursion ceilings, fail closed (docs/PLAN.md §5.5) ---------
    max_model_calls_per_day: int = 500
    # A ceiling per *session*, not per single analysis (docs/PLAN.md §5.5,
    # finding B2/ARCH-02): a thread is created once on "Start my process"
    # and reused for every item a user looks at in that session, and one
    # completed analysis costs exactly 2 model calls (describe_item +
    # summarize). The original value of 2 was exactly the happy-path cost
    # of the *first* analysis, leaving zero headroom for a second analysis
    # on the same thread (BUDGET_EXCEEDED on every user's second item, every
    # time) and zero retry headroom within even the first analysis (a
    # guardrail rejection or a transient failure permanently exhausted the
    # thread). 20 supports roughly 10 full analyses per session plus real
    # retry headroom.
    max_model_calls_per_thread: int = 20
    recursion_limit: int = 15

    # --- API edge (docs/PLAN.md §6: CORS allowlist, per-IP rate limiting) ---
    cors_allowed_origins: list[str] = Field(default_factory=list)
    rate_limit_per_minute: int = 60
    # Request body-size ceiling, checked from the `Content-Length` header
    # before any body parsing (finding B9/QA-02, master brief §8: "input
    # validation and size limits at the API edge... before any model spend
    # occurs"). 64 KiB is generous headroom — QA-02 confirmed live that the
    # largest legitimate `TradeQuery` payload is well under 1 KB.
    max_request_body_bytes: int = 64 * 1024

    # --- Structured logging ---------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("gemini_api_keys_extra")
    @classmethod
    def _drop_blank_extra_keys(cls, value: list[str]) -> list[str]:
        """A stray empty string in a hand-edited JSON array (e.g.
        `["", "AQ...."]`) must never become a "real" pool member that every
        request round-robins into and fails against — silently drop blanks
        rather than trusting the env var was well-formed."""
        return [key for key in value if key.strip()]

    @property
    def gemini_key_pool(self) -> list[str]:
        """`gemini_api_key` first, then `gemini_api_keys_extra`, with any
        accidental duplicate (e.g. the primary key re-pasted into the extra
        list) removed while preserving order — every other call site that
        wants "the" key still reads `gemini_api_key` directly and is
        unaffected by this property existing."""
        seen: dict[str, None] = {}
        for key in [self.gemini_api_key, *self.gemini_api_keys_extra]:
            seen.setdefault(key, None)
        return list(seen)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide `Settings` singleton, constructed on first use."""
    return Settings()  # required fields are supplied by the environment / .env at runtime
