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

from pydantic import Field
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

    # --- Model routing (master brief §3: node-role -> model, config-driven,
    # never hard-coded; docs/PLAN.md §5.1) -----------------------------------
    llm_provider: Literal["gemini", "mock"] = "mock"
    model_utility: str = "gemini-flash-lite-latest"
    model_analysis: str = "gemini-flash-latest"

    # --- Cost & recursion ceilings, fail closed (docs/PLAN.md §5.5) ---------
    max_model_calls_per_day: int = 500
    max_model_calls_per_thread: int = 2
    recursion_limit: int = 15

    # --- API edge (docs/PLAN.md §6: CORS allowlist, per-IP rate limiting) ---
    cors_allowed_origins: list[str] = Field(default_factory=list)
    rate_limit_per_minute: int = 60

    # --- Structured logging ---------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide `Settings` singleton, constructed on first use."""
    return Settings()  # required fields are supplied by the environment / .env at runtime
