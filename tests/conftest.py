"""Shared pytest fixtures and session setup.

Sets safe placeholder values for the required settings fields
(`COMTRADE_API_KEY`, `GEMINI_API_KEY`, `AGMARKNET_API_KEY`, ...) *before* any test module
imports `app.*`, so the suite never depends on a real `.env` file or real
secrets — see `.env.example` for what each field means and
`app/settings.py` for why they're required at all ("fails loudly at
startup"). `LLM_PROVIDER=mock` matches the master brief §6 requirement that
all CI and all tests run against the mock provider, zero token spend.
"""

from __future__ import annotations

import os

os.environ.setdefault("COMTRADE_API_KEY", "test-placeholder-comtrade-key")
os.environ.setdefault("GEMINI_API_KEY", "test-placeholder-gemini-key")
# Real env vars take precedence over `.env` (pydantic-settings' own
# precedence order) - without this, a developer's real local `.env` setting
# GEMINI_API_KEYS_EXTRA (the load balancer's extra-keys field,
# app/settings.py) would leak into every bare `Settings()` construction in
# this suite, exactly like every other field pre-empted below.
os.environ.setdefault("GEMINI_API_KEYS_EXTRA", "[]")
os.environ.setdefault("AGMARKNET_API_KEY", "test-placeholder-agmarknet-key")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("MODEL_UTILITY", "mock-utility")
os.environ.setdefault("MODEL_ANALYSIS", "mock-analysis")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./.pytest-local.db")
os.environ.setdefault("ENVIRONMENT", "test")
