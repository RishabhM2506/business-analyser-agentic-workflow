"""Unit tests for `app/settings.py` — real, implemented config, not a stub."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.settings import Settings


@pytest.mark.unit
def test_settings_reads_required_fields_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMTRADE_API_KEY", "k-comtrade")
    monkeypatch.setenv("GEMINI_API_KEY", "k-gemini")
    settings = Settings()
    assert settings.comtrade_api_key == "k-comtrade"
    assert settings.gemini_api_key == "k-gemini"


@pytest.mark.unit
def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMTRADE_API_KEY", "k-comtrade")
    monkeypatch.setenv("GEMINI_API_KEY", "k-gemini")
    settings = Settings()
    assert settings.llm_provider == "mock"
    # A per-session, not per-analysis, ceiling (docs/PLAN.md §5.5, finding
    # B2/ARCH-02) - 20 supports ~10 full analyses per session, not just the
    # first one.
    assert settings.max_model_calls_per_thread == 20
    assert settings.max_model_calls_per_day == 500
    assert settings.recursion_limit == 15
    assert settings.checkpoint_retention_days == 90


@pytest.mark.unit
def test_settings_llm_provider_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMTRADE_API_KEY", "k-comtrade")
    monkeypatch.setenv("GEMINI_API_KEY", "k-gemini")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValidationError):
        Settings()


# --- gemini_api_keys_extra / gemini_key_pool (2026-08-26 load balancer) --------


@pytest.mark.unit
def test_gemini_api_keys_extra_defaults_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMTRADE_API_KEY", "k-comtrade")
    monkeypatch.setenv("GEMINI_API_KEY", "k-gemini")
    settings = Settings()
    assert settings.gemini_api_keys_extra == []
    # A single-key deployment (every existing one, until this is explicitly
    # configured) must see the pool as exactly [gemini_api_key] - unchanged
    # single-key behavior by default.
    assert settings.gemini_key_pool == ["k-gemini"]


@pytest.mark.unit
def test_gemini_key_pool_puts_the_primary_key_first_then_extras_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMTRADE_API_KEY", "k-comtrade")
    monkeypatch.setenv("GEMINI_API_KEY", "k-a")
    monkeypatch.setenv("GEMINI_API_KEYS_EXTRA", '["k-b", "k-c"]')
    settings = Settings()
    assert settings.gemini_key_pool == ["k-a", "k-b", "k-c"]


@pytest.mark.unit
def test_gemini_key_pool_deduplicates_while_preserving_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real mistake this guards against: re-pasting the primary key into
    # the extra list must not double-count it as two pool members that
    # round-robin to the same underlying key.
    monkeypatch.setenv("COMTRADE_API_KEY", "k-comtrade")
    monkeypatch.setenv("GEMINI_API_KEY", "k-a")
    monkeypatch.setenv("GEMINI_API_KEYS_EXTRA", '["k-b", "k-a", "k-c"]')
    settings = Settings()
    assert settings.gemini_key_pool == ["k-a", "k-b", "k-c"]


@pytest.mark.unit
def test_gemini_api_keys_extra_drops_blank_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMTRADE_API_KEY", "k-comtrade")
    monkeypatch.setenv("GEMINI_API_KEY", "k-a")
    monkeypatch.setenv("GEMINI_API_KEYS_EXTRA", '["k-b", "", "  ", "k-c"]')
    settings = Settings()
    assert settings.gemini_api_keys_extra == ["k-b", "k-c"]
    assert settings.gemini_key_pool == ["k-a", "k-b", "k-c"]
