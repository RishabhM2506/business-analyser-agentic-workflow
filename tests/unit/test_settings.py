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
    assert settings.max_model_calls_per_thread == 2
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
