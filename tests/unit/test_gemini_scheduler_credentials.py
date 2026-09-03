"""Unit tests for `app.gemini_scheduler.credentials` -- project grouping,
the legacy-pool fallback, and env-var resolution/failure modes."""

from __future__ import annotations

import json

import pytest

from app.gemini_scheduler.credentials import GeminiCredential, build_credential_pool
from app.settings import Settings


def _set_gemini_credentials(
    monkeypatch: pytest.MonkeyPatch, entries: list[dict[str, object]]
) -> None:
    monkeypatch.setenv("GEMINI_CREDENTIALS", json.dumps(entries))


@pytest.mark.unit
def test_legacy_fallback_when_no_structured_credentials_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    monkeypatch.setenv("GEMINI_API_KEYS_EXTRA", '["key-b", "key-c"]')
    settings = Settings()

    pool = build_credential_pool(settings)

    assert pool == [
        GeminiCredential(id="legacy-0", project_id="legacy-0", api_key="primary-key"),
        GeminiCredential(id="legacy-1", project_id="legacy-1", api_key="key-b"),
        GeminiCredential(id="legacy-2", project_id="legacy-2", api_key="key-c"),
    ]


@pytest.mark.unit
def test_legacy_fallback_never_consolidates_keys_into_one_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The safe default when project grouping is genuinely unknown: every
    key gets its own distinct project_id, never incorrectly consolidated."""
    monkeypatch.setenv("GEMINI_API_KEYS_EXTRA", '["key-b"]')
    settings = Settings()

    pool = build_credential_pool(settings)

    assert len({credential.project_id for credential in pool}) == len(pool)


@pytest.mark.unit
def test_structured_credentials_group_by_shared_project_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_gemini_credentials(
        monkeypatch,
        [
            {"id": "cred-1", "env_var": "GEMINI_CRED_1", "project_id": "proj-a"},
            {"id": "cred-2", "env_var": "GEMINI_CRED_2", "project_id": "proj-a"},
            {"id": "cred-3", "env_var": "GEMINI_CRED_3", "project_id": "proj-b"},
        ],
    )
    settings = Settings()
    env = {"GEMINI_CRED_1": "k1", "GEMINI_CRED_2": "k2", "GEMINI_CRED_3": "k3"}

    pool = build_credential_pool(settings, env=env)

    project_ids = {credential.id: credential.project_id for credential in pool}
    assert project_ids == {"cred-1": "proj-a", "cred-2": "proj-a", "cred-3": "proj-b"}


@pytest.mark.unit
def test_enabled_credential_with_missing_env_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_gemini_credentials(
        monkeypatch, [{"id": "cred-1", "env_var": "GEMINI_MISSING", "project_id": "proj-a"}]
    )
    settings = Settings()

    with pytest.raises(ValueError, match="GEMINI_MISSING"):
        build_credential_pool(settings, env={})


@pytest.mark.unit
def test_enabled_credential_with_blank_env_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_gemini_credentials(
        monkeypatch, [{"id": "cred-1", "env_var": "GEMINI_BLANK", "project_id": "proj-a"}]
    )
    settings = Settings()

    with pytest.raises(ValueError, match="cred-1"):
        build_credential_pool(settings, env={"GEMINI_BLANK": "   "})


@pytest.mark.unit
def test_disabled_credential_does_not_require_its_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_gemini_credentials(
        monkeypatch,
        [
            {
                "id": "cred-1",
                "env_var": "GEMINI_NOT_SET_YET",
                "project_id": "proj-a",
                "enabled": False,
            }
        ],
    )
    settings = Settings()

    pool = build_credential_pool(settings, env={})

    assert pool == [GeminiCredential(id="cred-1", project_id="proj-a", api_key="", enabled=False)]


@pytest.mark.unit
def test_credential_models_restriction_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_gemini_credentials(
        monkeypatch,
        [
            {
                "id": "cred-1",
                "env_var": "GEMINI_CRED_1",
                "project_id": "proj-a",
                "models": ["gemini-flash-latest"],
            }
        ],
    )
    settings = Settings()

    pool = build_credential_pool(settings, env={"GEMINI_CRED_1": "k1"})

    assert pool[0].supports("gemini-flash-latest") is True
    assert pool[0].supports("gemini-flash-lite-latest") is False


@pytest.mark.unit
def test_credential_with_no_models_restriction_supports_any_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gemini_credentials(
        monkeypatch, [{"id": "cred-1", "env_var": "GEMINI_CRED_1", "project_id": "proj-a"}]
    )
    settings = Settings()

    pool = build_credential_pool(settings, env={"GEMINI_CRED_1": "k1"})

    assert pool[0].supports("anything-at-all") is True


@pytest.mark.unit
def test_disabled_credential_never_supports_any_model() -> None:
    credential = GeminiCredential(id="c", project_id="p", api_key="k", enabled=False)
    assert credential.supports("gemini-flash-latest") is False


@pytest.mark.unit
def test_duplicate_credential_ids_rejected_at_settings_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gemini_credentials(
        monkeypatch,
        [
            {"id": "dup", "env_var": "A", "project_id": "proj-a"},
            {"id": "dup", "env_var": "B", "project_id": "proj-b"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        Settings()
