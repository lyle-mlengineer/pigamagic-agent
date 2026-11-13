"""Step 0.2: settings load with safe defaults and reject bad values."""

import pytest
from pydantic import ValidationError

from app.agent.config import AgentSettings, get_settings, reset_settings_cache


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with no AGENT_* variables leaking in from the shell."""
    for name in (
        "AGENT_LLM_PROVIDER",
        "AGENT_VISION_PROVIDER",
        "AGENT_IMAGE_PROVIDER",
        "AGENT_WEB_PROVIDER",
        "AGENT_LLM_API_KEY",
        "AGENT_VISION_API_KEY",
        "AGENT_IMAGE_API_KEY",
        "AGENT_WEB_API_KEY",
        "AGENT_LLM_MODEL",
        "AGENT_VISION_MODEL",
        "AGENT_IMAGE_MODEL",
        "AGENT_PROMPT_VERSION",
        "AGENT_MAX_SHOTS",
        "AGENT_MAX_SEARCH_CALLS",
        "AGENT_MAX_REFERENCE_NOTES_TOKENS",
        "AGENT_IDENTITY_SIMILARITY_FLOOR",
        "AGENT_IDENTITY_SIMILARITY_TARGET",
        "AGENT_MEDIA_DIR",
        "AGENT_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_settings_cache()


def _settings_without_env_file() -> AgentSettings:
    """Build settings ignoring any developer .env on disk."""
    return AgentSettings(_env_file=None)


def test_defaults_load_with_no_environment() -> None:
    settings = _settings_without_env_file()

    assert settings.llm_provider == "fixture"
    assert settings.vision_provider == "fixture"
    assert settings.image_provider == "fixture"
    assert settings.web_provider == "fixture"

    assert settings.llm_api_key == ""
    assert settings.image_api_key == ""

    assert settings.prompt_version == "v1"
    assert settings.max_shots == 16
    assert settings.max_search_calls == 6
    assert settings.max_reference_notes_tokens == 300

    assert settings.identity_similarity_floor == 0.45
    assert settings.identity_similarity_target == 0.62

    assert settings.offline is False
    assert settings.strict_research is False


def test_defaults_need_no_credentials_and_use_fixtures() -> None:
    settings = _settings_without_env_file()
    assert settings.uses_fixtures() is True
    assert settings.missing_credentials() == []


def test_env_override_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("AGENT_MAX_SHOTS", "8")
    monkeypatch.setenv("AGENT_PROMPT_VERSION", "  v2  ")

    settings = _settings_without_env_file()

    assert settings.llm_provider == "openai"
    assert settings.max_shots == 8
    # prompt_version is stripped, because whitespace would silently fork the cache.
    assert settings.prompt_version == "v2"


def test_settings_without_file_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_IMAGE_PROVIDER", "google")
    settings = AgentSettings()
    assert settings.image_provider == "google"


def test_invalid_provider_name_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        AgentSettings(_env_file=None, llm_provider="not-a-provider")  # type: ignore[arg-type]
    assert "llm_provider" in str(excinfo.value)


def test_empty_prompt_version_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        AgentSettings(_env_file=None, prompt_version="   ")
    assert "prompt_version" in str(excinfo.value)


def test_target_below_floor_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        AgentSettings(
            _env_file=None,
            identity_similarity_floor=0.8,
            identity_similarity_target=0.5,
        )
    assert "identity_similarity_target" in str(excinfo.value)


def test_max_shots_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        AgentSettings(_env_file=None, max_shots=0)
    with pytest.raises(ValidationError):
        AgentSettings(_env_file=None, max_shots=10_000)


def test_non_fixture_provider_without_key_is_reported() -> None:
    settings = AgentSettings(_env_file=None, llm_provider="openai", llm_api_key="")
    assert settings.uses_fixtures() is False
    assert settings.missing_credentials() == ["openai (key missing)"]

    configured = AgentSettings(_env_file=None, llm_provider="openai", llm_api_key="sk-test")
    assert configured.missing_credentials() == []


def test_hash_relevant_settings_exclude_credentials_and_budgets() -> None:
    """Credentials and budgets must never fork the generation cache."""
    base = AgentSettings(_env_file=None)
    varied = AgentSettings(
        _env_file=None,
        llm_api_key="sk-different",
        max_shots=4,
        max_search_calls=1,
        media_dir="./elsewhere",
    )

    assert base.hash_relevant_settings() == varied.hash_relevant_settings()

    bumped = AgentSettings(_env_file=None, prompt_version="v9")
    assert bumped.hash_relevant_settings() != base.hash_relevant_settings()


def test_get_settings_is_cached_and_resettable() -> None:
    first = get_settings()
    assert get_settings() is first

    reset_settings_cache()
    assert get_settings() is not first
