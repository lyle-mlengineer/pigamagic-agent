"""Agent configuration.

Everything tunable lives here and is read from the environment (or `.env`).
Defaults are chosen so that `AgentSettings()` succeeds with **no environment
at all** — that is what lets Steps 1-7 run offline against fixtures.

Two rules:
1. Never import a provider SDK to read settings. Config is pure data.
2. Any value that changes a generated image MUST participate in the prompt
   hash (see agent/hashing.py, Step 0.3), otherwise the cache silently serves
   stale photos. The fields marked "hash-relevant" below are collected by
   `hash_relevant_settings()`.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Provider names are validated, not free-form, so a typo fails at startup
# rather than at the first generation call.
ProviderName = Literal[
    "fixture",
    "openai",
    "anthropic",
    "google",
    "gemini",
    "fal",
    "flux",
    "nvidia",
    "huggingface",
    "hf",
    "local",
    "qwen",
    "flux-local",
    "sd",
    "colab",
    "custom",
]

StorageName = Literal[
    "local",
    "file",
    "s3",
    "backblaze",
    "b2",
    "r2",
    "minio",
    "wasabi",
]

# Which similarity floor is "acceptable" vs "good enough to target".
# A photo below `floor` triggers an identity retry; above `target` is ideal.
DEFAULT_SIMILARITY_FLOOR = 0.45
DEFAULT_SIMILARITY_TARGET = 0.62


class AgentSettings(BaseSettings):
    """Runtime settings for the agent pipeline.

    Field names map to upper-case environment variables with the `AGENT_`
    prefix, e.g. `AGENT_LLM_PROVIDER`. Keys default to empty strings so the
    object is constructible with no secrets present.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider selection -------------------------------------------------
    llm_provider: ProviderName = "fixture"
    vision_provider: ProviderName = "fixture"
    image_provider: ProviderName = "fixture"
    web_provider: ProviderName = "fixture"

    # --- Credentials (never logged, never defaulted to a real value) --------
    llm_api_key: str = ""
    vision_api_key: str = ""
    image_api_key: str = ""
    web_api_key: str = ""

    # --- NVIDIA NIM / Black Forest Labs ------------------------------------
    # The sample implementation reads these two names, so they are kept as-is.
    # `nvidia_api_key` is an alias for `image_api_key`; either will work.
    nvidia_api_key: str = ""
    nvidia_image_base_url: str = ""
    # A self-hosted NIM accepts real reference images (up to 8). The hosted
    # preview at build.nvidia.com accepts only predefined example images, so
    # set this true only when pointing at your own NIM container.
    nvidia_nim_self_hosted: bool = False

    # --- Models -------------------------------------------------------------
    llm_model: str = ""
    vision_model: str = ""
    image_model: str = ""

    # --- Prompt versioning (hash-relevant) ----------------------------------
    # Bumping this intentionally invalidates every cached generation.
    prompt_version: str = "v1"

    # --- Budgets and limits ------------------------------------------------
    max_shots: int = Field(default=16, ge=1, le=64)
    max_search_calls: int = Field(default=6, ge=1, le=20)
    max_reference_notes_tokens: int = Field(default=300, ge=0, le=4000)
    max_pages_per_venue: int = Field(default=6, ge=1, le=20)
    max_images_per_venue: int = Field(default=8, ge=0, le=30)

    # Hard ceiling on real spend for image generation, in USD. 0 means
    # unlimited (the default, so fixtures and tests are unaffected). A positive
    # value makes BudgetLedger refuse any call that would exceed it, which is
    # the guard that stops a runaway loop from draining an account.
    image_budget_usd: float = Field(default=0.0, ge=0.0)
    # Retry attempts for outbound provider calls (1 = no retry).
    provider_max_attempts: int = Field(default=3, ge=1, le=6)
    provider_timeout_s: float = Field(default=120.0, gt=0.0, le=900.0)

    # --- Quality thresholds (hash-relevant) --------------------------------
    identity_similarity_floor: float = Field(default=DEFAULT_SIMILARITY_FLOOR, ge=0.0, le=1.0)
    identity_similarity_target: float = Field(default=DEFAULT_SIMILARITY_TARGET, ge=0.0, le=1.0)

    # --- Storage -----------------------------------------------------------
    storage_provider: StorageName = "local"
    media_dir: str = "./var/media"
    media_base_url: str = ""

    # S3-compatible object storage (AWS S3, Backblaze B2, R2, MinIO, Wasabi).
    storage_bucket: str = ""
    storage_endpoint: str = ""
    storage_region: str = ""
    storage_access_key_id: str = ""
    storage_secret_access_key: str = ""
    storage_prefix: str = "plusone"
    # Public/CDN base for read URLs. Leave blank to serve through the app.
    storage_public_base_url: str = ""

    # --- Behaviour switches -------------------------------------------------
    # When true, venue research that finds nothing raises instead of
    # degrading to a conventions-based profile. Useful in tests.
    strict_research: bool = False
    # Offline mode forbids any outbound HTTP. Fixture providers stay usable.
    offline: bool = False

    # --- Admin surface ------------------------------------------------------
    # Shared secret for `/api/v1/admin/*`. Empty means the admin routes are
    # **disabled**: an unset token must never mean "open to everyone", which is
    # the classic way a review console leaks every flagged photo.
    admin_token: str = ""

    @field_validator("prompt_version")
    @classmethod
    def _non_empty_prompt_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt_version must not be empty; it keys the generation cache.")
        return value.strip()

    @field_validator("media_dir")
    @classmethod
    def _non_empty_media_dir(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("media_dir must not be empty.")
        return value.strip()

    @model_validator(mode="after")
    def _target_not_below_floor(self) -> "AgentSettings":
        if self.identity_similarity_target < self.identity_similarity_floor:
            raise ValueError(
                "identity_similarity_target must be >= identity_similarity_floor "
                f"(got {self.identity_similarity_target} < {self.identity_similarity_floor})."
            )
        return self

    # --- Derived helpers ---------------------------------------------------

    def uses_fixtures(self) -> bool:
        """True when every provider is offline-capable."""
        return all(
            name == "fixture"
            for name in (
                self.llm_provider,
                self.vision_provider,
                self.image_provider,
                self.web_provider,
            )
        )

    def missing_credentials(self) -> list[str]:
        """Names of providers selected but lacking a key. Empty under fixtures."""
        missing: list[str] = []
        for provider_name, key in (
            (self.llm_provider, self.llm_api_key),
            (self.vision_provider, self.vision_api_key),
            (self.image_provider, self.image_api_key),
            (self.web_provider, self.web_api_key),
        ):
            if provider_name != "fixture" and not key.strip():
                missing.append(f"{provider_name} (key missing)")
        return missing

    def hash_relevant_settings(self) -> dict[str, object]:
        """The subset of settings that must invalidate a cached image if changed.

        Deliberately excludes budgets, credentials and storage paths: those
        change *how* we work, not *what* the photo looks like.
        """
        return {
            "prompt_version": self.prompt_version,
            "llm_model": self.llm_model,
            "vision_model": self.vision_model,
            "image_model": self.image_model,
            "identity_similarity_floor": self.identity_similarity_floor,
            "identity_similarity_target": self.identity_similarity_target,
        }


@lru_cache
def get_settings() -> AgentSettings:
    """Process-wide settings. Cached so env is read once."""
    return AgentSettings()


def reset_settings_cache() -> None:
    """Clear the cache. Tests use this after monkeypatching the environment."""
    get_settings.cache_clear()


# Convenience alias matching the reference shell's `app.config.settings` style.
settings = get_settings()
