"""Provider registry and factory (design.md §8.2).

One place decides which adapter a configured provider name maps to. The pipeline
never imports a vendor module directly; it asks for a provider by name, so the
fixture and the real vendors are interchangeable at runtime.

Adding a vendor means one entry here plus one adapter module — nothing else in
the codebase changes.
"""

from __future__ import annotations

from typing import Any, Callable

from pigamagic_agent.agent.config import AgentSettings, get_settings
from pigamagic_agent.agent.providers.base import (
    ImageProvider,
    LLMProvider,
    ProviderNotConfigured,
    VisionProvider,
    WebProvider,
)
from pigamagic_agent.agent.providers.http_base import BudgetLedger


def build_image_provider(
    settings: AgentSettings | None = None,
    *,
    budget: BudgetLedger | None = None,
    name: str | None = None,
) -> ImageProvider:
    """Construct the configured image provider.

    `fixture` needs no credentials and never touches the network, so tests and
    the CLI are unaffected by this module existing.
    """
    active = settings or get_settings()
    chosen = (name or active.image_provider or "fixture").strip().casefold()

    if chosen in {"fixture", "fake", "offline"}:
        from pigamagic_agent.agent.providers.image_fixture import FixtureImageProvider

        return FixtureImageProvider()

    if chosen in {"gemini", "google", "nano-banana"}:
        from pigamagic_agent.agent.providers.image_gemini import GeminiImageProvider

        return GeminiImageProvider(
            api_key=active.image_api_key,
            model=active.image_model,
            settings=active,
            budget=budget,
        )

    if chosen in {"openai", "gpt-image"}:
        from pigamagic_agent.agent.providers.image_openai import OpenAIImageProvider

        return OpenAIImageProvider(
            api_key=active.image_api_key,
            model=active.image_model,
            settings=active,
            budget=budget,
        )

    if chosen in {
        "huggingface", "hf", "local", "colab", "transformers", "diffusers",
        "qwen", "flux-local", "sd",
    }:
        # `qwen` selects the identity-preserving Qwen edit model by default.
        from pigamagic_agent.agent.providers.image_huggingface import HuggingFaceImageProvider

        from pigamagic_agent.agent.providers.image_huggingface import DEFAULT_MODEL

        model = active.image_model
        if not model and chosen == "qwen":
            model = "Qwen/Qwen-Image-Edit-2509"
        return HuggingFaceImageProvider(
            api_key=active.image_api_key,
            model=model or DEFAULT_MODEL,
            settings=active,
            budget=budget,
        )

    if chosen in {"nvidia", "nim", "bfl", "black-forest-labs"}:
        from pigamagic_agent.agent.providers.image_nvidia import NvidiaImageProvider

        # The sample implementation used `nvidia_api_key`; accept either name.
        key = active.image_api_key or active.nvidia_api_key
        return NvidiaImageProvider(
            api_key=key,
            model=active.image_model,
            base_url=active.nvidia_image_base_url,
            settings=active,
            budget=budget,
        )

    if chosen in {"fal", "flux"}:
        from pigamagic_agent.agent.providers.image_fal import FalImageProvider

        return FalImageProvider(
            api_key=active.image_api_key,
            model=active.image_model,
            settings=active,
            budget=budget,
        )

    raise ProviderNotConfigured(
        f"Unknown image provider {chosen!r}. Choose one of: "
        f"fixture, nvidia, huggingface, gemini, openai, fal."
    )


def available_image_providers() -> tuple[str, ...]:
    return ("fixture", "nvidia", "huggingface", "gemini", "openai", "fal")


_IMAGE_REGISTRY: dict[str, Callable[..., ImageProvider]] = {
    "fixture": build_image_provider,
    "gemini": build_image_provider,
    "openai": build_image_provider,
    "fal": build_image_provider,
}


def describe_image_providers() -> list[dict[str, Any]]:
    """Capability summary, for the CLI and for choosing a provider before paying."""
    rows: list[dict[str, Any]] = []
    for name in available_image_providers():
        if name == "fixture":
            rows.append(
                {
                    "name": name,
                    "needs_key": False,
                    "identity_refs": True,
                    "max_refs": 4,
                    "price_usd": 0.0,
                    "note": "deterministic synthetic bytes, no network",
                }
            )
            continue
        if name == "huggingface":
            from pigamagic_agent.agent.providers.image_huggingface import DEFAULT_MODEL, resolve_spec

            model = get_settings().image_model or DEFAULT_MODEL
            spec = resolve_spec(model)
            rows.append(
                {
                    "name": name,
                    "needs_key": spec.gated,
                    "identity_refs": spec.identity_refs > 0,
                    "max_refs": spec.identity_refs,
                    "price_usd": 0.0,
                    "note": f"{model} - {spec.note}",
                }
            )
            continue
        try:
            provider = build_image_provider(
                AgentSettings(_env_file=None, image_provider=name, image_api_key="placeholder")
            )
        except Exception:  # pragma: no cover - defensive
            continue
        caps = getattr(provider, "capabilities", None)
        rows.append(
            {
                "name": name,
                "needs_key": True,
                "identity_refs": bool(getattr(caps, "supports_identity_refs", False)),
                "max_refs": int(getattr(caps, "max_identity_refs", 0)),
                "price_usd": float(getattr(provider, "price_per_image_usd", 0.0)),
                "note": getattr(provider, "model", ""),
            }
        )
    return rows


def available_web_providers() -> tuple[str, ...]:
    """Search adapters a host can select with AGENT_WEB_PROVIDER."""
    return ("fixture", "wikipedia", "duckduckgo", "searxng", "serper", "brave")


def build_web_provider(
    settings: AgentSettings | None = None,
    *,
    name: str | None = None,
    budget: BudgetLedger | None = None,
) -> WebProvider:
    """Construct the configured web-research provider.

    `fixture` stays the default so tests, the CLI and CI never touch the network
    by accident. The real adapters differ only in how they search; page
    extraction and image download are shared (providers/web_common.py).
    """
    active = settings or get_settings()
    chosen = (name or active.web_provider or "fixture").strip().casefold()

    if chosen in {"fixture", "fake", "offline"}:
        from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider

        return FixtureWebProvider()

    if chosen in {"wikipedia", "wiki", "mediawiki"}:
        from pigamagic_agent.agent.providers.web_wikipedia import (
            WIKIPEDIA_API,
            WikipediaWebProvider,
        )

        return WikipediaWebProvider(
            api_key=active.web_api_key,
            base_url=active.web_base_url.strip() or WIKIPEDIA_API,
            settings=active,
            budget=budget,
        )

    if chosen in {"searxng", "searx"}:
        from pigamagic_agent.agent.providers.web_searxng import SearxngWebProvider

        return SearxngWebProvider(
            api_key=active.web_api_key,
            base_url=active.web_base_url.strip(),
            settings=active,
            budget=budget,
        )

    if chosen in {"duckduckgo", "ddg", "duck-duck-go"}:
        from pigamagic_agent.agent.providers.web_duckduckgo import DuckDuckGoWebProvider

        return DuckDuckGoWebProvider(settings=active, budget=budget)

    if chosen in {"serper", "serper-dev"}:
        from pigamagic_agent.agent.providers.web_serper import SerperWebProvider

        return SerperWebProvider(
            api_key=active.web_api_key or active.serper_api_key,
            settings=active,
            budget=budget,
        )

    if chosen in {"brave", "brave-search"}:
        from pigamagic_agent.agent.providers.web_brave import BraveWebProvider

        return BraveWebProvider(
            api_key=active.web_api_key,
            settings=active,
            budget=budget,
        )

    raise ProviderNotConfigured(
        f"Unknown web provider {chosen!r}. Choose one of: "
        f"{', '.join(available_web_providers())}."
    )


def describe_web_providers() -> list[dict[str, Any]]:
    """Capability summary, so a host can pick a research backend knowingly."""
    return [
        {
            "name": "fixture",
            "needs_key": False,
            "needs_base_url": False,
            "note": "offline canned corpus, deterministic, free",
        },
        {
            "name": "wikipedia",
            "needs_key": False,
            "needs_base_url": False,
            "note": "keyless public API; best for notable venues (heritage hotels, churches, gardens)",
        },
        {
            "name": "searxng",
            "needs_key": False,
            "needs_base_url": True,
            "note": "keyless open web via your own SearXNG instance (JSON format enabled)",
        },
        {
            "name": "duckduckgo",
            "needs_key": False,
            "needs_base_url": False,
            "note": "keyless open web with no signup; rate-limits datacenter IPs",
        },
        {
            "name": "serper",
            "needs_key": True,
            "needs_base_url": False,
            "note": "Google's index as JSON; ~$0.001 per search, free starter credits",
        },
        {
            "name": "brave",
            "needs_key": True,
            "needs_base_url": False,
            "note": "open web, clean JSON, free monthly quota then ~$5/1000 queries",
        },
    ]


# --------------------------------------------------------------------------
# LLM and vision (the chat-completion providers)
# --------------------------------------------------------------------------


def available_llm_providers() -> tuple[str, ...]:
    """Planner adapters a host can select with AGENT_LLM_PROVIDER."""
    return ("fixture", "groq")


def build_llm_provider(
    settings: AgentSettings | None = None,
    *,
    name: str | None = None,
) -> LLMProvider:
    """Construct the configured planner.

    `fixture` stays the default: a compile step must never be able to spend
    money or reach the network by accident.
    """
    active = settings or get_settings()
    chosen = (name or active.llm_provider or "fixture").strip().casefold()

    if chosen in {"fixture", "fake", "offline"}:
        from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider

        return FixtureLLMProvider()

    if chosen in {"groq", "gpt-oss", "gptoss", "oss"}:
        from pigamagic_agent.agent.providers.llm_groq import (
            DEFAULT_MODEL as GROQ_DEFAULT_MODEL,
        )
        from pigamagic_agent.agent.providers.llm_groq import GROQ_CHAT_URL, GroqLLMProvider

        return GroqLLMProvider(
            api_key=active.llm_api_key or active.groq_api_key,
            base_url=active.llm_base_url.strip() or GROQ_CHAT_URL,
            # A blank setting must keep the adapter's pinned default. Passing ""
            # through overrode it, so every request went out with no model.
            model=active.llm_model.strip() or GROQ_DEFAULT_MODEL,
            settings=active,
        )

    raise ProviderNotConfigured(
        f"Unknown llm provider {chosen!r}. Choose one of: "
        f"{', '.join(available_llm_providers())}."
    )


def available_vision_providers() -> tuple[str, ...]:
    """Vision adapters a host can select with AGENT_VISION_PROVIDER."""
    return ("fixture", "nemotron")


def build_vision_provider(
    settings: AgentSettings | None = None,
    *,
    name: str | None = None,
) -> VisionProvider:
    """Construct the configured vision provider."""
    active = settings or get_settings()
    chosen = (name or active.vision_provider or "fixture").strip().casefold()

    if chosen in {"fixture", "fake", "offline"}:
        from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider

        return FixtureVisionProvider()

    if chosen in {"nemotron", "nvidia", "nvidia-vl", "nim"}:
        from pigamagic_agent.agent.providers.vision_nvidia import (
            DEFAULT_MODEL as NVIDIA_VISION_MODEL,
        )
        from pigamagic_agent.agent.providers.vision_nvidia import (
            NVIDIA_VISION_URL,
            NvidiaVisionProvider,
        )

        return NvidiaVisionProvider(
            api_key=active.vision_api_key or active.nvidia_api_key,
            base_url=active.vision_base_url.strip() or NVIDIA_VISION_URL,
            model=active.vision_model.strip() or NVIDIA_VISION_MODEL,
            settings=active,
        )

    raise ProviderNotConfigured(
        f"Unknown vision provider {chosen!r}. Choose one of: "
        f"{', '.join(available_vision_providers())}."
    )


def describe_chat_providers() -> list[dict[str, Any]]:
    """Capability summary for the planner and vision providers."""
    return [
        {
            "kind": "llm",
            "name": "fixture",
            "needs_key": False,
            "note": "offline deterministic planner",
        },
        {
            "kind": "llm",
            "name": "groq",
            "needs_key": True,
            "note": "openai/gpt-oss-120b on Groq, free tier, json_schema output",
        },
        {
            "kind": "vision",
            "name": "fixture",
            "needs_key": False,
            "note": "offline deterministic appearance analysis",
        },
        {
            "kind": "vision",
            "name": "nemotron",
            "needs_key": True,
            "note": "nvidia/nemotron-nano-12b-v2-vl on NVIDIA NIM, base64 image parts",
        },
    ]


#: Providers that work with no credential at all, so "no key" is not a fault.
KEYLESS_PROVIDERS: frozenset[str] = frozenset({"fixture", "wikipedia", "searxng", "duckduckgo"})


def describe_active_stack(settings: AgentSettings | None = None) -> list[dict[str, Any]]:
    """What each surface will actually use, given the current settings.

    The answer to "is it really Groq + Nemotron + FLUX Klein + DuckDuckGo?"
    without spending a call. Every row reports the *resolved* provider, its
    model, whether the credential it needs is present (never the value), and a
    note where the choice has a cost the operator should know about.

    A row with `usable: False` means the configured provider would raise
    `ProviderNotConfigured` at the first call, so the pipeline would degrade.
    """
    active = settings or get_settings()

    surfaces: list[tuple[str, str, list[str], Callable[[], Any]]] = [
        (
            "llm",
            active.llm_provider,
            ["AGENT_LLM_API_KEY", "AGENT_GROQ_API_KEY"],
            lambda: build_llm_provider(active),
        ),
        (
            "vision",
            active.vision_provider,
            ["AGENT_VISION_API_KEY", "AGENT_NVIDIA_API_KEY"],
            lambda: build_vision_provider(active),
        ),
        (
            "image",
            active.image_provider,
            ["AGENT_IMAGE_API_KEY", "AGENT_NVIDIA_API_KEY"],
            lambda: build_image_provider(active),
        ),
        (
            "web",
            active.web_provider,
            ["AGENT_WEB_API_KEY", "AGENT_SERPER_API_KEY"],
            lambda: build_web_provider(active),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for surface, configured, key_names, factory in surfaces:
        row: dict[str, Any] = {
            "surface": surface,
            "configured": configured,
            "provider": configured,
            "model": "",
            "key_envs": key_names,
            "key_present": False,
            "usable": True,
            "note": "",
        }
        try:
            provider = factory()
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            row.update(usable=False, note=f"cannot be built: {exc}")
            rows.append(row)
            continue

        row["provider"] = getattr(provider, "name", configured)
        row["model"] = str(getattr(provider, "model", "") or "")
        row["key_present"] = bool(_resolved_key(active, surface).strip())
        row["usable"] = bool(row["key_present"] or row["provider"] in KEYLESS_PROVIDERS)
        if not row["usable"] and not row["note"]:
            row["note"] = "needs a key; calls raise ProviderNotConfigured and the pipeline degrades"

        caps = getattr(provider, "capabilities", None)
        if row["usable"] and caps is not None and not getattr(
            caps, "supports_identity_refs", True
        ):
            row["note"] = "text-to-image only: references are dropped and likeness is not kept"
        elif row["usable"] and surface == "web" and row["provider"] == "duckduckgo":
            row["note"] = "keyless; rate-limits datacenter IPs"
        rows.append(row)
    return rows


def _resolved_key(settings: AgentSettings, surface: str) -> str:
    """The credential a surface would use, following the documented aliases."""
    if surface == "llm":
        return settings.llm_api_key or settings.groq_api_key
    if surface == "vision":
        return settings.vision_api_key or settings.nvidia_api_key
    if surface == "image":
        return settings.image_api_key or settings.nvidia_api_key
    return settings.web_api_key or settings.serper_api_key


__all__ = [
    "available_image_providers",
    "available_llm_providers",
    "available_vision_providers",
    "available_web_providers",
    "build_image_provider",
    "build_llm_provider",
    "build_vision_provider",
    "build_web_provider",
    "describe_active_stack",
    "describe_chat_providers",
    "describe_image_providers",
    "describe_web_providers",
]
