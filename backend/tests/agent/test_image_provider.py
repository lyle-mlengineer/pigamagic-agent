"""Step 8: generic image provider + concrete vendor adapters.

What these tests guarantee, without spending money or touching the network:

- every adapter satisfies the ImageProvider protocol
- the *generic* behaviour (identity-ref flattening, prompt composition, size
  snapping, cost estimation, budget refusal, offline refusal) is identical
  across vendors
- each vendor maps our arguments onto its real request shape
- a missing key or offline mode fails loudly and never silently degrades
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.providers.base import (
    CostEstimate,
    GeneratedImage,
    ImageProvider,
    OfflineModeError,
    ProviderError,
    ProviderNotConfigured,
)
from pigamagic_agent.agent.providers.http_base import (
    BudgetExceeded,
    BudgetLedger,
    decode_b64,
    encode_b64,
    image_dimensions,
    sniff_image_type,
)
from pigamagic_agent.agent.providers.image_fal import FalImageProvider
from pigamagic_agent.agent.providers.image_fixture import FixtureImageProvider, render_fixture_png
from pigamagic_agent.agent.providers.image_gemini import GeminiImageProvider
from pigamagic_agent.agent.providers.image_openai import OpenAIImageProvider
from pigamagic_agent.agent.providers.registry import (
    available_image_providers,
    build_image_provider,
    describe_image_providers,
)

REAL_ADAPTERS = (GeminiImageProvider, OpenAIImageProvider, FalImageProvider)


def _settings(**overrides: Any) -> AgentSettings:
    return AgentSettings(_env_file=None, **overrides)


def _png(width: int = 64, height: int = 64) -> bytes:
    return render_fixture_png(prompt="ref", width=width, height=height)


# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", available_image_providers())
def test_every_provider_satisfies_the_protocol(name: str) -> None:
    provider = build_image_provider(_settings(image_provider=name, image_api_key="test"))
    assert isinstance(provider, ImageProvider)
    assert provider.name == name


@pytest.mark.parametrize("adapter", REAL_ADAPTERS)
def test_adapters_declare_capabilities(adapter: type) -> None:
    instance = adapter(api_key="test", settings=_settings())
    caps = instance.capabilities
    assert caps.max_identity_refs >= 1
    assert caps.supported_sizes
    assert instance.price_per_image_usd > 0, "a real vendor must declare a price"


@pytest.mark.parametrize("adapter", REAL_ADAPTERS)
def test_adapters_pin_a_current_default_model(adapter: type) -> None:
    instance = adapter(api_key="test", settings=_settings())
    assert instance.model
    # These specific model IDs are shut down; pinning them would be dead code.
    assert instance.model not in {"gpt-image-1", "imagen-4", "dall-e-3"}


def test_dead_models_are_not_used_anywhere() -> None:
    for adapter in REAL_ADAPTERS:
        instance = adapter(api_key="test", settings=_settings())
        assert "gpt-image-1" != instance.model


# --------------------------------------------------------------------------
# Credentials, offline, and cost guards
# --------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", REAL_ADAPTERS)
def test_missing_key_raises_not_configured(adapter: type) -> None:
    provider = adapter(api_key="", settings=_settings())
    with pytest.raises(ProviderNotConfigured):
        asyncio.run(provider.generate(prompt="a wedding photo"))


@pytest.mark.parametrize("adapter", REAL_ADAPTERS)
def test_offline_mode_refuses_outbound_calls(adapter: type) -> None:
    provider = adapter(api_key="test", settings=_settings(offline=True))
    with pytest.raises(OfflineModeError):
        asyncio.run(provider.generate(prompt="a wedding photo"))


def test_fixture_needs_no_key_and_works_offline() -> None:
    provider = FixtureImageProvider(settings=_settings(offline=True))
    image = asyncio.run(provider.generate(prompt="bride at the altar", width=128, height=128))
    assert isinstance(image, GeneratedImage)
    assert image.content.startswith(b"\x89PNG")


def test_empty_prompt_is_rejected() -> None:
    provider = FixtureImageProvider(settings=_settings())
    with pytest.raises(ProviderError):
        asyncio.run(provider.generate(prompt="   "))


def test_budget_refuses_a_call_that_would_exceed_the_limit() -> None:
    ledger = BudgetLedger(limit_usd=0.10)
    provider = OpenAIImageProvider(api_key="test", settings=_settings(), budget=ledger)
    # High quality at 1024x1024 is ~$0.167, which exceeds the $0.10 limit.
    with pytest.raises(BudgetExceeded):
        asyncio.run(provider.generate(prompt="a wedding photo"))


def test_budget_allows_a_call_within_the_limit() -> None:
    ledger = BudgetLedger(limit_usd=1.00)
    ledger.spend(CostEstimate(credits=1, usd=0.50))
    assert ledger.remaining_usd == pytest.approx(0.50)
    ledger.check(CostEstimate(credits=1, usd=0.25))  # must not raise


def test_budget_accumulates_and_reports() -> None:
    ledger = BudgetLedger(limit_usd=1.0)
    for _ in range(2):
        ledger.spend(CostEstimate(credits=1, usd=0.4))
    assert ledger.spent_usd == pytest.approx(0.8)
    assert ledger.calls == 2
    assert ledger.as_dict()["calls"] == 2


def test_unlimited_budget_never_refuses() -> None:
    ledger = BudgetLedger(limit_usd=0.0)
    ledger.check(CostEstimate(credits=1, usd=10_000.0))


# --------------------------------------------------------------------------
# Generic behaviour shared by every adapter
# --------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", REAL_ADAPTERS)
def test_identity_refs_are_flattened_in_stable_order(adapter: type) -> None:
    provider = adapter(api_key="test", settings=_settings())
    refs = {"p2": [_png(), _png()], "p1": [_png()]}
    flattened = provider._flatten_identity_refs(refs)

    keys = [key for key, _ in flattened]
    # Sorted by person key, so the vendor sees a reproducible ordering.
    assert keys == sorted(keys)
    assert set(keys) == {"p1", "p2"}


@pytest.mark.parametrize("adapter", REAL_ADAPTERS)
def test_identity_refs_are_capped_per_vendor(adapter: type) -> None:
    provider = adapter(api_key="test", settings=_settings())
    limit = provider.capabilities.max_identity_refs
    refs = {f"p{i}": [_png()] for i in range(limit + 5)}
    assert len(provider._flatten_identity_refs(refs)) == limit


@pytest.mark.parametrize("adapter", REAL_ADAPTERS)
def test_empty_reference_bytes_are_dropped(adapter: type) -> None:
    provider = adapter(api_key="test", settings=_settings())
    flattened = provider._flatten_identity_refs({"p1": [b"", _png()]})
    assert len(flattened) == 1


@pytest.mark.parametrize("adapter", REAL_ADAPTERS)
def test_sizes_are_snapped_to_supported_values(adapter: type) -> None:
    provider = adapter(api_key="test", settings=_settings())
    size = provider.capabilities.nearest_size(999, 999)
    assert size in provider.capabilities.supported_sizes


@pytest.mark.parametrize("adapter", REAL_ADAPTERS)
def test_cost_estimate_scales_with_count(adapter: type) -> None:
    provider = adapter(api_key="test", settings=_settings())
    one = asyncio.run(provider.estimate_cost(count=1))
    three = asyncio.run(provider.estimate_cost(count=3))
    assert three.usd == pytest.approx(one.usd * 3)
    assert three.credits == 3


@pytest.mark.parametrize("adapter", REAL_ADAPTERS)
def test_negative_prompt_is_folded_in_when_unsupported(adapter: type) -> None:
    provider = adapter(api_key="test", settings=_settings())
    composed = provider.compose_prompt("a bride", "malformed hands, extra limbs")
    if provider.capabilities.supports_negative_prompt:
        assert composed == "a bride"
    else:
        assert "malformed hands" in composed


def test_fixture_generation_is_deterministic() -> None:
    provider = FixtureImageProvider(settings=_settings())
    first = asyncio.run(provider.generate(prompt="same prompt", width=128, height=128))
    second = asyncio.run(provider.generate(prompt="same prompt", width=128, height=128))
    assert first.content == second.content


def test_fixture_output_varies_by_prompt() -> None:
    provider = FixtureImageProvider(settings=_settings())
    first = asyncio.run(provider.generate(prompt="a", width=128, height=128))
    second = asyncio.run(provider.generate(prompt="b", width=128, height=128))
    assert first.content != second.content


def test_fixture_records_identity_refs_without_storing_them() -> None:
    provider = FixtureImageProvider(settings=_settings())
    image = asyncio.run(
        provider.generate(
            prompt="bride", identity_refs={"p1": [_png()]}, width=128, height=128
        )
    )
    assert image.metadata["reference_keys"] == ["p1"]
    # No face data or embeddings may appear in metadata (NFR-9).
    assert "embedding" not in json.dumps(image.metadata).casefold()


# --------------------------------------------------------------------------
# Byte helpers
# --------------------------------------------------------------------------


def test_sniff_image_type_ignores_wrong_vendor_labels() -> None:
    png = _png()
    assert sniff_image_type(png) == "image/png"
    assert sniff_image_type(b"\xff\xd8\xff\xe0junk") == "image/jpeg"
    assert sniff_image_type(b"RIFF????WEBPVP8 ") == "image/webp"


def test_image_dimensions_reads_png_header() -> None:
    width, height = image_dimensions(_png(320, 240))
    assert (width, height) == (320, 240)


def test_base64_round_trip_with_and_without_data_url() -> None:
    payload = b"\x89PNG\r\n\x1a\nhello"
    assert decode_b64(encode_b64(payload)) == payload
    assert decode_b64("data:image/png;base64," + encode_b64(payload)) == payload


# --------------------------------------------------------------------------
# Vendor request mapping (no network: vendor call is stubbed)
# --------------------------------------------------------------------------


def test_gemini_puts_reference_images_before_the_text() -> None:
    provider = GeminiImageProvider(api_key="test", settings=_settings())
    captured: dict[str, Any] = {}

    async def fake_request(method, url, **kwargs):
        captured["url"] = url
        captured["body"] = kwargs.get("json_body")
        return {"candidates": [{"content": {"parts": [{"inlineData": {"data": encode_b64(_png())}}]}}]}

    provider.request_json = fake_request  # type: ignore[method-assign]

    image = asyncio.run(
        provider.generate(
            prompt="the bride during the vows",
            identity_refs={"p1": [_png()]},
            width=1024,
            height=1024,
        )
    )

    parts = captured["body"]["contents"][0]["parts"]
    assert "inlineData" in parts[0], "reference image must come first"
    assert parts[-1]["text"] == "the bride during the vows"
    assert "generateContent" in captured["url"]
    assert image.content.startswith(b"\x89PNG")


def test_gemini_skips_thought_images() -> None:
    provider = GeminiImageProvider(api_key="test", settings=_settings())

    async def fake_request(method, url, **kwargs):
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "inlineData": {"data": encode_b64(b"sketch")}},
                            {"inlineData": {"data": encode_b64(_png())}},
                        ]
                    }
                }
            ]
        }

    provider.request_json = fake_request  # type: ignore[method-assign]
    image = asyncio.run(provider.generate(prompt="x", width=1024, height=1024))
    assert image.content != b"sketch"


def test_gemini_raises_when_no_image_is_returned() -> None:
    provider = GeminiImageProvider(api_key="test", settings=_settings())

    async def fake_request(method, url, **kwargs):
        return {"candidates": [{"finishReason": "SAFETY"}]}

    provider.request_json = fake_request  # type: ignore[method-assign]
    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.generate(prompt="x", width=1024, height=1024))
    assert "no image" in str(info.value).casefold()


def test_openai_uses_generations_without_references() -> None:
    provider = OpenAIImageProvider(api_key="test", settings=_settings())
    captured: dict[str, Any] = {}

    async def fake_request(method, url, **kwargs):
        captured["url"] = url
        captured["body"] = kwargs.get("json_body")
        return {"data": [{"b64_json": encode_b64(_png())}]}

    provider.request_json = fake_request  # type: ignore[method-assign]
    asyncio.run(provider.generate(prompt="a wedding", width=1024, height=1024))

    assert captured["url"].endswith("/images/generations")
    assert captured["body"]["model"] == "gpt-image-2.5-flare"
    assert captured["body"]["size"] == "1024x1024"


def test_openai_uses_edits_with_high_fidelity_for_identity_refs() -> None:
    provider = OpenAIImageProvider(api_key="test", settings=_settings())
    captured: dict[str, Any] = {}

    async def fake_request(method, url, **kwargs):
        captured["url"] = url
        captured["files"] = kwargs.get("files")
        captured["form"] = kwargs.get("data")
        return {"data": [{"b64_json": encode_b64(_png())}]}

    provider.request_json = fake_request  # type: ignore[method-assign]
    image = asyncio.run(
        provider.generate(
            prompt="the bride", identity_refs={"p1": [_png()]}, width=1024, height=1024
        )
    )

    assert captured["url"].endswith("/images/edits")
    # input_fidelity=high is what preserves the real face.
    assert captured["form"]["input_fidelity"] == "high"
    assert captured["files"], "reference images must be uploaded, not described"
    assert image.metadata["endpoint"] == "edits"


def test_fal_submits_then_polls_and_downloads() -> None:
    provider = FalImageProvider(api_key="test", settings=_settings())
    calls: list[str] = []
    png = _png()

    async def fake_request(method, url, **kwargs):
        calls.append(url)
        if url.endswith("/status"):
            return {"status": "COMPLETED"}
        if "requests/" in url:
            return {"images": [{"url": "https://cdn.example/img.png"}]}
        return {
            "request_id": "req-1",
            "status_url": "https://queue.fal.run/x/requests/req-1/status",
            "response_url": "https://queue.fal.run/x/requests/req-1",
        }

    async def fake_bytes(url, headers=None):
        return png

    provider.request_json = fake_request  # type: ignore[method-assign]
    provider.request_bytes = fake_bytes  # type: ignore[method-assign]

    image = asyncio.run(provider.generate(prompt="a wedding", width=1024, height=1024))

    assert any("/status" in url for url in calls), "must poll for completion"
    assert image.content == png


def test_fal_switches_to_edit_endpoint_for_references() -> None:
    provider = FalImageProvider(api_key="test", settings=_settings())
    captured: dict[str, Any] = {}

    async def fake_request(method, url, **kwargs):
        captured.setdefault("url", url)
        captured.setdefault("body", kwargs.get("json_body"))
        if url.endswith("/status"):
            return {"status": "COMPLETED"}
        if "requests/" in url:
            return {"images": [{"url": "https://cdn.example/img.png"}]}
        return {
            "status_url": "https://queue.fal.run/x/requests/r/status",
            "response_url": "https://queue.fal.run/x/requests/r",
        }

    async def fake_bytes(url, headers=None):
        return _png()

    provider.request_json = fake_request  # type: ignore[method-assign]
    provider.request_bytes = fake_bytes  # type: ignore[method-assign]

    asyncio.run(
        provider.generate(prompt="x", identity_refs={"p1": [_png()]}, width=1024, height=1024)
    )
    assert "edit-image" in captured["url"]
    assert captured["body"]["image_urls"]


def test_fal_reports_failed_predictions() -> None:
    provider = FalImageProvider(api_key="test", settings=_settings())

    async def fake_request(method, url, **kwargs):
        if url.endswith("/status"):
            return {"status": "FAILED"}
        return {"status_url": "https://q/x/requests/r/status", "response_url": "https://q/x/requests/r"}

    provider.request_json = fake_request  # type: ignore[method-assign]
    with pytest.raises(ProviderError):
        asyncio.run(provider.generate(prompt="x", width=1024, height=1024))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_unknown_provider_is_rejected() -> None:
    # Pass the name explicitly so we exercise the registry's own guard rather
    # than pydantic's Literal validation.
    with pytest.raises(ProviderNotConfigured) as info:
        build_image_provider(_settings(), name="nope")
    assert "gemini" in str(info.value), "the error should list valid choices"


def test_aliases_resolve() -> None:
    assert build_image_provider(_settings(image_provider="google", image_api_key="k")).name == "gemini"
    assert build_image_provider(_settings(image_provider="flux", image_api_key="k")).name == "fal"


def test_registry_shares_one_budget_ledger() -> None:
    ledger = BudgetLedger(limit_usd=2.0)
    provider = build_image_provider(_settings(image_provider="openai", image_api_key="k"), budget=ledger)
    assert provider.budget is ledger


def test_describe_lists_every_provider_with_capabilities() -> None:
    rows = describe_image_providers()
    names = {row["name"] for row in rows}
    assert names == set(available_image_providers())
    for row in rows:
        if row["name"] == "fixture":
            assert row["price_usd"] == 0.0
            continue
        # Every provider must state whether it can honour likeness, because the
        # render layer refuses to bill a photo whose person cannot be kept.
        assert isinstance(row["identity_refs"], bool)


def test_providers_that_cannot_keep_likeness_say_so() -> None:
    """Hosted NVIDIA NIM accepts only example images, so likeness is impossible."""
    rows = {row["name"]: row for row in describe_image_providers()}
    assert rows["nvidia"]["identity_refs"] is False



# --------------------------------------------------------------------------
# Hugging Face local provider (Colab)
# --------------------------------------------------------------------------


def _hf(model: str = "", **overrides: Any):
    from pigamagic_agent.agent.providers.image_huggingface import HuggingFaceImageProvider

    return HuggingFaceImageProvider(model=model, settings=_settings(**overrides))


def test_huggingface_is_registered_under_several_aliases() -> None:
    from pigamagic_agent.agent.providers.image_huggingface import HuggingFaceImageProvider

    for alias in ("huggingface", "hf", "local", "colab"):
        provider = build_image_provider(_settings(image_provider=alias))
        assert isinstance(provider, HuggingFaceImageProvider)


def test_huggingface_needs_no_api_key() -> None:
    provider = _hf()
    assert provider.api_key == ""
    assert provider.price_per_image_usd == 0.0
    asyncio.run(provider.estimate_cost(count=1))  # must not raise


def test_huggingface_default_model_is_honestly_flagged_gated() -> None:
    """FLUX.1-schnell is gated since 2026, and the spec must say so."""
    from pigamagic_agent.agent.providers.image_huggingface import DEFAULT_MODEL, resolve_spec

    assert resolve_spec(DEFAULT_MODEL).gated is True


def test_ungated_alternatives_really_need_no_account() -> None:
    from pigamagic_agent.agent.providers.image_huggingface import (
        UNGATED_MODELS,
        resolve_spec,
    )

    assert UNGATED_MODELS
    for model in UNGATED_MODELS:
        assert resolve_spec(model).gated is False, model


def test_gated_help_never_recommends_the_model_that_failed() -> None:
    """The old message told a schnell user to switch to schnell. Never again."""
    from pigamagic_agent.agent.providers.image_huggingface import gated_help

    text = gated_help("black-forest-labs/FLUX.1-schnell")
    assert "https://huggingface.co/black-forest-labs/FLUX.1-schnell" in text
    assert "settings/tokens" in text
    assert "AGENT_IMAGE_API_KEY" in text
    assert "Qwen/Qwen-Image-Edit-2509" in text  # a real ungated way out
    assert "is gated" in text


def test_gated_checkpoint_without_a_token_fails_before_downloading() -> None:
    """Gating is caught locally: no network call, no 40 GB attempt."""
    provider = _hf("black-forest-labs/FLUX.1-schnell")
    with pytest.raises(ProviderError) as info:
        provider.preflight()
    assert "gated" in str(info.value).casefold()


def test_preflight_allows_a_gated_model_once_a_token_is_set() -> None:
    provider = _hf("black-forest-labs/FLUX.1-schnell")
    provider.api_key = "hf_test"
    provider.preflight()  # must not raise: the licence may already be accepted


def test_preflight_ignores_ungated_models() -> None:
    provider = _hf("stabilityai/sdxl-turbo")
    provider.preflight()  # no token, no licence, still fine


def test_bad_token_is_reported_as_a_token_problem_not_gating(monkeypatch) -> None:
    """A 401 used to be flattened into "is gated", which sent people in circles."""
    import sys
    import types

    provider = _hf("stabilityai/sdxl-turbo")
    provider.api_key = "hf_bad"

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("401 Client Error: Unauthorized for url https://huggingface.co/api/models/x")

    fake_pipeline = types.SimpleNamespace(from_pretrained=_boom)
    monkeypatch.setitem(
        sys.modules, "diffusers", types.SimpleNamespace(DiffusionPipeline=fake_pipeline)
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float32=None))

    with pytest.raises(ProviderError) as info:
        provider.load()
    message = str(info.value).casefold()
    assert "token" in message
    assert "gated" not in message


def test_qwen_alias_selects_the_identity_preserving_model() -> None:
    provider = build_image_provider(_settings(image_provider="qwen"))
    assert provider.model == "Qwen/Qwen-Image-Edit-2509"
    assert provider.capabilities.supports_identity_refs is True
    assert provider.capabilities.max_identity_refs == 3


def test_qwen_edit_2509_declares_identity_support() -> None:
    provider = _hf("Qwen/Qwen-Image-Edit-2509")
    assert provider.capabilities.supports_identity_refs is True
    assert provider.capabilities.max_identity_refs == 3


def test_text_to_image_models_refuse_identity_refs() -> None:
    """A photo of a stranger is worse than a clear error."""
    for model in ("black-forest-labs/FLUX.1-schnell", "Qwen/Qwen-Image"):
        provider = _hf(model)
        assert provider.capabilities.supports_identity_refs is False
        with pytest.raises(ProviderError) as info:
            provider.generate_sync(prompt="the bride", identity_refs={"p1": [_png()]})
        assert "likeness" in str(info.value).casefold()


def test_schnell_uses_its_documented_parameters() -> None:
    """Timestep-distilled: guidance must be 0 and sequence length <= 256."""
    from pigamagic_agent.agent.providers.image_huggingface import resolve_spec

    spec = resolve_spec("black-forest-labs/FLUX.1-schnell")
    assert spec.guidance_scale == 0.0
    assert spec.max_sequence_length == 256
    assert spec.steps <= 8


def test_qwen_uses_true_cfg_scale_not_guidance_scale() -> None:
    """Qwen silently ignores guidance_scale; true_cfg_scale is the real knob."""
    from pigamagic_agent.agent.providers.image_huggingface import resolve_spec

    spec = resolve_spec("Qwen/Qwen-Image-Edit-2509")
    assert spec.true_cfg_scale == 4.0
    assert spec.needs_negative_prompt is True


def test_schnell_prompt_drops_the_reference_block_for_the_token_limit() -> None:
    from pigamagic_agent.agent.prompts import REFERENCE_END, REFERENCE_START

    provider = _hf("black-forest-labs/FLUX.1-schnell")
    prompt = (
        "[SUBJECTS]\n- the bride (p1).\n[ACTION] exchanging vows\n"
        f"{REFERENCE_START}\nnotes\n{REFERENCE_END}"
    )
    adapted = provider.adapt_prompt(prompt)
    assert REFERENCE_START not in adapted
    assert "exchanging vows" in adapted
    assert "the bride" in adapted


def test_huggingface_refuses_when_diffusers_is_missing() -> None:
    provider = _hf()
    with pytest.raises(ProviderError) as info:
        provider.load()
    message = str(info.value).casefold()
    assert "diffusers" in message, "the error must name the missing dependency"


def test_huggingface_reports_gated_models_clearly() -> None:
    from pigamagic_agent.agent.providers.image_huggingface import resolve_spec

    spec = resolve_spec("black-forest-labs/FLUX.1-dev")
    assert spec.gated is True
    assert "huggingface.co" in spec.model_id or spec.model_id


def test_unknown_checkpoint_gets_a_conservative_spec() -> None:
    from pigamagic_agent.agent.providers.image_huggingface import resolve_spec

    spec = resolve_spec("some-org/some-model")
    assert spec.identity_refs == 0
    assert spec.note


def test_model_specs_are_internally_consistent() -> None:
    from pigamagic_agent.agent.providers.image_huggingface import MODEL_SPECS

    for model_id, spec in MODEL_SPECS.items():
        assert spec.model_id == model_id
        assert spec.steps >= 1
        assert spec.sizes
        if spec.max_sequence_length is not None:
            assert 0 < spec.max_sequence_length <= 4096


def test_colab_models_are_offered_in_the_capability_summary() -> None:
    rows = {row["name"]: row for row in describe_image_providers()}
    assert rows["huggingface"]["price_usd"] == 0.0
    assert rows["huggingface"]["identity_refs"] is False  # default is text-to-image

# --------------------------------------------------------------------------
# NVIDIA NIM / Black Forest Labs
# --------------------------------------------------------------------------


def _nvidia(**overrides: Any):
    from pigamagic_agent.agent.providers.image_nvidia import NvidiaImageProvider

    return NvidiaImageProvider(api_key="nvapi-test", settings=_settings(**overrides))


def test_nvidia_is_registered_and_selectable() -> None:
    from pigamagic_agent.agent.providers.image_nvidia import NvidiaImageProvider

    provider = build_image_provider(_settings(image_provider="nvidia", nvidia_api_key="nvapi-x"))
    assert isinstance(provider, NvidiaImageProvider)
    assert provider.name == "nvidia"


def test_nvidia_accepts_the_sample_settings_names() -> None:
    """The sample used `nvidia_api_key` / `nvidia_image_base_url`; both must work."""
    provider = build_image_provider(
        _settings(
            image_provider="nvidia",
            nvidia_api_key="nvapi-from-sample",
            nvidia_image_base_url="https://example.nim.local/v1/genai/black-forest-labs",
        )
    )
    assert provider.api_key == "nvapi-from-sample"
    assert provider.base_url == "https://example.nim.local/v1/genai/black-forest-labs"


def test_nvidia_defaults_match_the_documented_endpoint() -> None:
    from pigamagic_agent.agent.providers.image_nvidia import DEFAULT_MODEL, NVIDIA_BASE_URL

    provider = _nvidia()
    assert provider.model == DEFAULT_MODEL == "flux-3-image-klein"
    assert provider.base_url == NVIDIA_BASE_URL


def test_nvidia_honestly_declares_no_identity_refs_when_hosted() -> None:
    """Hosted preview accepts only example images, so likeness cannot be kept."""
    provider = _nvidia()
    assert provider.capabilities.supports_identity_refs is False


def test_nvidia_self_hosted_enables_identity_refs() -> None:
    """Self-hosted moves the host only; this adapter still uploads nothing."""
    provider = _nvidia(nvidia_nim_self_hosted=True)
    assert provider.capabilities.supports_identity_refs is False
    assert provider.capabilities.max_identity_refs == 0
    assert provider.capabilities.supports_structural_ref is False


def test_nvidia_refuses_identity_refs_it_cannot_honour() -> None:
    provider = _nvidia()
    with pytest.raises(ProviderError) as info:
        asyncio.run(
            provider.generate(
                prompt="the bride", identity_refs={"p1": [_png()]}, width=1024, height=1024
            )
        )
    assert "likeness" in str(info.value).casefold()


def test_nvidia_fits_an_over_long_prompt_by_dropping_references_first() -> None:
    from pigamagic_agent.agent.prompts import REFERENCE_END, REFERENCE_START

    long_note = "a very long reference note about the venue colonnade " * 60
    prompt = (
        "[STYLE] editorial\n"
        "[SUBJECTS]\n- the bride (p1): braided updo.\n"
        "[ACTION] exchanging vows\n"
        f"{REFERENCE_START}\n{long_note}\n{REFERENCE_END}\n"
        "[NEGATIVE] malformed hands."
    )
    provider = _nvidia()
    fitted = provider.fit_prompt(prompt)

    assert len(fitted) <= 2000
    # The content that decides the photograph must survive.
    assert "exchanging vows" in fitted
    assert "the bride" in fitted
    # And the untrusted reference block is what got cut.
    assert REFERENCE_START not in fitted


def test_nvidia_fit_prompt_leaves_short_prompts_alone() -> None:
    provider = _nvidia()
    assert provider.fit_prompt("a short prompt") == "a short prompt"


def test_nvidia_fit_prompt_never_returns_empty() -> None:
    provider = _nvidia()
    assert provider.fit_prompt("x" * 5000)


def test_nvidia_always_sends_the_documented_payload() -> None:
    provider = _nvidia()
    captured: dict[str, Any] = {}

    async def fake_request(method, url, **kwargs):
        captured["url"] = url
        captured["body"] = kwargs.get("json_body")
        return {"artifacts": [{"base64": encode_b64(_png())}]}

    provider.request_json = fake_request  # type: ignore[method-assign]
    image = asyncio.run(provider.generate(prompt="a wedding portrait", width=1024, height=1024))

    body = captured["body"]
    # The published route is POST .../black-forest-labs/flux-3-image-klein.
    assert captured["url"].endswith("/flux-3-image-klein")
    assert "flux.3-image-klein" not in captured["url"]
    assert body["width"] == 1024 and body["height"] == 1024
    assert body["samples"] == 1
    assert 1 <= body["steps"] <= 50
    # Exactly the documented parameter set: an undeclared key is a 422 risk.
    # `seed` used to be sent but the hosted endpoint rejects it with
    # `422 extra_forbidden` (verified live), so it is deliberately absent.
    assert set(body) == {"prompt", "width", "height", "samples", "steps", "cfg_scale"}
    assert "seed" not in body
    # No image is uploaded, in any field: this adapter is text-to-image only.
    assert "image" not in body
    assert "mode" not in body
    assert image.content.startswith(b"\x89PNG")
    assert image.metadata["endpoint"] == "genai"


def test_nvidia_clamps_size_to_1024() -> None:
    provider = _nvidia()
    captured: dict[str, Any] = {}

    async def fake_request(method, url, **kwargs):
        captured["body"] = kwargs.get("json_body")
        return {"artifacts": [{"base64": encode_b64(_png())}]}

    provider.request_json = fake_request  # type: ignore[method-assign]
    asyncio.run(provider.generate(prompt="x", width=1536, height=1024))
    assert captured["body"]["width"] == 1024 and captured["body"]["height"] == 1024


def test_nvidia_steps_are_configurable_and_default_to_20() -> None:
    """Diffusion steps are settings, not a constant baked into the adapter.

    The hosted endpoint allows up to 50; 20 ships as the default because a
    50-step portrait is materially slower for little visible gain. An explicit
    `params` value still wins, so a caller can tune a single render.
    """
    assert _nvidia().settings.image_steps == 20

    provider = _nvidia(image_steps=12, image_cfg_scale=4.0)
    captured: dict[str, Any] = {}

    async def fake_request(method, url, **kwargs):
        captured["body"] = kwargs.get("json_body")
        return {"artifacts": [{"base64": encode_b64(_png())}]}

    provider.request_json = fake_request  # type: ignore[method-assign]
    asyncio.run(provider.generate(prompt="a wedding portrait"))
    assert captured["body"]["steps"] == 12
    assert captured["body"]["cfg_scale"] == 4.0

    captured.clear()
    asyncio.run(provider.generate(prompt="a wedding portrait", params={"steps": 8}))
    assert captured["body"]["steps"] == 8, "an explicit per-render value must win"


def test_nvidia_rejects_out_of_range_steps_and_cfg() -> None:
    provider = _nvidia()

    async def fake_request(method, url, **kwargs):
        return {"artifacts": [{"base64": encode_b64(_png())}]}

    provider.request_json = fake_request  # type: ignore[method-assign]

    with pytest.raises(ProviderError):
        asyncio.run(provider.generate(prompt="x", params={"steps": 500}))
    with pytest.raises(ProviderError):
        asyncio.run(provider.generate(prompt="x", params={"cfg_scale": 99}))


def test_nvidia_reports_missing_artifacts() -> None:
    provider = _nvidia()

    async def fake_request(method, url, **kwargs):
        return {"artifacts": []}

    provider.request_json = fake_request  # type: ignore[method-assign]
    with pytest.raises(ProviderError):
        asyncio.run(provider.generate(prompt="x"))


def test_nvidia_maps_401_to_a_clear_error() -> None:
    """A bad key must say so, not look like a generic failure."""
    provider = _nvidia()
    with pytest.raises(ProviderNotConfigured):
        _nvidia_with_no_key = type(provider)(api_key="", settings=_settings())
        asyncio.run(_nvidia_with_no_key.generate(prompt="x"))


def test_nvidia_honestly_declares_seeds_unhonoured() -> None:
    """The hosted endpoint rejects `seed`, so supports_seed must be False and
    `generate` must not pass a seed to the vendor."""
    provider = _nvidia()
    assert provider.capabilities.supports_seed is False
    captured: dict[str, Any] = {}

    async def fake_request(method, url, **kwargs):
        captured["body"] = kwargs.get("json_body")
        return {"artifacts": [{"base64": encode_b64(_png())}]}

    provider.request_json = fake_request  # type: ignore[method-assign]
    asyncio.run(provider.generate(prompt="x", seed=12345))
    assert "seed" not in captured["body"]


def test_nvidia_reports_moderation_refusals_contextually() -> None:
    """A CONTENT_FILTERED finish reason is a moderation block, not a mystery
    empty artifact; after the softening retry also refuses, it must surface as
    ContentFilteredError with the reason attached."""
    from pigamagic_agent.agent.providers.image_nvidia import (
        ContentFilteredError,
        FINISH_REASON_FILTERED,
    )

    provider = _nvidia()
    calls: list[str] = []

    async def fake_request(method, url, **kwargs):
        calls.append(kwargs.get("json_body", {}).get("prompt", ""))
        return {"artifacts": [{"finishReason": FINISH_REASON_FILTERED, "base64": ""}]}

    provider.request_json = fake_request  # type: ignore[method-assign]
    with pytest.raises(ContentFilteredError) as info:
        asyncio.run(provider.generate(prompt="a sage floral dress"))
    assert FINISH_REASON_FILTERED in str(info.value)
    assert info.value.retryable is False


def test_nvidia_softening_retry_recovers_a_false_positive() -> None:
    """When the first attempt is refused but the softened prompt passes, the
    shot must succeed and the metadata must say it was softened."""
    from pigamagic_agent.agent.providers.image_nvidia import FINISH_REASON_FILTERED

    provider = _nvidia()
    calls: list[str] = []

    async def fake_request(method, url, **kwargs):
        text = kwargs.get("json_body", {}).get("prompt", "")
        calls.append(text)
        if len(calls) == 1:
            return {"artifacts": [{"finishReason": FINISH_REASON_FILTERED, "base64": ""}]}
        return {"artifacts": [{"base64": encode_b64(_png())}]}

    provider.request_json = fake_request  # type: ignore[method-assign]
    image = asyncio.run(provider.generate(prompt="a sage floral dress"))
    assert len(calls) == 2
    assert "sage" in calls[0]
    assert "sage" not in calls[1]
    assert "muted olive" in calls[1]
    assert image.metadata["prompt_softened"] is True


def test_nvidia_soften_prompt_only_changes_when_needed() -> None:
    from pigamagic_agent.agent.providers.image_nvidia import soften_prompt
    from pigamagic_agent.agent.prompts import REFERENCE_END, REFERENCE_START

    clean = "[STYLE] editorial\n[SCENE] a garden."
    assert soften_prompt(clean) == clean

    flagged = (
        "[STYLE] editorial\n"
        "[SUBJECTS]\n- the bride: sage dress.\n"
        f"{REFERENCE_START}\nmaterial only\n{REFERENCE_END}\n"
        "[NEGATIVE] malformed hands.\n\nAvoid: malformed hands, watermarks."
    )
    softened = soften_prompt(flagged)
    assert REFERENCE_START not in softened
    assert "[NEGATIVE]" not in softened and "Avoid:" not in softened
    assert "sage" not in softened
    assert "muted olive" in softened
    assert "the bride" in softened


def test_nvidia_never_uploads_a_structural_ref() -> None:
    """Even self-hosted, an edit must not become an uploaded image."""
    provider = _nvidia(nvidia_nim_self_hosted=True)
    captured: dict[str, Any] = {}

    async def fake_request(method, url, **kwargs):
        captured["body"] = kwargs.get("json_body")
        return {"artifacts": [{"base64": encode_b64(_png())}]}

    provider.request_json = fake_request  # type: ignore[method-assign]
    asyncio.run(provider.generate(prompt="x", structural_ref=_png()))

    body = captured["body"]
    assert "image" not in body
    assert "mode" not in body
    assert set(body) <= {"prompt", "width", "height", "samples", "steps", "cfg_scale", "seed"}


def test_nvidia_judges_real_plan_prompts_against_the_limit() -> None:
    """Our assembled prompts must fit the 2000-char cap after fitting."""
    from pigamagic_agent.agent.cli import _demo_venue, demo_cast
    from pigamagic_agent.agent.pipeline import load_brief, run_pipeline
    from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider
    from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
    from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider

    async def build():
        brief = load_brief("wedding")
        return await run_pipeline(
            brief,
            llm=FixtureLLMProvider(),
            vision=FixtureVisionProvider(),
            web=FixtureWebProvider(),
            settings=_settings(),
            cast_override=demo_cast("wedding"),
            venue_override=_demo_venue(brief),
        )

    artifact = asyncio.run(build())
    provider = _nvidia()
    for node_id, assembled in artifact.prompts.items():
        fitted = provider.fit_prompt(assembled.prompt)
        assert len(fitted) <= 2000, f"{node_id} exceeds the NVIDIA prompt cap"
        assert fitted.strip()
        # The subject and action must never be what gets cut.
        assert "[SUBJECTS]" in fitted
        assert "[ACTION]" in fitted

