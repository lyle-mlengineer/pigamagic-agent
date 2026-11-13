"""Step 0.2: provider protocols are satisfiable and their value types are sound.

These tests are the contract future adapters (Steps 8, 9, 15) must meet. If a
protocol grows a method, the dummy adapters here fail first, which is the point.
"""

from typing import Any

import pytest

from app.agent.providers import (
    CostEstimate,
    GeneratedImage,
    ImageProvider,
    LLMProvider,
    LLMResult,
    OfflineModeError,
    ProviderError,
    ProviderNotConfigured,
    SearchHit,
    VisionProvider,
    VisionResult,
    WebImage,
    WebPage,
    WebProvider,
)

# --------------------------------------------------------------------------
# Dummy adapters: the smallest thing that satisfies each protocol.
# --------------------------------------------------------------------------


class DummyLLM:
    name = "dummy-llm"

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        return LLMResult(data={}, provider=self.name, model=model or "dummy")

    async def estimate_cost(self, *, user: str, model: str | None = None) -> CostEstimate:
        return CostEstimate(credits=0, usd=0.001)


class DummyVision:
    name = "dummy-vision"

    async def describe_image(
        self,
        *,
        image: bytes,
        instruction: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> VisionResult:
        return VisionResult(data={}, provider=self.name, model=model or "dummy")

    async def estimate_cost(self, *, image_count: int = 1, model: str | None = None) -> CostEstimate:
        return CostEstimate(credits=0, usd=0.001 * image_count)


class DummyImage:
    name = "dummy-image"

    async def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        identity_refs: dict[str, list[bytes]] | None = None,
        structural_ref: bytes | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> GeneratedImage:
        return GeneratedImage(
            content=b"\x89PNG\r\n\x1a\nfake",
            content_type="image/png",
            width=width,
            height=height,
            seed=seed,
            provider=self.name,
        )

    async def estimate_cost(self, *, count: int = 1, model: str | None = None) -> CostEstimate:
        return CostEstimate(credits=count, usd=0.04 * count)


class DummyWeb:
    name = "dummy-web"

    async def search(self, *, query: str, limit: int = 5) -> list[SearchHit]:
        return [SearchHit(url="https://example.test/a", title=query)]

    async def fetch_page(self, *, url: str) -> WebPage:
        return WebPage(url=url, title="Example", text="text")

    async def fetch_image(self, *, url: str) -> WebImage:
        return WebImage(url=url)


# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------


def test_dummy_adapters_satisfy_their_protocols() -> None:
    assert isinstance(DummyLLM(), LLMProvider)
    assert isinstance(DummyVision(), VisionProvider)
    assert isinstance(DummyImage(), ImageProvider)
    assert isinstance(DummyWeb(), WebProvider)


def test_adapters_do_not_satisfy_unrelated_protocols() -> None:
    """Guards against protocols collapsing into one another by accident."""
    assert not isinstance(DummyLLM(), ImageProvider)
    assert not isinstance(DummyImage(), VisionProvider)
    assert not isinstance(DummyWeb(), LLMProvider)


def test_object_missing_a_method_does_not_satisfy_the_protocol() -> None:
    class NotAnImageProvider:
        name = "incomplete"

        async def generate(self, **kwargs: Any) -> GeneratedImage:  # no estimate_cost
            raise NotImplementedError

    assert not isinstance(NotAnImageProvider(), ImageProvider)


# --------------------------------------------------------------------------
# Value types
# --------------------------------------------------------------------------


def test_cost_estimate_adds_and_rejects_negatives() -> None:
    total = CostEstimate(credits=2, usd=0.5) + CostEstimate(credits=1, usd=0.25)
    assert total == CostEstimate(credits=3, usd=0.75)
    assert CostEstimate.zero() == CostEstimate(credits=0, usd=0.0)

    with pytest.raises(ValueError):
        CostEstimate(credits=-1)
    with pytest.raises(ValueError):
        CostEstimate(usd=-0.01)


def test_cost_estimate_is_immutable() -> None:
    cost = CostEstimate(credits=1)
    with pytest.raises(AttributeError):
        cost.credits = 5  # type: ignore[misc]


def test_generated_image_rejects_empty_content_and_bad_type() -> None:
    with pytest.raises(ValueError, match="content"):
        GeneratedImage(content=b"", content_type="image/png", width=10, height=10)

    with pytest.raises(ValueError, match="content_type"):
        GeneratedImage(content=b"x", content_type="image/gif", width=10, height=10)

    with pytest.raises(ValueError, match="dimensions"):
        GeneratedImage(content=b"x", content_type="image/png", width=0, height=10)


def test_generated_image_accepts_allowed_content_types() -> None:
    for content_type in ("image/jpeg", "image/png", "image/webp"):
        image = GeneratedImage(content=b"x", content_type=content_type, width=1, height=1)
        assert image.content_type == content_type


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_provider_error_message_includes_diagnostics() -> None:
    error = ProviderError("rate limited", provider="acme", retryable=True, status=429)
    message = str(error)

    assert "rate limited" in message
    assert "provider=acme" in message
    assert "status=429" in message
    assert "retryable=True" in message
    assert error.retryable is True


def test_offline_mode_error_is_not_retryable() -> None:
    error = OfflineModeError()
    assert error.retryable is False
    assert isinstance(error, ProviderError)


def test_provider_not_configured_is_an_agent_error() -> None:
    from app.agent.providers import AgentError

    assert isinstance(ProviderNotConfigured("no key"), AgentError)
