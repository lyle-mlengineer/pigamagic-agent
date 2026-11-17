"""Provider protocols and their shared data types.

Every capability the pipeline needs from the outside world is expressed here as
a narrow protocol. The pipeline depends on *these*, never on a vendor SDK.

Why protocols rather than base classes:
- Adapters stay plain classes with no inheritance requirement.
- Fixtures for Steps 1-7 satisfy the same contract as real vendors, so pipeline
  code is exercised for real from day one.
- `@runtime_checkable` lets tests assert conformance with isinstance().

Rules for implementers (from design.md §8.4):
- Map vendor types to the dataclasses below; never leak SDK objects.
- Own retries and timeouts inside the adapter. The pipeline sees success or a
  typed AgentError, never a raw transport exception.
- Report cost per call so the run layer can settle credits accurately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class AgentError(Exception):
    """Base class for every failure the pipeline is expected to handle."""


class ProviderError(AgentError):
    """A provider call failed.

    `retryable` distinguishes "try again" (timeout, 429, 5xx) from
    "this will never work" (bad request, content policy rejection). The render
    layer retries only when this is True, at most twice (design.md §16).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        retryable: bool = False,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status = status

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.status is not None:
            parts.append(f"status={self.status}")
        parts.append(f"retryable={self.retryable}")
        return " ".join(parts)


class ProviderNotConfigured(ProviderError):
    """The selected provider is missing credentials or is unimplemented.

    Deliberately a `ProviderError` with `retryable=False`, not a bare
    `AgentError`. Every degrade-gracefully ladder in the pipeline catches
    `ProviderError` (compile falls back to a procedural plan, research admits it
    found nothing, render records a failed shot). While this lived outside that
    hierarchy, a keyless real provider raised straight through those ladders and
    killed the process with a traceback — the exact opposite of the design's
    "the caller always gets a usable plan plus warnings".
    """

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message, provider=provider, retryable=False)


class OfflineModeError(ProviderError):
    """A network call was attempted while settings.offline is True."""

    def __init__(self, message: str = "Outbound HTTP is disabled (offline mode).") -> None:
        super().__init__(message, retryable=False)


# --------------------------------------------------------------------------
# Shared value types
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """What a call is expected to cost, in the cheapest honest units.

    `credits` is the user-facing unit. `usd` exists for internal dashboards and
    cost reconciliation; it is never shown to users.
    """

    credits: int = 0
    usd: float = 0.0

    def __post_init__(self) -> None:
        if self.credits < 0:
            raise ValueError("credits must be >= 0")
        if self.usd < 0:
            raise ValueError("usd must be >= 0")

    def __add__(self, other: CostEstimate) -> CostEstimate:
        return CostEstimate(credits=self.credits + other.credits, usd=self.usd + other.usd)

    @classmethod
    def zero(cls) -> CostEstimate:
        return cls(credits=0, usd=0.0)


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """Raw image bytes plus what we need to persist and audit them.

    Deliberately has no `path`: writing to storage is the blob store's job
    (see the shell's `BlobStore`), so providers stay unaware of our filesystem.
    """

    content: bytes
    content_type: str
    width: int
    height: int
    seed: int | None = None
    provider: str = ""
    model: str = ""
    cost: CostEstimate = field(default_factory=CostEstimate.zero)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("GeneratedImage.content must not be empty")
        if self.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"unsupported content_type: {self.content_type!r}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("GeneratedImage dimensions must be positive")


@dataclass(frozen=True, slots=True)
class LLMResult:
    """A structured-output LLM response.

    `data` is the parsed JSON object. Adapters must validate it with the
    caller's schema before returning; the pipeline never parses free text.
    """

    data: dict[str, Any]
    raw_text: str = ""
    provider: str = ""
    model: str = ""
    cost: CostEstimate = field(default_factory=CostEstimate.zero)
    latency_ms: int = 0


@dataclass(frozen=True, slots=True)
class VisionResult:
    """A structured description of one image.

    `description` is free text for humans and prompt assembly; `data` is the
    validated structured payload the caller asked for.
    """

    data: dict[str, Any]
    description: str = ""
    provider: str = ""
    model: str = ""
    cost: CostEstimate = field(default_factory=CostEstimate.zero)
    latency_ms: int = 0


@dataclass(frozen=True, slots=True)
class WebPage:
    """One extracted web page used as venue evidence.

    `images` carries absolute URLs found in the page's markup. It is part of the
    contract because the research stage must fetch *real* image candidates: for
    a long time this field did not exist and the pipeline synthesised
    `{page_url}/photo-N.png` instead, which 404s against every real website.
    """

    url: str
    title: str
    text: str
    fetched_at: str = ""
    images: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WebImage:
    """An image candidate found during research."""

    url: str
    page_url: str = ""
    alt_text: str = ""
    content: bytes = b""
    content_type: str = ""


@dataclass(frozen=True, slots=True)
class SearchHit:
    url: str
    title: str = ""
    snippet: str = ""


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Text reasoning: plan compilation, edit-instruction compilation.

    Implementations must request structured JSON output and raise
    ProviderError if the response cannot be parsed into `schema`.
    """

    name: str

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult: ...

    async def estimate_cost(self, *, user: str, model: str | None = None) -> CostEstimate: ...


@runtime_checkable
class VisionProvider(Protocol):
    """Image understanding: reference analysis and post-generation QA.

    `schema` is a JSON Schema the response must satisfy. Implementations must
    never assert a real identity from a face; they describe appearance only
    (design.md §5.3, SR-3).
    """

    name: str

    async def describe_image(
        self,
        *,
        image: bytes,
        instruction: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> VisionResult: ...

    async def estimate_cost(self, *, image_count: int = 1, model: str | None = None) -> CostEstimate: ...


@runtime_checkable
class ImageProvider(Protocol):
    """Image synthesis and image-conditioned edits.

    `identity_refs` maps a person key to reference image bytes. Adapters must
    pass these to the model's identity/IP mechanism, not merely describe them
    in the prompt (design.md §7.3).
    """

    name: str

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
    ) -> GeneratedImage: ...

    async def estimate_cost(self, *, count: int = 1, model: str | None = None) -> CostEstimate: ...


@runtime_checkable
class WebProvider(Protocol):
    """Web research: search, page extraction, image fetch.

    Callers enforce the budget (settings.max_search_calls); adapters do not.
    Every returned item keeps its URL so provenance survives to the plan.
    """

    name: str

    async def search(self, *, query: str, limit: int = 5) -> list[SearchHit]: ...

    async def fetch_page(self, *, url: str) -> WebPage: ...

    async def fetch_image(self, *, url: str) -> WebImage: ...


__all__ = [
    "AgentError",
    "CostEstimate",
    "GeneratedImage",
    "LLMProvider",
    "LLMResult",
    "OfflineModeError",
    "ProviderError",
    "ProviderNotConfigured",
    "SearchHit",
    "VisionProvider",
    "VisionResult",
    "WebImage",
    "WebPage",
    "WebProvider",
    "ImageProvider",
]
