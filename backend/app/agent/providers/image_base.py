"""Generic image provider with vendor-specific subclasses (design.md §7.4, §8).

`BaseImageProvider` implements everything that is the *same* across vendors:

- prompt composition from `prompt` + `negative_prompt`
- ordering and flattening of `identity_refs` into a stable labelled list
- reference-image budget (how many a vendor accepts, how big)
- turning raw vendor output into a validated `GeneratedImage`
- cost estimation from a published price table

Subclasses implement only `_call_vendor`, which maps our arguments onto that
vendor's request shape and returns raw bytes. That split is the whole point: the
identity-anchoring contract is written once, so it cannot be implemented three
different ways.

**Identity anchoring rule (SR-5, design.md §7.4):** reference images are passed
to the vendor's image-conditioning parameter. They are never described in the
prompt text and never stored as embeddings. The prompt therefore carries scene,
activity and wardrobe — not a face.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from app.agent.config import AgentSettings, get_settings
from app.agent.providers.base import (
    CostEstimate,
    GeneratedImage,
    ProviderError,
    ProviderNotConfigured,
)
from app.agent.providers.http_base import (
    BudgetLedger,
    HttpProvider,
    image_dimensions,
    require_positive,
    sniff_image_type,
)


@dataclass(frozen=True, slots=True)
class ImageCapabilities:
    """What a vendor can actually do. Consulted before spending money.

    `supports_identity_refs` is the blockbuster flag: a model that cannot accept
    reference images cannot preserve a real person's likeness, so the render
    layer must know that before it bills a photo.
    """

    supports_identity_refs: bool = True
    max_identity_refs: int = 1
    supports_negative_prompt: bool = True
    supports_seed: bool = True
    supports_structural_ref: bool = False
    supported_sizes: tuple[tuple[int, int], ...] = ((1024, 1024),)

    def nearest_size(self, width: int, height: int) -> tuple[int, int]:
        """Snap a requested size to one the vendor supports."""
        if (width, height) in self.supported_sizes:
            return width, height
        target = width / height
        return min(self.supported_sizes, key=lambda size: abs((size[0] / size[1]) - target))

    def clamp_refs(self, count: int) -> int:
        return max(0, min(count, self.max_identity_refs))


@dataclass
class BaseImageProvider(HttpProvider):
    """Vendor-neutral image synthesis. Subclasses override `_call_vendor`.

    `name` is inherited as a ClassVar from HttpProvider, so subclasses set it
    with a plain attribute and it cannot be shadowed by a field default.
    """

    # ClassVars, NOT fields: declaring these as annotated fields in subclasses
    # would re-register them as dataclass fields and let the base default
    # silently shadow the vendor's real capabilities and price.
    capabilities: ClassVar[ImageCapabilities] = ImageCapabilities()
    price_per_image_usd: ClassVar[float] = 0.0

    # -- public API ------------------------------------------------------

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
        self.guard_offline()
        self.require_credentials()

        if not prompt or not prompt.strip():
            raise ProviderError("prompt must not be empty", provider=self.name, retryable=False)

        require_positive(width, "width")
        require_positive(height, "height")

        resolved_model = self.resolve_model(model)
        size = self.capabilities.nearest_size(width, height)

        labelled_refs = self._flatten_identity_refs(identity_refs)
        # Guard on the *input*, not the flattened list: max_identity_refs is 0 for
        # providers that cannot condition on images, which would otherwise empty
        # the list and let a real network call slip through with the refs dropped.
        requested_refs = sum(len(items) for items in (identity_refs or {}).values())
        if requested_refs and not self.capabilities.supports_identity_refs:
            raise ProviderError(
                f"{self.name} cannot condition on reference images, so it cannot keep "
                f"this person's likeness. Use a provider with identity support.",
                provider=self.name,
                retryable=False,
            )
        if labelled_refs and not self.capabilities.supports_identity_refs:
            raise ProviderError(
                f"{self.name} cannot condition on reference images, so it cannot keep "
                f"this person's likeness. Use a provider with identity support.",
                provider=self.name,
                retryable=False,
            )

        if structural_ref is not None and not self.capabilities.supports_structural_ref:
            structural_ref = None

        full_prompt = self.compose_prompt(prompt, negative_prompt)
        estimate = self.estimate_for(size, ref_count=len(labelled_refs))

        raw, vendor_meta = await self._call_vendor(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            identity_refs=labelled_refs,
            structural_ref=structural_ref,
            size=size,
            seed=seed if self.capabilities.supports_seed else None,
            model=resolved_model,
            params=dict(params or {}),
            estimate=estimate,
        )

        return self._build_image(
            raw,
            model=resolved_model,
            size=size,
            seed=seed,
            estimate=estimate,
            extra=vendor_meta,
        )

    async def estimate_cost(self, *, count: int = 1, model: str | None = None) -> CostEstimate:
        if count < 0:
            raise ValueError("count must be >= 0")
        usd = self.price_per_image_usd * count
        return CostEstimate(credits=count, usd=round(usd, 6))

    # -- helpers ---------------------------------------------------------

    def compose_prompt(self, prompt: str, negative_prompt: str) -> str:
        """Combine positive and negative text the way this vendor needs.

        Vendors with a dedicated negative-prompt field get the text separately;
        those without get an explicit "avoid" clause appended here. Either way
        the caller passes the same two strings.
        """
        positive = prompt.strip()
        if not negative_prompt.strip() or self.capabilities.supports_negative_prompt:
            return positive
        avoid = ", ".join(item.strip() for item in negative_prompt.split(",") if item.strip())
        if not avoid:
            return positive
        return f"{positive}\n\nAvoid: {avoid}."

    def _flatten_identity_refs(
        self, identity_refs: Mapping[str, Sequence[bytes]] | None
    ) -> list[tuple[str, bytes]]:
        """Flatten `{person_key: [bytes]}` into a deterministic labelled list.

        Sorted by person key so the request order — and therefore the vendor's
        output — is reproducible. Only the first N references per person are
        kept, matching how vendors index references ("the person from image 1").
        """
        if not identity_refs:
            return []

        per_person: list[tuple[str, bytes]] = []
        for person_key in sorted(identity_refs):
            for content in identity_refs[person_key]:
                if content:
                    per_person.append((person_key, content))

        allowed = self.capabilities.clamp_refs(len(per_person))
        return per_person[:allowed]

    def estimate_for(self, size: tuple[int, int], *, ref_count: int = 0) -> CostEstimate:
        usd = self.price_for(size, ref_count=ref_count)
        return CostEstimate(credits=1, usd=round(usd, 6))

    def price_for(self, size: tuple[int, int], *, ref_count: int = 0) -> float:
        """Price one image. Subclasses override for size- or ref-dependent pricing."""
        return self.price_per_image_usd

    def _build_image(
        self,
        raw: bytes,
        *,
        model: str,
        size: tuple[int, int],
        seed: int | None,
        estimate: CostEstimate,
        extra: Mapping[str, Any] | None = None,
    ) -> GeneratedImage:
        if not raw:
            raise ProviderError(
                f"{self.name} returned an empty image.", provider=self.name, retryable=True
            )

        content_type = sniff_image_type(raw)
        width, height = image_dimensions(raw)
        if (width, height) == (0, 0):  # pragma: no cover - defensive
            width, height = size

        return GeneratedImage(
            content=raw,
            content_type=content_type,
            width=width,
            height=height,
            seed=seed,
            provider=self.name,
            model=model,
            cost=estimate,
            metadata=dict(extra or {}),
        )

    async def _call_vendor(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        identity_refs: list[tuple[str, bytes]],
        structural_ref: bytes | None,
        size: tuple[int, int],
        seed: int | None,
        model: str,
        params: dict[str, Any],
        estimate: CostEstimate,
    ) -> tuple[bytes, dict[str, Any]]:
        raise ProviderNotConfigured(f"{self.name} does not implement image generation.")


__all__ = ["BaseImageProvider", "ImageCapabilities"]
