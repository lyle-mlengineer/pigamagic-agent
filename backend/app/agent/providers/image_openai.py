"""OpenAI GPT Image provider.

Verified against current OpenAI docs (2026):

- Text-to-image: `POST /v1/images/generations` (JSON).
- Reference-conditioned: `POST /v1/images/edits` (multipart) with up to 16
  images; always used here when identity refs exist, because that is the only
  endpoint that accepts input images.
- `input_fidelity: "high"` is the documented lever for matching facial
  features, so it is always sent when references are present. It is the
  difference between "a person" and "this person".

`gpt-image-1` was shut down 2026-10-23. Pinning it (as older tutorials do) would
produce dead code, so the default is `gpt-image-2.5-flare`.

Pricing is token-based; per-image figures below are OpenAI's published
1024x1024 rates for the equivalent quality tier, used only for pre-flight
estimation and the budget guard.
"""

from __future__ import annotations

import json
from typing import Any

from app.agent.providers.base import ProviderError
from app.agent.providers.http_base import decode_b64
from app.agent.providers.image_base import BaseImageProvider, ImageCapabilities

OPENAI_BASE_URL = "https://api.openai.com/v1"

DEFAULT_MODEL = "gpt-image-2.5-flare"

_SUPPORTED_SIZES: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1536, 1024),
    (1024, 1536),
)

# Published per-image prices at 1024x1024 by quality tier. Only used to estimate
# before a call; the vendor's `usage` field is authoritative afterwards.
_PRICE_BY_QUALITY: dict[str, float] = {
    "low": 0.011,
    "medium": 0.042,
    "high": 0.167,
    "auto": 0.042,
}

_MAX_REFS = 16  # documented limit for the edits endpoint


class OpenAIImageProvider(BaseImageProvider):
    """GPT Image generation and high-fidelity reference-conditioned editing."""

    name: str = "openai"
    capabilities = ImageCapabilities(
        supports_identity_refs=True,
        max_identity_refs=_MAX_REFS,
        supports_negative_prompt=False,  # folded into the prompt text
        supports_seed=False,  # not exposed by the Image API
        supports_structural_ref=True,  # an extra image in /edits
        supported_sizes=_SUPPORTED_SIZES,
    )
    price_per_image_usd = _PRICE_BY_QUALITY["high"]
    quality: str = "high"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.base_url:
            self.base_url = OPENAI_BASE_URL
        if not self.model:
            self.model = DEFAULT_MODEL

    def price_for(self, size: tuple[int, int], *, ref_count: int = 0) -> float:
        base = _PRICE_BY_QUALITY.get(self.quality, _PRICE_BY_QUALITY["high"])
        # Non-square sizes are priced higher by OpenAI (0.25 vs 0.167 at high).
        if size != (1024, 1024):
            base *= 1.5
        # High input fidelity adds ~$0.041 per reference image.
        if ref_count and self.quality in {"high", "medium"}:
            base += 0.041 * ref_count
        return round(base, 4)

    # -- vendor call -----------------------------------------------------

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
        estimate: Any,
    ) -> tuple[bytes, dict[str, Any]]:
        size_label = f"{size[0]}x{size[1]}"
        quality = str(params.get("quality") or self.quality)

        if identity_refs or structural_ref is not None:
            return await self._edit(
                prompt=prompt,
                identity_refs=identity_refs,
                structural_ref=structural_ref,
                size_label=size_label,
                quality=quality,
                model=model,
                estimate=estimate,
            )
        return await self._generate(
            prompt=prompt,
            size_label=size_label,
            quality=quality,
            model=model,
            estimate=estimate,
        )

    async def _generate(
        self,
        *,
        prompt: str,
        size_label: str,
        quality: str,
        model: str,
        estimate: Any,
    ) -> tuple[bytes, dict[str, Any]]:
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size_label,
            "quality": quality,
            "n": 1,
            "output_format": "png",
        }
        response = await self.request_json(
            "POST",
            f"{self.base_url}/images/generations",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json_body=payload,
            estimate=estimate,
        )
        return _extract_image(response, self.name, model), {"endpoint": "generations"}

    async def _edit(
        self,
        *,
        prompt: str,
        identity_refs: list[tuple[str, bytes]],
        structural_ref: bytes | None,
        size_label: str,
        quality: str,
        model: str,
        estimate: Any,
    ) -> tuple[bytes, dict[str, Any]]:
        # /images/edits is multipart: repeated `image[]` fields, one per input.
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, (person_key, content) in enumerate(identity_refs):
            files.append(
                ("image[]", (f"{person_key}-{index}.png", content, "image/png"))
            )
        if structural_ref is not None:
            files.append(("image[]", ("structure.png", structural_ref, "image/png")))

        form = {
            "model": model,
            "prompt": prompt,
            "size": size_label,
            "quality": quality,
            "n": "1",
            "output_format": "png",
            # The documented lever for preserving facial features from inputs.
            "input_fidelity": "high",
        }

        response = await self.request_json(
            "POST",
            f"{self.base_url}/images/edits",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files=files,
            data=form,
            estimate=estimate,
        )
        return _extract_image(response, self.name, model), {
            "endpoint": "edits",
            "input_fidelity": "high",
            "reference_count": len(identity_refs),
        }


def _extract_image(response: Any, provider: str, model: str) -> bytes:
    """Read `data[0].b64_json` (or a URL, if the vendor returns one)."""
    if not isinstance(response, dict):
        raise ProviderError(f"{provider} returned an unexpected payload.", provider=provider)

    items = response.get("data") or []
    if not items:
        raise ProviderError(
            f"{provider} returned no image data.",
            provider=provider,
            retryable=False,
        )

    first = items[0] or {}
    encoded = first.get("b64_json") or first.get("b64") or ""
    if encoded:
        return decode_b64(encoded)

    url = first.get("url")
    if url:
        raise ProviderError(
            f"{provider} returned a URL instead of inline data; configure b64 output.",
            provider=provider,
            retryable=False,
        )

    raise ProviderError(f"{provider} response contained no usable image.", provider=provider)


__all__ = ["DEFAULT_MODEL", "OPENAI_BASE_URL", "OpenAIImageProvider"]
