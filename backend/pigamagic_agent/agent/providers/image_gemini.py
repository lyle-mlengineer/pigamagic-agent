"""Google Gemini image provider (Nano Banana family).

Verified against the current Gemini API docs (2026): image generation and editing
go through `models/{model}:generateContent` with `responseModalities` including
`IMAGE`. Reference images are passed as `inlineData` parts **before** the text
instruction, which is how the model is documented to associate subjects.

Model IDs are pinned in `DEFAULT_MODEL` rather than left blank, because the
family churns: `gemini-2.5-flash-image` is already legacy, and Imagen 4 was
shut down in Aug 2026. Override with `AGENT_IMAGE_MODEL`.

Pricing (per image, 1K): Nano Banana 2.1 ~$0.067, Lite ~$0.0336, Pro ~$0.09.
These are defaults for estimation only; the ledger is the authority.
"""

from __future__ import annotations

from typing import Any, Sequence

from pigamagic_agent.agent.providers.base import ProviderError
from pigamagic_agent.agent.providers.http_base import decode_b64, encode_b64
from pigamagic_agent.agent.providers.image_base import BaseImageProvider, ImageCapabilities

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Current Gemini image models, cheapest-capable first.
DEFAULT_MODEL = "gemini-3.1-flash-image"

# Most per-image families only accept a fixed set of ratios; these are the
# documented 1K sizes.
_SUPPORTED_SIZES: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1344, 768),   # 16:9
    (768, 1344),   # 9:16
    (1184, 888),   # 4:3
    (888, 1184),   # 3:4
)

_ASPECT_LABELS: dict[tuple[int, int], str] = {
    (1024, 1024): "1:1",
    (1344, 768): "16:9",
    (768, 1344): "9:16",
    (1184, 888): "4:3",
    (888, 1184): "3:4",
}


class GeminiImageProvider(BaseImageProvider):
    """Gemini native image generation and reference-conditioned editing."""

    name: str = "gemini"
    capabilities = ImageCapabilities(
        supports_identity_refs=True,
        max_identity_refs=14,  # documented limit for Nano Banana Pro
        supports_negative_prompt=False,  # no negative field; folded into prompt
        supports_seed=False,  # Gemini image generation does not honour seeds
        supports_structural_ref=True,
        supported_sizes=_SUPPORTED_SIZES,
    )
    price_per_image_usd = 0.067

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.base_url:
            self.base_url = GEMINI_BASE_URL
        if not self.model:
            self.model = DEFAULT_MODEL

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
        model_id = model if model.startswith("models/") else f"models/{model}"
        url = f"{self.base_url}/{model_id}:generateContent"

        parts: list[dict[str, Any]] = []

        # Reference images first, in stable order, each labelled by position so
        # the prompt can say "the person in image 1".
        ref_index = 0
        for person_key, content in identity_refs:
            ref_index += 1
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": encode_b64(content),
                    }
                }
            )
            # A label attached to the image helps the model associate a person
            # with their reference without describing the face in the prompt.
            parts.append({"text": f"(reference image {ref_index}: {person_key})"})

        if structural_ref is not None:
            parts.append(
                {"inlineData": {"mimeType": "image/png", "data": encode_b64(structural_ref)}}
            )

        parts.append({"text": prompt})

        generation_config: dict[str, Any] = {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
                "aspectRatio": _ASPECT_LABELS.get(size, "1:1"),
                "imageSize": "1K",
            },
        }
        generation_config.update(params.get("generation_config") or {})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }

        response = await self.request_json(
            "POST",
            url,
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            json_body=payload,
            estimate=estimate,
        )

        image_bytes, meta = _extract_gemini_image(response, self.name)
        return image_bytes, meta

    def price_for(self, size: tuple[int, int], *, ref_count: int = 0) -> float:
        # 2K and 4K cost more; we request 1K, so a flat rate is honest here.
        return self.price_per_image_usd


def _extract_gemini_image(response: Any, provider: str) -> tuple[bytes, dict[str, Any]]:
    """Pull the last non-thought inline image out of a generateContent response.

    Gemini can return interim "thought" images alongside the final one, so the
    filter must skip `thought: true` parts rather than taking the first image.
    """
    if not isinstance(response, dict):
        raise ProviderError(f"{provider} returned an unexpected payload.", provider=provider)

    candidates = response.get("candidates") or []
    for candidate in candidates:
        content = (candidate or {}).get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if not inline or part.get("thought"):
                continue
            data = inline.get("data")
            if data:
                return decode_b64(data), {
                    "mime_type": inline.get("mimeType") or inline.get("mime_type", ""),
                    "finish_reason": (candidate or {}).get("finishReason", ""),
                }

    # No image: surface why, rather than returning empty bytes.
    blocked = response.get("promptFeedback") or {}
    reasons = [str((c or {}).get("finishReason", "")) for c in candidates]
    detail = blocked.get("blockReason") or ", ".join(r for r in reasons if r) or "no image returned"
    raise ProviderError(
        f"{provider} returned no image ({detail}).",
        provider=provider,
        retryable=False,
    )


__all__ = ["DEFAULT_MODEL", "GEMINI_BASE_URL", "GeminiImageProvider"]
