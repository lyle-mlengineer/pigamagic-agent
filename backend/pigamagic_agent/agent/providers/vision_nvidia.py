"""NVIDIA NIM `VisionProvider`: read reference photos with Nemotron VL.

Why this model: NVIDIA serves `nvidia/nemotron-nano-12b-v2-vl` (and the larger
Nemotron 3 Nano Omni) on `integrate.api.nvidia.com` with a free tier and the
standard OpenAI chat-completions shape. It is the free route to "actually look
at the reference photos", which is the difference between a shoot that keeps a
face and one that invents a stranger.

Three details come from the live NIM documentation, not from memory:

1. **Images ride in `image_url` as a base64 data URI** —
   `data:image/{format};base64,{...}` — so no public URL is needed for a private
   family photo. This is the only acceptable mechanism: SR-5 forbids uploading
   someone's reference photo to a public bucket just to describe it.
2. **`system` must be a plain string.** The NIM rejects the content-parts array
   on non-user roles, so the appearance policy goes in as text and only the user
   turn carries parts.
3. **There is no `response_format` field on this endpoint.** Structure is asked
   for in the instruction and enforced here by parsing, because a NIM that
   silently ignores an unknown field would leave us trusting unchecked prose.

`name` probe: the model is addressed as `nvidia/nemotron-nano-12b-v2-vl`; any
other vision model on the same endpoint works by setting `AGENT_VISION_MODEL`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from pigamagic_agent.agent.providers.base import CostEstimate, ProviderNotConfigured, VisionResult
from pigamagic_agent.agent.providers.chat_json import clamp_tokens, parse_structured
from pigamagic_agent.agent.providers.http_base import (
    HttpProvider,
    encode_b64,
    sniff_image_type,
)

NVIDIA_VISION_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

#: Nemotron Nano 12B v2 VL — small, free-tier, and good at describing a photo.
DEFAULT_MODEL = "nvidia/nemotron-nano-12b-v2-vl"

#: The shape `analyze.py` reads. Sent as the schema when the caller supplies
#: none, and used to reject a reply that contains no people list at all.
ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_quality": {"type": "string"},
        "people": {"type": "array"},
        "group": {"type": "object"},
        "venue_cues": {"type": "array"},
        "notes": {"type": "string"},
    },
    "required": ["people"],
}

#: Appearance-only policy. The model must never be asked who someone *is*
#: (SR-3), only what is visible.
APPEARANCE_SYSTEM = (
    "You are a photographic analyst. You describe what is visible in an image "
    "in visual, photographic terms: shape, colour, texture, framing, light. "
    "You never infer or state a person's identity, name, ethnicity, religion or "
    "nationality, and you never guess a field you cannot see. "
    "Reply with a single JSON object and nothing else."
)

_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}


@dataclass
class NvidiaVisionProvider(HttpProvider):
    """Reference analysis and post-generation QA through NVIDIA NIM."""

    name: ClassVar[str] = "nemotron"
    base_url: str = NVIDIA_VISION_URL
    model: str = DEFAULT_MODEL

    def require_credentials(self) -> None:
        if not self.api_key.strip():
            raise ProviderNotConfigured(
                "nvidia vision needs a NIM API key. Set AGENT_VISION_API_KEY (or "
                "AGENT_NVIDIA_API_KEY) with a key from build.nvidia.com, or use "
                "the fixture provider."
            )

    async def describe_image(
        self,
        *,
        image: bytes,
        instruction: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> VisionResult:
        if not image:
            raise ValueError("describe_image requires image bytes")

        chosen = self.resolve_model(model)
        active_schema = schema or ANALYSIS_SCHEMA
        content_type = sniff_image_type(image, fallback="")
        if content_type not in _IMAGE_TYPES:
            # Never hand a vendor bytes we could not identify as an image.
            raise ValueError(f"unsupported image type {content_type!r} for vision")

        payload = await self.request_json(
            "POST",
            self.base_url or NVIDIA_VISION_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json_body={
                "model": chosen,
                "messages": [
                    {"role": "system", "content": APPEARANCE_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{content_type};base64,{encode_b64(image)}"
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": clamp_tokens(max_tokens),
            },
            estimate=CostEstimate.zero(),
        )

        data, _ = parse_structured(payload, active_schema, provider=self.name)
        return VisionResult(
            data=data,
            description=_summarise(data),
            provider=self.name,
            model=chosen,
        )

    async def estimate_cost(self, *, image_count: int = 1, model: str | None = None) -> CostEstimate:
        """NVIDIA's hosted NIM tier is free; nothing is charged to the ledger."""
        return CostEstimate(credits=0, usd=0.0)


def _summarise(data: Mapping[str, Any]) -> str:
    """One human-readable line, for logs and prompt assembly.

    Built from the person entries so it stays useful even when the model omits
    a top-level description, and deliberately free of any identity claim.
    """
    people = data.get("people") or []
    labels: list[str] = []
    for person in people:
        if not isinstance(person, Mapping):
            continue
        hair = person.get("hair") or {}
        style = str((hair or {}).get("style") or "").strip() if isinstance(hair, Mapping) else ""
        label = str(person.get("label") or "a person").strip()
        labels.append(f"{label} with {style} hair" if style else label)
    if labels:
        return ", ".join(labels)
    return str(data.get("notes") or "").strip()[:200]


__all__ = [
    "ANALYSIS_SCHEMA",
    "APPEARANCE_SYSTEM",
    "DEFAULT_MODEL",
    "NVIDIA_VISION_URL",
    "NvidiaVisionProvider",
]
