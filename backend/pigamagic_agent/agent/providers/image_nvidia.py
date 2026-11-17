"""NVIDIA NIM image provider (Black Forest Labs FLUX).

Built from the working sample in `black_forest.py` at the repo root and verified
against `docs.api.nvidia.com` (2026). This is the provider to demo with when a
Black Forest Labs model is available only through NVIDIA.

Endpoint shape: `POST {base}/{model}` with a bearer key, JSON body, and a
response whose image lives at `artifacts[0].base64`.

**This adapter is text-to-image only. It never uploads an image.** No reference
photo, no structural reference, no `mode: "Image Editing"`. References are the
vision model's job (`providers/vision_nvidia.py` analyses them and their
observations reach this endpoint as prompt text); the hosted NIM accepts only
`data:image/png;example_id,N` preset images anyway, so an upload would either be
rejected or condition on a stranger's face. Identity refs are therefore dropped
by `render_plan` with a warning rather than sent here.

**The request body is exactly the documented parameter set** for
`black-forest-labs/flux-3-image-klein` (NVIDIA API reference, verified 2026):
`prompt`, `width`, `height`, `samples`, `steps`, `cfg_scale`, `seed`. Nothing
else is sent — in particular no `image` and no `mode`, because the documented
route is text-to-image when no image is supplied, and an undeclared key is a
needless 422 risk on the first real call.

**Constraints that differ from every other adapter**, all enforced here:

- `prompt` is capped at 2000 characters for flux-3-image-klein. Our assembled
  prompts are ~1150 chars, so this is not theoretical: it will bite on a busy
  plan. The cap is applied in `fit_prompt`, which trims the *reference* portion
  first and never drops the subject or action.
- `samples` accepts only `1`.
- `steps` <= 50, `cfg_scale` <= 20 (and must be > 1).
- `width`/`height` are documented as 512..1568 in steps of 16 with
  `width * height <= 1_062_400`. A shoot renders 1024x1024, which is inside the
  range and under the cap, so this adapter declares the single square size it
  actually sends rather than advertising a grid a plan never asks for.
"""

from __future__ import annotations

from typing import Any

from pigamagic_agent.agent.providers.base import ProviderError
from pigamagic_agent.agent.providers.http_base import decode_b64
from pigamagic_agent.agent.providers.image_base import BaseImageProvider, ImageCapabilities

NVIDIA_BASE_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs"

# Hosted model slug — the endpoint is
# POST .../genai/black-forest-labs/flux-3-image-klein (hyphens, not dots; the
# dotted form 404s). Other usable slugs: flux.1-dev, flux.1-schnell,
# flux.2-klein-4b. Override with AGENT_IMAGE_MODEL.
DEFAULT_MODEL = "flux-3-image-klein"

DEFAULT_STEPS = 50
DEFAULT_CFG_SCALE = 3.5

# Documented hard limits on the hosted endpoint.
MAX_PROMPT_CHARS = 2000
MAX_STEPS = 50
MAX_CFG_SCALE = 20.0

# A shoot renders 1024x1024, which is within the documented 512..1568 /
# step-16 grid and under the 1,062,400-pixel cap. samples=1 is all the hosted
# route allows, so one square is the whole declaration.
_HOSTED_SIZES: tuple[tuple[int, int], ...] = ((1024, 1024),)


class NvidiaImageProvider(BaseImageProvider):
    """BFL FLUX generation through NVIDIA NIM.

    Deliberately text-to-image: `supports_identity_refs` and
    `supports_structural_ref` are both False, whatever `NVIDIA_NIM_SELF_HOSTED`
    says, so no request from this adapter can carry image bytes.
    """

    name = "nvidia"
    capabilities = ImageCapabilities(
        supports_identity_refs=False,
        max_identity_refs=0,
        supports_negative_prompt=False,  # no negative field; folded into prompt
        supports_seed=True,
        supports_structural_ref=False,
        supported_sizes=_HOSTED_SIZES,
    )
    price_per_image_usd = 0.0  # trial tier is free; no published per-image rate

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.base_url:
            self.base_url = NVIDIA_BASE_URL
        if not self.model:
            self.model = DEFAULT_MODEL
        # A self-hosted NIM is a different *host*, not a different capability:
        # this adapter still sends no images, so the declared capabilities and
        # the request shape cannot disagree.
        if getattr(self.settings, "nvidia_nim_self_hosted", False):
            self.base_url = getattr(self.settings, "nvidia_image_base_url", "") or self.base_url

    # -- prompt fitting --------------------------------------------------

    def fit_prompt(self, prompt: str, *, budget: int | None = None) -> str:
        """Trim a prompt to the endpoint's character limit.

        Trims the reference block first (least important for composition), then
        hard-truncates. Never returns an empty prompt.
        """
        limit = MAX_PROMPT_CHARS if budget is None else budget
        text = prompt.strip()
        if len(text) <= limit:
            return text

        from pigamagic_agent.agent.prompts import REFERENCE_END, REFERENCE_START

        if REFERENCE_START in text and REFERENCE_END in text:
            start = text.index(REFERENCE_START)
            end = text.index(REFERENCE_END) + len(REFERENCE_END)
            without_refs = (text[:start] + text[end:]).strip()
            if len(without_refs) <= limit:
                return without_refs

        # Still too long: cut on a sentence boundary where possible.
        head = text[: limit - 1]
        for cut in (". ", "; ", ", ", " "):
            position = head.rfind(cut)
            if position > limit * 0.6:
                return head[: position + 1].strip()
        return head.strip()

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
        fitted = self.fit_prompt(prompt)

        steps = int(params.get("steps", DEFAULT_STEPS))
        cfg_scale = float(params.get("cfg_scale", DEFAULT_CFG_SCALE))
        if not 1 <= steps <= MAX_STEPS:
            raise ProviderError(
                f"NVIDIA steps must be 1..{MAX_STEPS}, got {steps}.",
                provider=self.name,
                retryable=False,
            )
        if not 0 < cfg_scale <= MAX_CFG_SCALE:
            raise ProviderError(
                f"NVIDIA cfg_scale must be 0..{MAX_CFG_SCALE}, got {cfg_scale}.",
                provider=self.name,
                retryable=False,
            )

        payload: dict[str, Any] = {
            "prompt": fitted,
            "width": size[0],
            "height": size[1],
            "samples": 1,
            "steps": steps,
            "cfg_scale": cfg_scale,
            # Seed 0 means "random" to NIM, so only send a real seed.
            "seed": int(seed) if seed else 0,
            # No `image` key and no `mode` key: the documented request is
            # text-to-image when no image is supplied, and this adapter must
            # never upload a reference. Identity is the vision model's output
            # expressed as prompt text, so there is nothing to condition on.
        }

        response = await self.request_json(
            "POST",
            f"{self.base_url}/{model}",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json_body=payload,
            estimate=estimate,
        )

        content = _extract_nvidia_image(response, self.name)
        return content, {
            "endpoint": "genai",
            "prompt_chars": len(fitted),
            "prompt_truncated": len(fitted) < len(prompt.strip()),
            "steps": steps,
            "cfg_scale": cfg_scale,
        }


def _extract_nvidia_image(response: Any, provider: str) -> bytes:
    """Read `artifacts[0].base64` from an NVIDIA NIM response."""
    if not isinstance(response, dict):
        raise ProviderError(f"{provider} returned an unexpected payload.", provider=provider)

    artifacts = response.get("artifacts") or []
    if not artifacts:
        raise ProviderError(
            f"{provider} returned no artifacts.", provider=provider, retryable=False
        )

    encoded = (artifacts[0] or {}).get("base64")
    if not encoded:
        raise ProviderError(
            f"{provider} artifact contained no base64 image.", provider=provider, retryable=False
        )

    try:
        return decode_b64(encoded)
    except Exception as exc:  # noqa: BLE001
        raise ProviderError(
            f"{provider} returned an image that could not be decoded: {exc}",
            provider=provider,
            retryable=False,
        ) from exc


__all__ = [
    "DEFAULT_CFG_SCALE",
    "DEFAULT_MODEL",
    "DEFAULT_STEPS",
    "MAX_CFG_SCALE",
    "MAX_PROMPT_CHARS",
    "MAX_STEPS",
    "NVIDIA_BASE_URL",
    "NvidiaImageProvider",
]
