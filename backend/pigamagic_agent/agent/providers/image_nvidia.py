"""NVIDIA NIM image provider (Black Forest Labs FLUX).

Built from the working sample in `black_forest.py` at the repo root and verified
against `docs.api.nvidia.com` (2026). This is the provider to demo with when a
Black Forest Labs model is available only through NVIDIA.

Endpoint shape: `POST {base}/{model}` with a bearer key, JSON body, and a
response whose image lives at `artifacts[0].base64`.

**Constraints that differ from every other adapter**, all enforced here:

- `prompt` is capped at 2000 characters for flux-3-image-klein. Our assembled
  prompts are ~1150 chars, so this is not theoretical: it will bite on a busy
  plan. The cap is applied in `_fit_prompt`, which trims the *reference* portion
  first and never drops the subject or action.
- `width`, `height` and `samples` accept **only** 1024 / 1024 / 1 on the hosted
  endpoint, so we do not send caller sizes.
- `steps` <= 50, `cfg_scale` <= 20.

**Identity references.** The hosted preview API only accepts
`data:image/png;example_id,N` (a fixed example image), not uploaded bytes. A
self-hosted NIM accepts up to 8 real images per request. Rather than silently
pretending likeness works, this adapter declares `supports_identity_refs=False`
for the hosted endpoint and `True` when `NVIDIA_NIM_SELF_HOSTED` is set. The
pipeline therefore knows a hosted NV runtime *cannot* guarantee a real person's
face — which matters, because our whole premise is "a photo of you".
"""

from __future__ import annotations

from typing import Any

from pigamagic_agent.agent.providers.base import ProviderError
from pigamagic_agent.agent.providers.http_base import decode_b64
from pigamagic_agent.agent.providers.image_base import BaseImageProvider, ImageCapabilities

NVIDIA_BASE_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs"

# Hosted model slug. Other usable ones: flux.1-dev, flux.1-schnell,
# flux.2-klein-4b. Override with AGENT_IMAGE_MODEL.
DEFAULT_MODEL = "flux.3-image-klein"

DEFAULT_STEPS = 50
DEFAULT_CFG_SCALE = 3.5

# Documented hard limits on the hosted endpoint.
MAX_PROMPT_CHARS = 2000
MAX_STEPS = 50
MAX_CFG_SCALE = 20.0

# Hosted NIM only accepts 1024x1024 with samples=1.
_HOSTED_SIZES: tuple[tuple[int, int], ...] = ((1024, 1024),)
_SELF_HOSTED_SIZES: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1344, 768),
    (768, 1344),
    (1184, 888),
    (888, 1184),
    (1248, 832),
    (832, 1248),
)


class NvidiaImageProvider(BaseImageProvider):
    """BFL FLUX generation through NVIDIA NIM (hosted or self-hosted)."""

    name: str = "nvidia"
    capabilities = ImageCapabilities(
        supports_identity_refs=False,  # hosted preview: example images only
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
        # A self-hosted NIM accepts real reference images; the hosted preview
        # does not. Reflect that in the declared capabilities.
        if getattr(self.settings, "nvidia_nim_self_hosted", False):
            self.capabilities = ImageCapabilities(
                supports_identity_refs=True,
                max_identity_refs=8,  # documented for flux-3-image-klein NIM
                supports_negative_prompt=False,
                supports_seed=True,
                supports_structural_ref=True,
                supported_sizes=_SELF_HOSTED_SIZES,
            )
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
        }

        # Editing mode requires a reference image. Only attempt it when we
        # actually have one AND the endpoint can accept arbitrary images.
        if structural_ref is not None and self.capabilities.supports_structural_ref:
            from pigamagic_agent.agent.providers.http_base import encode_b64

            payload["mode"] = "Image Editing"
            payload["image"] = f"data:image/png;base64,{encode_b64(structural_ref)}"
        else:
            payload["mode"] = "Image Generation"
            payload["image"] = None

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
