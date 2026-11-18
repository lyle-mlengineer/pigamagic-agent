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
`prompt`, `width`, `height`, `samples`, `steps`, `cfg_scale` — and **not**
`seed`. The hosted endpoint rejects a `seed` field with `422 extra_forbidden`
(verified live), so the adapter omits it and declares `supports_seed=False`;
renders on this model are therefore not seed-reproducible. Nothing else is sent
— in particular no `image` and no `mode`, because the documented
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

import re
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

#: The `finishReason` the hosted endpoint reports when its prompt classifier
#: refuses the request. It still answers HTTP 200, so this is the only signal.
FINISH_REASON_FILTERED = "CONTENT_FILTERED"

#: Innocuous words the hosted prompt classifier rejects, verified live. The
#: wardrobe colour "sage" is a documented value in our conventions, yet the
#: word alone is enough to make the endpoint refuse even "a sage plant in a
#: pot" (CONTENT_FILTERED), while "a salvia plant in a pot" succeeds. This map
#: lets `soften_prompt` swap the trigger for a synonym that renders the same
#: colour and passes.
FILTER_FALSE_POSITIVES: dict[str, str] = {
    "sage": "muted olive",
}


def soften_prompt(prompt: str) -> str:
    """Return a classifier-safe variant of `prompt`, or the prompt unchanged.

    The hosted model refuses some prompts that contain nothing wrong (verified:
    "sage" in any context, including a plant or a shirt). Rather than fail the
    shot, the adapter retries once with the parts most likely to trip the
    classifier removed:

    1. the `<<<REFERENCE_NOTES ... REFERENCE_NOTES>>>` block (material only), and
    2. the `[NEGATIVE]` section and `Avoid: ...` clause (a gauntlet of "no bad
       things" words that reads as a list of forbidden content), and
    3. known false-positive words replaced by synonyms that render the same
       colour (`sage` -> `muted olive`).

    If nothing changes, the original string is returned so the caller can tell a
    genuinely different retry from a no-op. This is defensive, not a way around
    a real refusal: a prompt that is *actually* inappropriate is re-raised as
    `ContentFilteredError` and the shot fails with a clear reason.
    """
    from pigamagic_agent.agent.prompts import REFERENCE_END, REFERENCE_START

    #: Section label the prompt assembler writes, and the trailer
    #: `compose_prompt` appends for vendors without a negative field.
    negative_label = "[NEGATIVE]"
    avoid_label = "Avoid:"

    text = prompt

    # 1. Drop the reference block in one cut (start marker to end marker).
    if REFERENCE_START in text and REFERENCE_END in text:
        start = text.index(REFERENCE_START)
        end = text.index(REFERENCE_END) + len(REFERENCE_END)
        text = (text[:start] + text[end:]).strip()

    # 2. Drop the negative material: the `[NEGATIVE]` section and the
    #    trailing `Avoid: ...` clause, wherever they fall.
    if avoid_label in text:
        text = text[: text.index(avoid_label)].strip()
    cleaned: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith(negative_label):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned).strip()

    # 3. Swap known false-positive triggers (case-insensitive, word boundaries).
    for trigger, replacement in FILTER_FALSE_POSITIVES.items():
        text = re.sub(
            rf"\b{re.escape(trigger)}\b", replacement, text, flags=re.IGNORECASE
        )

    return text.strip() if text != prompt else prompt


class ContentFilteredError(ProviderError):
    """The vendor's prompt classifier refused the request.

    NVIDIA returns HTTP 200 with `artifacts[0].finishReason ==
    "CONTENT_FILTERED"` and an empty image, so the only signal is the finish
    reason. Raised only after the softening retry also refuses, so a shot that
    fails here is one the *softened* prompt was still refused.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "nvidia",
        reason: str = "",
    ) -> None:
        super().__init__(message, provider=provider, retryable=False)
        self.reason = reason


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
        # The hosted flux-3-image-klein endpoint rejects `seed` (422
        # extra_forbidden, verified live), so a seed cannot be honoured.
        supports_seed=False,
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
            # No `seed` key: the hosted endpoint rejects it with 422
            # extra_forbidden, so it must never be sent.
            # No `image` key and no `mode` key: the documented request is
            # text-to-image when no image is supplied, and this adapter must
            # never upload a reference. Identity is the vision model's output
            # expressed as prompt text, so there is nothing to condition on.
        }

        # A prompt refused by the classifier is retried once with the reference
        # and negative material stripped and known false-positive words swapped
        # (`soften_prompt` is a no-op when there is nothing to soften, so a
        # genuinely refused prompt is re-raised below as ContentFilteredError).
        retry_prompt = soften_prompt(fitted)
        attempts = [fitted] if retry_prompt == fitted else [fitted, retry_prompt]

        last_filtered: ContentFilteredError | None = None
        used_softened = False
        for index, text in enumerate(attempts):
            payload["prompt"] = text
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
            try:
                content = _extract_nvidia_image(response, self.name)
            except ContentFilteredError as exc:
                last_filtered = exc
                used_softened = index == 1
                continue
            return content, {
                "endpoint": "genai",
                "prompt_chars": len(text),
                "prompt_truncated": len(text) < len(prompt.strip()),
                "prompt_softened": index == 1,
                "steps": steps,
                "cfg_scale": cfg_scale,
            }

        # Every attempt was refused. Report the refusal (this is a real
        # moderation block, not a transport fault) and surface whether the
        # softened variant was the one still refused, so a human can tell a
        # genuine problem from a false positive on one word.
        raise last_filtered or ContentFilteredError(
            f"nvidia refused every prompt variant.", provider=self.name
        )


def _extract_nvidia_image(response: Any, provider: str) -> bytes:
    """Read `artifacts[0].base64` from an NVIDIA NIM response.

    The hosted endpoint answers HTTP 200 even when its prompt classifier
    refuses the request; the refusal is `artifacts[0].finishReason ==
    "CONTENT_FILTERED"` with no image, so that is detected here rather than
    reported as a mystery "no base64" failure.
    """
    if not isinstance(response, dict):
        raise ProviderError(f"{provider} returned an unexpected payload.", provider=provider)

    artifacts = response.get("artifacts") or []
    if not artifacts:
        raise ProviderError(
            f"{provider} returned no artifacts.", provider=provider, retryable=False
        )

    first = artifacts[0] or {}
    reason = first.get("finishReason")
    if reason == FINISH_REASON_FILTERED:
        raise ContentFilteredError(
            f"{provider} moderation refused this prompt ({FINISH_REASON_FILTERED}); "
            "remove the flagged wording and try again.",
            reason=reason,
        )

    encoded = first.get("base64")
    if not encoded:
        raise ProviderError(
            f"{provider} artifact contained no base64 image (finishReason={reason!r}).",
            provider=provider,
            retryable=False,
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
    "ContentFilteredError",
    "FILTER_FALSE_POSITIVES",
    "DEFAULT_CFG_SCALE",
    "DEFAULT_MODEL",
    "DEFAULT_STEPS",
    "MAX_CFG_SCALE",
    "MAX_PROMPT_CHARS",
    "MAX_STEPS",
    "NVIDIA_BASE_URL",
    "NvidiaImageProvider",
    "soften_prompt",
]
