"""fal.ai FLUX image provider.

fal exposes an asynchronous **queue**: submit returns a request id, then you poll
status and fetch the result. That is a different shape from Gemini/OpenAI's
synchronous responses, which is exactly why this adapter exists — it proves the
generic base covers both shapes rather than assuming "one HTTP call = one image".

Verified against current fal docs (2026):

- Submit:   `POST https://queue.fal.run/{owner}/{app}[/{path}]`
- Status:   `GET  https://queue.fal.run/{owner}/{app}/requests/{id}/status`
- Result:   `GET  https://queue.fal.run/{owner}/{app}/requests/{id}`
- Auth:     `Authorization: Key $FAL_KEY`
- Multi-reference edits accept up to 9-10 images (`flux-2-pro/edit`,
  `flux-3/edit-image`), so identity conditioning is supported.

Result images arrive as URLs, so the adapter downloads them to bytes — the
pipeline never sees a URL.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pigamagic_agent.agent.providers.base import ProviderError
from pigamagic_agent.agent.providers.http_base import encode_b64
from pigamagic_agent.agent.providers.image_base import BaseImageProvider, ImageCapabilities

FAL_QUEUE_URL = "https://queue.fal.run"

DEFAULT_MODEL = "blackforestlabs/flux-3/text-to-image"
EDIT_MODEL = "blackforestlabs/flux-3/edit-image"

_SUPPORTED_SIZES: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1344, 768),
    (768, 1344),
)

POLL_INTERVAL_S = 1.5
MAX_POLLS = 120  # ~3 minutes, then give up

# Documented per-image prices at 1K.
_PRICE_PER_IMAGE_USD = 0.048


class FalImageProvider(BaseImageProvider):
    """FLUX generation/editing via the fal queue API."""

    name: str = "fal"
    capabilities = ImageCapabilities(
        supports_identity_refs=True,
        max_identity_refs=9,
        supports_negative_prompt=True,  # FLUX exposes a real negative field
        supports_seed=True,
        supports_structural_ref=True,
        supported_sizes=_SUPPORTED_SIZES,
    )
    price_per_image_usd = _PRICE_PER_IMAGE_USD

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.base_url:
            self.base_url = FAL_QUEUE_URL
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
        # With references we must use an edit endpoint; text-to-image ignores them.
        endpoint = model
        inputs: dict[str, Any] = {
            "prompt": prompt,
            "num_images": 1,
            "image_size": {"width": size[0], "height": size[1]},
            "output_format": "png",
        }
        if negative_prompt:
            inputs["negative_prompt"] = negative_prompt
        if seed is not None:
            inputs["seed"] = seed

        refs = list(identity_refs)
        if refs or structural_ref is not None:
            # Reference images are sent as data URLs so no hosting is required.
            images = [
                {"image_url": f"data:image/png;base64,{encode_b64(content)}"}
                for _, content in refs
            ]
            if structural_ref is not None:
                images.append(
                    {"image_url": f"data:image/png;base64,{encode_b64(structural_ref)}"}
                )
            inputs["image_urls"] = [item["image_url"] for item in images]
            if "text-to-image" in endpoint:
                endpoint = endpoint.replace("text-to-image", "edit-image")

        inputs.update(params.get("inputs") or {})

        submitted = await self.request_json(
            "POST",
            f"{self.base_url}/{endpoint}",
            headers={
                "Authorization": f"Key {self.api_key}",
                "Content-Type": "application/json",
            },
            json_body=inputs,
            estimate=estimate,
        )
        if not isinstance(submitted, dict):
            raise ProviderError(f"{self.name} returned no request handle.", provider=self.name)

        status_url = submitted.get("status_url") or (
            f"{self.base_url}/{endpoint}/requests/{submitted.get('request_id')}/status"
        )
        response_url = submitted.get("response_url") or (
            f"{self.base_url}/{endpoint}/requests/{submitted.get('request_id')}"
        )

        result = await self._poll(status_url, response_url)
        image_url = _first_image_url(result, self.name)
        content = await self.request_bytes(
            image_url, headers={"Authorization": f"Key {self.api_key}"}
        )
        return content, {"endpoint": endpoint, "request_id": submitted.get("request_id", "")}

    async def _poll(self, status_url: str, response_url: str) -> Any:
        """Poll until COMPLETED, then fetch the result payload."""
        headers = {"Authorization": f"Key {self.api_key}"}
        for _ in range(MAX_POLLS):
            status = await self.request_json("GET", status_url, headers=headers)
            state = str((status or {}).get("status", "") if isinstance(status, dict) else "")

            if state == "COMPLETED":
                return await self.request_json("GET", response_url, headers=headers)
            if state in {"FAILED", "CANCELLED"}:
                raise ProviderError(
                    f"{self.name} prediction {state.lower()}.",
                    provider=self.name,
                    retryable=state == "FAILED",
                )
            await asyncio.sleep(POLL_INTERVAL_S)

        raise ProviderError(
            f"{self.name} prediction did not finish in time.", provider=self.name, retryable=True
        )


def _first_image_url(result: Any, provider: str) -> str:
    if not isinstance(result, dict):
        raise ProviderError(f"{provider} returned an unexpected result payload.", provider=provider)
    images = result.get("images") or []
    if not images:
        raise ProviderError(f"{provider} returned no images.", provider=provider, retryable=False)
    first = images[0] or {}
    url = first.get("url") if isinstance(first, dict) else str(first)
    if not url:
        raise ProviderError(f"{provider} image entry had no URL.", provider=provider)
    return str(url)


__all__ = [
    "DEFAULT_MODEL",
    "EDIT_MODEL",
    "FAL_QUEUE_URL",
    "FalImageProvider",
    "MAX_POLLS",
    "POLL_INTERVAL_S",
]
