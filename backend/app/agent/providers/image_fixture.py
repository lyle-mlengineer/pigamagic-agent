"""Deterministic offline image provider (Steps 1-7 default).

Produces real, valid PNG bytes with no network and no cost, so the render layer
can be exercised end to end — including blob storage, hashing and the feed — long
before a vendor is wired in. Output is a solid colour band whose hue is derived
from the prompt hash, which makes it visibly different per shot and byte-stable
per input.

This is the provider the pipeline and every test use unless a real provider is
explicitly configured.
"""

from __future__ import annotations

import struct
import zlib
from typing import Any

from app.agent.providers.base import CostEstimate, GeneratedImage, ProviderError
from app.agent.providers.http_base import require_positive
from app.agent.providers.image_base import BaseImageProvider, ImageCapabilities

FIXTURE_MODEL = "fixture-diffusion-1"


class FixtureImageProvider(BaseImageProvider):
    """Offline stand-in that satisfies the same protocol as real vendors."""

    name: str = "fixture"
    capabilities = ImageCapabilities(
        supports_identity_refs=True,
        max_identity_refs=4,
        supports_negative_prompt=True,
        supports_seed=True,
        supports_structural_ref=True,
        supported_sizes=((1024, 1024), (768, 1024), (1024, 768), (128, 128)),
    )
    price_per_image_usd = 0.0

    def __post_init__(self) -> None:
        # No credentials, no network, no settings required.
        if not self.model:
            self.model = FIXTURE_MODEL

    def resolve_model(self, override: str | None) -> str:
        return override or self.model or FIXTURE_MODEL

    def require_credentials(self) -> None:
        return None

    def guard_offline(self) -> None:
        return None

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
        if not prompt or not prompt.strip():
            raise ProviderError("prompt must not be empty", provider=self.name, retryable=False)

        require_positive(width, "width")
        require_positive(height, "height")

        labelled = self._flatten_identity_refs(identity_refs)
        content = render_fixture_png(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            reference_count=len(labelled),
        )

        return GeneratedImage(
            content=content,
            content_type="image/png",
            width=width,
            height=height,
            seed=seed,
            provider=self.name,
            model=self.resolve_model(model),
            cost=CostEstimate.zero(),
            metadata={
                "fixture": True,
                "reference_count": len(labelled),
                "reference_keys": [key for key, _ in labelled],
                "prompt_chars": len(prompt),
            },
        )

    async def estimate_cost(self, *, count: int = 1, model: str | None = None) -> CostEstimate:
        return CostEstimate(credits=0, usd=0.0)

    def price_for(self, size: tuple[int, int], *, ref_count: int = 0) -> float:
        return 0.0


# --------------------------------------------------------------------------
# PNG rendering (no Pillow dependency, so this always works)
# --------------------------------------------------------------------------


def render_fixture_png(
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    reference_count: int = 0,
) -> bytes:
    """A valid RGB PNG with a deterministic gradient. Cheap and dependency-free."""
    import hashlib

    key = f"{prompt}|{negative_prompt}|{seed}|{reference_count}".encode("utf-8")
    digest = hashlib.sha256(key).digest()

    base_r, base_g, base_b = digest[0], digest[1], digest[2]
    # Keep it muted so it reads as a placeholder, not a real photo.
    base_r = 40 + base_r % 120
    base_g = 40 + base_g % 120
    base_b = 40 + base_b % 120

    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter type: None
        vertical = (y * 255) // max(1, height - 1)
        for x in range(width):
            horizontal = (x * 255) // max(1, width - 1)
            rows.append((base_r + horizontal // 4) % 256)
            rows.append((base_g + vertical // 4) % 256)
            rows.append((base_b + (horizontal + vertical) // 8) % 256)

    return _png_bytes(bytes(rows), width=width, height=height)


def _png_bytes(raw: bytes, *, width: int, height: int) -> bytes:
    """Assemble a minimal PNG from already-filtered scanlines."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


__all__ = ["FIXTURE_MODEL", "FixtureImageProvider", "render_fixture_png"]
