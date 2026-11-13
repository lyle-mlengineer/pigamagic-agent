"""Deterministic vision adapter for offline development and tests.

Two responsibilities:

1. **Satisfy `VisionProvider`** so the pipeline can be exercised end-to-end with
   no credentials (Steps 2-7).
2. **Be genuinely deterministic.** The same bytes must always produce the same
   analysis, including across processes, because the analysis cache is keyed by
   content hash. Any nondeterminism here would make the cache lie.

How determinism is achieved: every "observation" is derived from the SHA-256 of
the image bytes. Different images give different (but reproducible) people, and
identical images give identical results.
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.agent.hashing import canonical_json
from app.agent.providers.base import CostEstimate, VisionResult

# Vocabulary pools. Kept small and visual-only: the fixture must never imply an
# identity, religion or ethnicity (SR-3).
_FACE_SHAPES = ("oval", "round", "heart-shaped", "square", "long", "diamond")
_EYES = ("wide-set", "close-set", "almond", "round", "deep-set", "upturned")
_BROWS = ("straight", "arched", "thick", "fine", "angular")
_NOSES = ("straight", "aquiline", "button", "broad", "narrow")
_MOUTHS = ("full", "thin-lipped", "wide smile", "bow-shaped", "straight")
_JAWS = ("soft", "defined", "square", "narrow", "rounded")

_HAIR_LENGTHS = ("long", "shoulder-length", "short", "cropped", "buzz")
_HAIR_STYLES = ("braided updo", "loose waves", "bun", "fade", "box braids", "ponytail")
_HAIR_COLOURS = ("black", "dark brown", "auburn", "blonde", "greying")
_HAIR_TEXTURES = ("coily", "wavy", "straight", "curly", "fine")

_BUILDS = ("slim", "average", "athletic", "curvy", "plus")
_SKIN_TONES = ("deep brown", "warm brown", "olive", "tan", "fair")
_PRESENTATIONS = ("masculine", "feminine", "androgynous")

_GARMENT_NAMES = ("linen shirt", "floor-length gown", "tailored suit", "sundress", "knit top")
_GARMENT_COLOURS = ("ivory", "charcoal", "sage green", "burgundy", "navy")
_ACCESSORIES = ("gold studs", "a wristwatch", "pearl earrings", "a signet ring")

# Scene-quality buckets, ordered worst to best for *identity observation*.
#
# A portrait close-up outranks a generic "usable" photo on purpose: a tight,
# well-lit portrait is the best possible anchor for facial features, whereas a
# usable group shot still shows each face small. Fusion prefers higher ranks.
SCENE_QUALITIES = ("too-dark", "blurry", "usable", "portrait-closeup")

VENUE_CUES = (
    "tiled floor",
    "tall windows",
    "warm wall lights",
    "green foliage",
    "polished wood",
)


def _digest(image: bytes) -> str:
    return hashlib.sha256(image).hexdigest()


def _pick(pool: tuple[str, ...], digest: str, salt: str) -> str:
    """Deterministically choose a pool member from the image digest."""
    seed = hashlib.sha256(f"{digest}:{salt}".encode()).digest()
    return pool[int.from_bytes(seed[:4], "big") % len(pool)]


def _quality_rank(scene_quality: str) -> int:
    try:
        return SCENE_QUALITIES.index(scene_quality)
    except ValueError:
        return 0


class FixtureVisionProvider:
    """Offline, deterministic `VisionProvider` implementation."""

    name = "fixture"

    def __init__(self, people_per_image: int = 1, default_quality: str = "usable") -> None:
        if people_per_image < 0:
            raise ValueError("people_per_image must be >= 0")
        if default_quality not in SCENE_QUALITIES:
            raise ValueError(f"default_quality must be one of {SCENE_QUALITIES}")
        self.people_per_image = people_per_image
        self.default_quality = default_quality
        self.calls: list[dict[str, Any]] = []

    # --- VisionProvider ---------------------------------------------------

    async def describe_image(
        self,
        *,
        image: bytes,
        instruction: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> VisionResult:
        digest = _digest(image)
        self.calls.append({"digest": digest, "instruction": instruction, "model": model})

        payload = self._analysis_for(digest)
        return VisionResult(
            data=payload,
            description=", ".join(
                f"{person['label'] or 'a person'} with {person['hair']['style']} hair"
                for person in payload["people"]
            ),
            provider=self.name,
            model=model or "fixture-vision",
            cost=CostEstimate(credits=0, usd=0.0),
            latency_ms=0,
        )

    async def estimate_cost(self, *, image_count: int = 1, model: str | None = None) -> CostEstimate:
        # Fixtures are free by design: Steps 2-7 must cost nothing.
        return CostEstimate(credits=0, usd=0.0)

    # --- fixture behaviour ------------------------------------------------

    def _analysis_for(self, digest: str) -> dict[str, Any]:
        """Build a deterministic analysis document from an image digest."""
        people = []
        for index in range(self.people_per_image):
            salt = f"person-{index}"
            people.append(
                {
                    "person_key": f"p{index + 1}",
                    "label": "",
                    "age_band": "adult",
                    "gender_presentation": _pick(_PRESENTATIONS, digest, f"{salt}-pres"),
                    "face": {
                        "shape": _pick(_FACE_SHAPES, digest, f"{salt}-shape"),
                        "eyes": _pick(_EYES, digest, f"{salt}-eyes"),
                        "brows": _pick(_BROWS, digest, f"{salt}-brows"),
                        "nose": _pick(_NOSES, digest, f"{salt}-nose"),
                        "mouth": _pick(_MOUTHS, digest, f"{salt}-mouth"),
                        "jaw": _pick(_JAWS, digest, f"{salt}-jaw"),
                    },
                    "hair": {
                        "length": _pick(_HAIR_LENGTHS, digest, f"{salt}-hlen"),
                        "style": _pick(_HAIR_STYLES, digest, f"{salt}-hstyle"),
                        "colour": _pick(_HAIR_COLOURS, digest, f"{salt}-hcol"),
                        "texture": _pick(_HAIR_TEXTURES, digest, f"{salt}-htex"),
                    },
                    "build": _pick(_BUILDS, digest, f"{salt}-build"),
                    "skin_tone": _pick(_SKIN_TONES, digest, f"{salt}-skin"),
                    "distinguishing": [],
                    "confidence": 0.8,
                    "wardrobe": {
                        "garments": [
                            {
                                "name": _pick(_GARMENT_NAMES, digest, f"{salt}-gname"),
                                "colour": _pick(_GARMENT_COLOURS, digest, f"{salt}-gcol"),
                                "fabric": None,
                                "fit": None,
                            }
                        ],
                        "accessories": [_pick(_ACCESSORIES, digest, f"{salt}-acc")],
                        "overall_style": "smart-casual",
                        "must_keep": [],
                        "avoid": [],
                    },
                }
            )

        return {
            "people": people,
            "group": {
                "size": max(len(people), 1) if people else None,
                "dynamics": ["affectionate"],
                "relative_heights": [],
            },
            "venue_cues": [_pick(VENUE_CUES, digest, "venue-cue")],
            "scene_quality": self._quality_for(digest),
            "notes": f"fixture analysis ({digest[:8]})",
        }

    def _quality_for(self, digest: str) -> str:
        """Mostly usable, occasionally degraded, deterministically."""
        seed = int.from_bytes(hashlib.sha256(f"{digest}:quality".encode()).digest()[:2], "big")
        if seed % 5 == 0:
            return "blurry"
        if seed % 7 == 0:
            return "too-dark"
        return self.default_quality


class ScriptedVisionProvider:
    """Test double that returns scripted analyses in call order.

    Used to test fusion decisions that the deterministic fixture cannot
    express, e.g. "the blurry image claims a different eye shape".
    """

    name = "scripted"

    def __init__(self, analyses: list[dict[str, Any]]) -> None:
        self._analyses = list(analyses)
        self.calls: list[bytes] = []

    async def describe_image(
        self,
        *,
        image: bytes,
        instruction: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> VisionResult:
        self.calls.append(image)
        if not self._analyses:
            raise AssertionError("ScriptedVisionProvider ran out of scripted analyses")
        payload = self._analyses.pop(0)
        return VisionResult(
            data=payload,
            provider=self.name,
            model=model or "scripted",
            cost=CostEstimate.zero(),
        )

    async def estimate_cost(self, *, image_count: int = 1, model: str | None = None) -> CostEstimate:
        return CostEstimate.zero()


def quality_rank(scene_quality: str) -> int:
    """Public accessor so fusion and tests share one ordering."""
    return _quality_rank(scene_quality)


__all__ = [
    "FixtureVisionProvider",
    "SCENE_QUALITIES",
    "ScriptedVisionProvider",
    "quality_rank",
]
