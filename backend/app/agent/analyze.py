"""Reference photo analysis and deterministic fusion (design.md §5).

Pipeline:

    images + descriptions
        │
        ├─ hash each image ─► cache hit? reuse : vision pass ─► ReferenceAnalysis
        │
        └─ fuse() ─► ShootCast (one entry per person, not per photo)

**Why fusion exists.** Three photos of a couple contain six person-detections.
The shoot has two people. Naive concatenation would give the planner six cast
members and produce photos with the wrong number of people in them.

**The quality rule that makes this work.** When two images both describe the
same person, the *sharper, more frontal* one wins for facial fields. Averaging
or last-wins both degrade identity, because a blurry reference contributed by a
user "so you can see the outfit" would otherwise overwrite a good face.

**Determinism.** Fusion is a pure function of its inputs: no dict iteration
order dependence, no randomness, no time. The same references always produce
the same cast, which is what lets the generation cache be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from app.agent.config import AgentSettings, get_settings
from app.agent.hashing import reference_analysis_key, sha256_bytes
from app.agent.profiles import (
    CastEntry,
    FaceFeatures,
    Garment,
    GroupObservation,
    HairFeatures,
    PersonProfile,
    SceneHints,
    WardrobeProfile,
)
from app.agent.providers.base import VisionProvider
from app.agent.providers.vision_fixture import SCENE_QUALITIES

# Fields where a sharper image's observation should win outright.
FACE_FIELDS = ("shape", "eyes", "brows", "nose", "mouth", "jaw")
HAIR_FIELDS = ("length", "style", "colour", "texture")

# Bumped when fusion logic changes shape, mirroring PROMPT_VERSION semantics.
FUSION_VERSION = "1"

# Quality ordering: higher is better for facial observation.
_QUALITY_RANK = {name: rank for rank, name in enumerate(SCENE_QUALITIES)}


def _quality_rank(scene_quality: str) -> int:
    return _QUALITY_RANK.get(scene_quality, 0)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReferenceImage:
    """One uploaded reference photo plus the user's description of it.

    `media_id` is the shell's Media row id; `content` is the raw bytes.
    `description` is user-authored text and is the ONLY trusted source of
    person labels (a description naming "my sister Ada" outranks ordinal
    guessing).
    """

    media_id: str
    content: bytes
    description: str = ""

    def __post_init__(self) -> None:
        if not self.media_id:
            raise ValueError("ReferenceImage.media_id must not be empty")
        if not self.content:
            raise ValueError(f"ReferenceImage {self.media_id!r} has no content")

    @property
    def content_hash(self) -> str:
        return sha256_bytes(self.content)


@dataclass(slots=True)
class ReferenceAnalysis:
    """What the vision pass saw in one image.

    Mirrors the structured-output schema the adapter must satisfy. Deliberately
    permits *omitted* face/hair fields (None) so an unobserved field can never
    overwrite a good one during fusion.
    """

    media_id: str
    scene_quality: str
    people: list[dict[str, Any]] = field(default_factory=list)
    group: dict[str, Any] = field(default_factory=dict)
    venue_cues: list[str] = field(default_factory=list)
    notes: str = ""

    def quality_rank(self) -> int:
        return _quality_rank(self.scene_quality)


@dataclass(slots=True)
class AnalysisOutcome:
    """Result of analysing a whole set of references."""

    cast: list[CastEntry]
    scene_hints: SceneHints
    analyses: list[ReferenceAnalysis]
    warnings: list[str] = field(default_factory=list)
    vision_calls: int = 0
    cache_hits: int = 0

    def cast_by_key(self) -> dict[str, CastEntry]:
        return {entry.person_key: entry for entry in self.cast}


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


class ReferenceAnalysisCache(Protocol):
    """Storage for per-image analyses, keyed by content hash."""

    def get(self, key: str) -> ReferenceAnalysis | None: ...

    def put(self, key: str, analysis: ReferenceAnalysis) -> None: ...


class InMemoryAnalysisCache:
    """Simple cache used in tests and single-process runs.

    A Postgres-backed implementation replaces this at Step 11 without changing
    the pipeline, because both satisfy the same protocol.
    """

    def __init__(self) -> None:
        self._store: dict[str, ReferenceAnalysis] = {}

    def get(self, key: str) -> ReferenceAnalysis | None:
        return self._store.get(key)

    def put(self, key: str, analysis: ReferenceAnalysis) -> None:
        self._store[key] = analysis

    def __len__(self) -> int:
        return len(self._store)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


ANALYSIS_INSTRUCTION = (
    "Describe only what is visible. Report appearance in photographic terms "
    "(shape, colour, texture). Do not infer identity, ethnicity, religion or "
    "nationality. Omit any field you cannot see rather than guessing. Set "
    "scene_quality to one of: " + ", ".join(SCENE_QUALITIES) + "."
)


async def analyze_references(
    references: Sequence[ReferenceImage],
    *,
    vision: VisionProvider,
    cache: ReferenceAnalysisCache | None = None,
    settings: AgentSettings | None = None,
) -> AnalysisOutcome:
    """Analyse each reference once, then fuse into a shoot cast.

    Identical bytes are analysed once even if supplied twice: the hash key is
    the identity of the work, not the media row (design.md §5.1).
    """
    active = settings or get_settings()
    store = cache if cache is not None else InMemoryAnalysisCache()

    analyses: list[ReferenceAnalysis] = []
    warnings: list[str] = []
    vision_calls = 0
    cache_hits = 0

    for reference in references:
        key = reference_analysis_key(reference.content)
        cached = store.get(key)
        if cached is not None:
            cache_hits += 1
            # Re-stamp the media id: the same photo may be used by a new row.
            analyses.append(_with_media_id(cached, reference.media_id))
            continue

        result = await vision.describe_image(
            image=reference.content,
            instruction=ANALYSIS_INSTRUCTION,
            schema={},  # adapters validate against their own schema registry
        )
        vision_calls += 1

        analysis = _analysis_from_payload(result.data, reference.media_id)
        if analysis.scene_quality not in SCENE_QUALITIES:
            warnings.append(
                f"Unrecognised scene_quality {analysis.scene_quality!r} for "
                f"{reference.media_id}; treated as 'usable'."
            )
            analysis.scene_quality = "usable"

        store.put(key, analysis)
        analyses.append(analysis)

    if not analyses:
        raise ValueError("analyze_references requires at least one reference image")

    # Descriptions travel with the media id; fusion needs them for labelling.
    descriptions = {ref.media_id: ref.description for ref in references}
    cast, fuse_warnings = fuse_cast(analyses, descriptions)
    warnings.extend(fuse_warnings)

    if len(cast) == 0:
        warnings.append(
            "No people could be identified in the reference photos; the shoot "
            "will fall back to text-described subjects."
        )

    for entry in cast:
        if not entry.person.is_identity_anchored():
            warnings.append(
                f"{entry.person.label or entry.person_key} has no usable reference "
                "image, so face consistency will be weaker."
            )

    scene_hints = derive_scene_hints(analyses, active)
    return AnalysisOutcome(
        cast=cast,
        scene_hints=scene_hints,
        analyses=analyses,
        warnings=warnings,
        vision_calls=vision_calls,
        cache_hits=cache_hits,
    )


def _with_media_id(analysis: ReferenceAnalysis, media_id: str) -> ReferenceAnalysis:
    """Copy a cached analysis onto the media id currently being analysed."""
    return ReferenceAnalysis(
        media_id=media_id,
        scene_quality=analysis.scene_quality,
        people=analysis.people,
        group=analysis.group,
        venue_cues=analysis.venue_cues,
        notes=analysis.notes,
    )


def _analysis_from_payload(payload: dict[str, Any], media_id: str) -> ReferenceAnalysis:
    return ReferenceAnalysis(
        media_id=media_id,
        scene_quality=str(payload.get("scene_quality", "usable")),
        people=list(payload.get("people", [])),
        group=dict(payload.get("group", {})),
        venue_cues=[str(cue) for cue in payload.get("venue_cues", [])],
        notes=str(payload.get("notes", "")),
    )


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def fuse_cast(
    analyses: Sequence[ReferenceAnalysis],
    descriptions: dict[str, str] | None = None,
) -> tuple[list[CastEntry], list[str]]:
    """Merge per-image people into one cast entry per person.

    Deterministic given the same analyses and descriptions. Returns the cast
    plus any warnings worth surfacing to the user.
    """
    texts = descriptions or {}
    warnings: list[str] = []

    # Rank best-first, so the first observation we keep for a field is the
    # sharpest one available.
    ranked = sorted(analyses, key=lambda item: (-item.quality_rank(), item.media_id))

    merged: dict[str, dict[str, Any]] = {}

    for analysis in ranked:
        for position, raw in enumerate(analysis.people):
            person_key = _resolve_person_key(raw, position)
            slot = merged.setdefault(
                person_key,
                {
                    "face": {},
                    "hair": {},
                    "confidence": 0.0,
                    "media_ids": [],
                    "labels": [],
                    "build": None,
                    "skin_tone": None,
                    "presentation": None,
                    "age_band": None,
                    "distinguishing": [],
                    "garments": [],
                    "accessories": [],
                    "styles": [],
                    "must_keep": [],
                    "avoid": [],
                },
            )

            # --- face & hair: first (sharpest) observation wins per field ----
            for name in FACE_FIELDS:
                value = (raw.get("face") or {}).get(name)
                if value and name not in slot["face"]:
                    slot["face"][name] = value
            for name in HAIR_FIELDS:
                value = (raw.get("hair") or {}).get(name)
                if value and name not in slot["hair"]:
                    slot["hair"][name] = value

            # --- scalars: first non-null wins (identical rule) --------------
            for source_key, slot_key in (
                ("build", "build"),
                ("skin_tone", "skin_tone"),
                ("gender_presentation", "presentation"),
                ("age_band", "age_band"),
            ):
                value = raw.get(source_key)
                if value and slot[slot_key] is None:
                    slot[slot_key] = value

            for item in raw.get("distinguishing") or []:
                if item and item not in slot["distinguishing"]:
                    slot["distinguishing"].append(item)

            # --- wardrobe: union, deduped, first colour wins ----------------
            wardrobe = raw.get("wardrobe") or {}
            for garment in wardrobe.get("garments") or []:
                _merge_garment(slot["garments"], garment)
            for accessory in wardrobe.get("accessories") or []:
                _append_unique(slot["accessories"], accessory)
            if wardrobe.get("overall_style"):
                slot["styles"].append(wardrobe["overall_style"])
            for item in wardrobe.get("must_keep") or []:
                _append_unique(slot["must_keep"], item)
            for item in wardrobe.get("avoid") or []:
                _append_unique(slot["avoid"], item)

            # --- confidence: sharpest observation dominates ----------------
            confidence = float(raw.get("confidence") or 0.0)
            slot["confidence"] = max(slot["confidence"], min(confidence, 1.0))

            # --- provenance -------------------------------------------------
            if analysis.media_id not in slot["media_ids"]:
                slot["media_ids"].append(analysis.media_id)
            label = str(raw.get("label") or "").strip()
            if label:
                _append_unique(slot["labels"], label)

            # A user description naming this person outranks a model label.
            description = texts.get(analysis.media_id, "")
            named = _label_from_description(description, position)
            if named:
                _append_unique(slot["labels"], named)

    cast: list[CastEntry] = []
    for position, person_key in enumerate(sorted(merged, key=_person_key_sort)):
        slot = merged[person_key]
        face = FaceFeatures(**slot["face"])
        hair = HairFeatures(**slot["hair"])

        # Confidence reflects how much was actually observed, so a person
        # glimpsed in one blurry photo is honestly weak.
        observed = face.observed_field_count() + hair.observed_field_count()
        confidence = round(min(slot["confidence"], observed / (len(FACE_FIELDS) + len(HAIR_FIELDS))), 3)

        person = PersonProfile(
            person_key=person_key,
            label=(slot["labels"][0] if slot["labels"] else ""),
            age_band=slot["age_band"] or "adult",
            gender_presentation=slot["presentation"] or "unspecified",
            face=face,
            hair=hair,
            build=slot["build"] or "unspecified",
            skin_tone=slot["skin_tone"],
            distinguishing=slot["distinguishing"],
            reference_media_ids=slot["media_ids"],
            confidence=confidence,
        )

        wardrobe = None
        if slot["garments"] or slot["accessories"]:
            wardrobe = WardrobeProfile(
                person_key=person_key,
                garments=slot["garments"],
                accessories=slot["accessories"],
                overall_style=_dominant_style(slot["styles"]),
                must_keep=slot["must_keep"],
                avoid=slot["avoid"],
            )

        cast.append(
            CastEntry(
                person=person,
                wardrobe=wardrobe,
                # The first person most people photograph is the subject.
                role="primary" if position == 0 else "guest",
                position=position,
            )
        )

    if len(cast) == 1 and len(analyses) > 1:
        warnings.append(
            "Only one person was identified across all references; check that the "
            "descriptions name everyone who should appear."
        )

    return cast, warnings


def _person_key_sort(key: str) -> int:
    """Sort p1, p2, ... p10 numerically rather than lexically."""
    digits = "".join(char for char in key if char.isdigit())
    return int(digits) if digits else 999


def _resolve_person_key(raw: dict[str, Any], position: int) -> str:
    """Use the model's key if it is well-formed, else fall back to position.

    Position is a stable proxy for identity across images taken the same way,
    which is why it is the documented fallback (design.md §5.2).
    """
    candidate = str(raw.get("person_key") or "").strip()
    if candidate.startswith("p") and candidate[1:].isdigit():
        return candidate
    return f"p{position + 1}"


def _merge_garment(garments: list[Garment], raw: dict[str, Any]) -> None:
    name = str(raw.get("name") or "").strip()
    if not name:
        return
    for existing in garments:
        if existing.name.casefold() == name.casefold():
            # Fill only what is missing; the first (sharpest) image's colour wins.
            if existing.colour is None and raw.get("colour"):
                existing.colour = str(raw["colour"])
            if existing.fabric is None and raw.get("fabric"):
                existing.fabric = str(raw["fabric"])
            if existing.fit is None and raw.get("fit"):
                existing.fit = str(raw["fit"])
            return
    garments.append(
        Garment(
            name=name,
            colour=str(raw["colour"]) if raw.get("colour") else None,
            fabric=str(raw["fabric"]) if raw.get("fabric") else None,
            fit=str(raw["fit"]) if raw.get("fit") else None,
        )
    )


def _append_unique(items: list[str], value: Any) -> None:
    text = str(value).strip()
    if not text:
        return
    if not any(existing.casefold() == text.casefold() for existing in items):
        items.append(text)


def _dominant_style(styles: list[str]) -> str:
    """Most-mentioned style, with a deterministic tie-break."""
    if not styles:
        return "casual"
    counts: dict[str, int] = {}
    for style in styles:
        counts[style] = counts.get(style, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _label_from_description(description: str, position: int) -> str:
    """Extract a person label from a user description.

    Users write things like "my sister Ada at the gate". We take the first
    clause and trim it to a short label; ordinals ("the second person") are
    handled by the caller's position fallback.
    """
    if not description:
        return ""
    clause = description.strip().split(".")[0]
    if not clause:
        return ""
    # Keep it short; a full sentence is not a label.
    words = clause.split()
    label = " ".join(words[:6])
    return label[:60]


# --------------------------------------------------------------------------
# Scene hints
# --------------------------------------------------------------------------


def derive_scene_hints(
    analyses: Sequence[ReferenceAnalysis], settings: AgentSettings
) -> SceneHints:
    """Summarise group and venue cues across all references, deterministically."""
    dynamics: list[str] = []
    venue_features: list[str] = []
    sizes: list[int] = []

    for analysis in sorted(analyses, key=lambda item: item.media_id):
        for item in analysis.group.get("dynamics") or []:
            _append_unique(dynamics, item)
        for cue in analysis.venue_cues:
            _append_unique(venue_features, cue)
        size = analysis.group.get("size")
        if isinstance(size, int) and size > 0:
            sizes.append(size)

    # The largest group observed is the most informative estimate.
    group_size = max(sizes) if sizes else None

    return SceneHints(
        group_size=group_size,
        group_dynamics=dynamics,
        clothing_style=None,
        glimpsed_venue_features=venue_features[: settings.max_images_per_venue or 10],
        mood=None,
    )


__all__ = [
    "AnalysisOutcome",
    "FUSION_VERSION",
    "FACE_FIELDS",
    "HAIR_FIELDS",
    "InMemoryAnalysisCache",
    "ReferenceAnalysis",
    "ReferenceAnalysisCache",
    "ReferenceImage",
    "analyze_references",
    "derive_scene_hints",
    "fuse_cast",
]
