"""The photoshoot plan: the compiled artifact every photo is generated from.

The plan is stored, versioned and diffable data, never a transient prompt
(design.md §3.2). That single choice is what buys reproducibility, cheap
single-photo edits, testability without a provider, and auditability.

    ShootPlan ── ShotSpec[] ──> one PhotoJobSpec each ──> one photo
                     ▲
                EditInstruction (delta from a comment)

Invariants enforced here, not left to the LLM:
- a shot must name at least one subject
- subjects must be valid person keys
- shot count is bounded by settings.max_shots
- coverage counters must agree with the shot list
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pigamagic_agent.agent.profiles import (
    CastEntry,
    Confidence,
    PersonProfile,
    ShortText,
    VenueProfile,
    WardrobeProfile,
)

Framing = Literal["wide", "medium", "close", "detail", "over-shoulder", "candid"]
Phase = Annotated[str, Field(min_length=1, max_length=60)]

# Edit vocabulary is closed on purpose: a delta must map to a known field so it
# can be applied deterministically (design.md §11.1).
EditChange = Literal[
    "subject",
    "activity",
    "framing",
    "lighting",
    "venue_emphasis",
    "wardrobe",
    "style",
    "remove_person",
    "add_person",
    "regenerate",
]


class PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------
# Camera and shots
# --------------------------------------------------------------------------


class CameraSpec(PlanModel):
    """Concrete photographic parameters. Deterministic, so prompts are stable."""

    focal_length_mm: int = Field(default=50, ge=8, le=600)
    angle: Literal["eye-level", "low", "high", "overhead", "shoulder"] = "eye-level"
    depth_of_field: Literal["shallow", "medium", "deep"] = "medium"
    aperture_f: float = Field(default=2.8, ge=0.95, le=32.0)


class ShotSpec(PlanModel):
    """One planned photograph. The unit of generation, retry and editing."""

    node_id: Annotated[str, Field(pattern=r"^s[0-9]{2,3}$")]
    phase: Phase = "main"
    moment: ShortText = Field(min_length=1)
    activity: ShortText = Field(min_length=1)
    description: str = Field(default="", max_length=600)
    subjects: list[str] = Field(min_length=1, max_length=20)
    framing: Framing = "medium"
    camera: CameraSpec = Field(default_factory=CameraSpec)
    lighting: str = Field(default="", max_length=300)
    venue_anchors: list[ShortText] = Field(default_factory=list, max_length=8)
    wardrobe_notes: list[ShortText] = Field(default_factory=list, max_length=8)
    must_include: list[ShortText] = Field(default_factory=list, max_length=8)
    must_avoid: list[ShortText] = Field(default_factory=list, max_length=8)
    style_preset: Annotated[str, Field(min_length=1, max_length=60)] = "editorial"
    caption: str = Field(default="", max_length=200)
    why: str = Field(default="", max_length=300)

    @field_validator("subjects")
    @classmethod
    def _subjects_are_person_keys(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for key in value:
            item = key.strip()
            if not item:
                continue
            if not (item.startswith("p") and item[1:].isdigit()):
                raise ValueError(f"subject {item!r} is not a person key like 'p1'")
            if item not in cleaned:
                cleaned.append(item)
        if not cleaned:
            raise ValueError("a shot must name at least one subject")
        return cleaned

    @field_validator(
        "venue_anchors", "wardrobe_notes", "must_include", "must_avoid"
    )
    @classmethod
    def _clean_lists(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    def signature(self) -> tuple[str, str, tuple[str, ...]]:
        """Dedup key: same moment + framing + subjects means the same photo."""
        return (self.moment.casefold(), self.framing, tuple(sorted(self.subjects)))


class ModelRef(PlanModel):
    """Which model produced something. Kept for audit and cost attribution."""

    provider: str = Field(default="", max_length=60)
    model: str = Field(default="", max_length=120)


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


class ShootPlan(PlanModel):
    """A complete, validated photoshoot plan for one shoot revision."""

    shoot_id: str = Field(min_length=1, max_length=60)
    revision: int = Field(default=1, ge=1)
    cast: list[CastEntry] = Field(min_length=1, max_length=30)
    venue: VenueProfile
    style_preset: Annotated[str, Field(min_length=1, max_length=60)] = "editorial"
    shots: list[ShotSpec] = Field(min_length=1, max_length=64)
    coverage: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list, max_length=50)
    compiler: ModelRef = Field(default_factory=ModelRef)
    confidence: Confidence = 0.0

    # --- validation ------------------------------------------------------

    @model_validator(mode="after")
    def _node_ids_are_unique(self) -> "ShootPlan":
        ids = [shot.node_id for shot in self.shots]
        duplicates = [node for node, count in Counter(ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate node_id(s): {sorted(duplicates)}")
        return self

    @model_validator(mode="after")
    def _subjects_exist_in_cast(self) -> "ShootPlan":
        known = self.person_keys()
        for shot in self.shots:
            unknown = [key for key in shot.subjects if key not in known]
            if unknown:
                raise ValueError(
                    f"shot {shot.node_id} references unknown subject(s) {unknown}; "
                    f"cast has {sorted(known)}"
                )
        return self

    @model_validator(mode="after")
    def _cast_keys_are_unique(self) -> "ShootPlan":
        keys = [entry.person_key for entry in self.cast]
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate cast person_key(s): {sorted(duplicates)}")
        return self

    @model_validator(mode="after")
    def _coverage_matches_shots(self) -> "ShootPlan":
        """`coverage` is derived data; a mismatch means the plan is inconsistent."""
        expected = dict(Counter(shot.moment for shot in self.shots))
        if self.coverage != expected:
            raise ValueError(
                f"coverage {self.coverage} does not match the shot list {expected}; "
                "recompute coverage from shots."
            )
        return self

    @model_validator(mode="after")
    def _no_duplicate_framings(self) -> "ShootPlan":
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        for shot in self.shots:
            signature = shot.signature()
            if signature in seen:
                raise ValueError(
                    f"shot {shot.node_id} duplicates moment/framing/subjects of an "
                    "earlier shot; every node must add a distinct framing."
                )
            seen.add(signature)
        return self

    # --- helpers ---------------------------------------------------------

    def person_keys(self) -> set[str]:
        return {entry.person_key for entry in self.cast}

    def person(self, person_key: str) -> PersonProfile:
        for entry in self.cast:
            if entry.person_key == person_key:
                return entry.person
        raise KeyError(f"no such person_key: {person_key}")

    def wardrobe(self, person_key: str) -> WardrobeProfile | None:
        for entry in self.cast:
            if entry.person_key == person_key:
                return entry.wardrobe
        return None

    def activity_coverage(self) -> Counter[str]:
        """How many shots cover each user-requested activity."""
        return Counter(shot.activity.casefold() for shot in self.shots)

    def person_coverage(self) -> Counter[str]:
        counter: Counter[str] = Counter()
        for shot in self.shots:
            for key in shot.subjects:
                counter[key] += 1
        return counter

    def shot(self, node_id: str) -> ShotSpec:
        for candidate in self.shots:
            if candidate.node_id == node_id:
                return candidate
        raise KeyError(f"no such node_id: {node_id}")

    def with_coverage(self) -> "ShootPlan":
        """Return a copy with `coverage` recomputed from the shot list."""
        return self.model_copy(update={"coverage": dict(Counter(s.moment for s in self.shots))})


# --------------------------------------------------------------------------
# Edits
# --------------------------------------------------------------------------


class EditInstruction(PlanModel):
    """A comment compiled into a delta against one plan node.

    `preserves` is the safety net: it carries identity and venue fields forward
    explicitly, so "make me smile more" cannot silently restyle a dress or move
    the shoot to a different venue (design.md §11.1).
    """

    base_node_id: Annotated[str, Field(pattern=r"^s[0-9]{2,3}$")]
    change: EditChange
    detail: str = Field(min_length=1, max_length=1000)
    preserves: list[Annotated[str, Field(max_length=60)]] = Field(
        default_factory=list, max_length=12
    )
    reference_media_ids: list[str] = Field(default_factory=list, max_length=8)
    new_seed: bool = True
    comment_id: str = Field(default="", max_length=60)

    @field_validator("preserves", "reference_media_ids")
    @classmethod
    def _clean_lists(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            entry = item.strip()
            if entry and entry not in cleaned:
                cleaned.append(entry)
        return cleaned

    @model_validator(mode="after")
    def _default_preserves_identity(self) -> "EditInstruction":
        """Every edit preserves identity unless it explicitly adds/removes someone."""
        if self.change in {"add_person", "remove_person", "regenerate"}:
            return self
        if not any(item.casefold() in {"identity", "wardrobe", "venue"} for item in self.preserves):
            self.preserves = [*self.preserves, "identity", "wardrobe", "venue"]
        return self

    def preserves_field(self, name: str) -> bool:
        needle = name.casefold()
        return any(item.casefold() == needle for item in self.preserves)


__all__ = [
    "CameraSpec",
    "EditChange",
    "EditInstruction",
    "Framing",
    "ModelRef",
    "Phase",
    "PlanModel",
    "ShotSpec",
    "ShootPlan",
]
