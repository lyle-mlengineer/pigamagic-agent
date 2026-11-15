"""Structured profiles: the facts the pipeline reasons about.

These models are the *typed truth* that replaces free-text prompting. Each one
is produced by one stage and consumed by the next:

    vision analysis  ->  PersonProfile, WardrobeProfile, GroupObservation
    venue research   ->  VenueProfile
    either           ->  SceneHints
    fusion           ->  CastEntry

Two conventions hold across every model here:

1. **`None` means "not observed", never "observed as empty".** The vision pass
   is instructed to omit fields it cannot see rather than guess (design.md
   §5.3). Collapsing that into `""` would let an unobserved field silently
   overwrite a good observation during fusion.
2. **Descriptions are visual, never identitarian.** Fields describe what a
   photo shows so it can be re-rendered. Nothing here asserts who someone is,
   their religion, ethnicity or nationality (SR-3, design.md §5.3).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A 0..1 confidence, shared by every profile.
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]

AgeBand = Literal["child", "teen", "adult", "senior"]
GenderPresentation = Literal["masculine", "feminine", "androgynous", "unspecified"]
Build = Literal["slim", "average", "athletic", "curvy", "plus", "unspecified"]
OverallStyle = Literal["formal", "smart-casual", "casual", "traditional", "athletic", "festive"]
IndoorOutdoor = Literal["indoor", "outdoor", "mixed"]

# Short free-text fields are bounded so a runaway vision response cannot blow
# the prompt budget; the limits are generous for a descriptive phrase.
ShortText = Annotated[str, Field(max_length=120)]
DetailText = Annotated[str, Field(max_length=300)]


class ProfileModel(BaseModel):
    """Base for every profile: forbid unknown fields, strip text."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _clean_list(values: list[str]) -> list[str]:
    """Strip entries, drop blanks, and remove order-preserving duplicates."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        item = value.strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


# --------------------------------------------------------------------------
# Person
# --------------------------------------------------------------------------


class FaceFeatures(ProfileModel):
    """Facial description in visual terms only. All fields optional."""

    shape: ShortText | None = None
    eyes: ShortText | None = None
    brows: ShortText | None = None
    nose: ShortText | None = None
    mouth: ShortText | None = None
    jaw: ShortText | None = None

    def observed_field_count(self) -> int:
        """How many fields the vision pass could actually see."""
        return sum(1 for value in self.model_dump().values() if value)


class HairFeatures(ProfileModel):
    length: ShortText | None = None
    style: ShortText | None = None
    colour: ShortText | None = None
    texture: ShortText | None = None

    def observed_field_count(self) -> int:
        return sum(1 for value in self.model_dump().values() if value)


class PersonProfile(ProfileModel):
    """One person as described by the user's reference photos.

    `person_key` is stable within a shoot ("p1", "p2"...) and is what the
    image provider uses to attach identity references (design.md §7.3).
    `label` comes from the user's own words, never inferred from a face.
    """

    person_key: Annotated[str, Field(pattern=r"^p[0-9]{1,3}$")]
    label: ShortText = Field(default="", max_length=120)
    age_band: AgeBand = "adult"
    gender_presentation: GenderPresentation = "unspecified"
    face: FaceFeatures = Field(default_factory=FaceFeatures)
    hair: HairFeatures = Field(default_factory=HairFeatures)
    build: Build = "unspecified"
    skin_tone: ShortText | None = None
    distinguishing: list[ShortText] = Field(default_factory=list, max_length=12)
    reference_media_ids: list[str] = Field(default_factory=list, max_length=8)
    confidence: Confidence = 0.0

    @field_validator("distinguishing")
    @classmethod
    def _clean_distinguishing(cls, value: list[str]) -> list[str]:
        return _clean_list(value)

    @field_validator("reference_media_ids")
    @classmethod
    def _clean_media_ids(cls, value: list[str]) -> list[str]:
        return _clean_list(value)

    @field_validator("label")
    @classmethod
    def _label_without_leading_at(cls, value: str) -> str:
        return value.lstrip("@").strip()

    def is_identity_anchored(self) -> bool:
        """True when at least one reference image can pin this person's face."""
        return bool(self.reference_media_ids)


# --------------------------------------------------------------------------
# Wardrobe
# --------------------------------------------------------------------------


class Garment(ProfileModel):
    name: ShortText = Field(min_length=1)
    colour: ShortText | None = None
    fabric: ShortText | None = None
    fit: ShortText | None = None


class WardrobeProfile(ProfileModel):
    """What a person was wearing. `must_keep` is honoured at render time."""

    person_key: Annotated[str, Field(pattern=r"^p[0-9]{1,3}$")]
    garments: list[Garment] = Field(default_factory=list, max_length=12)
    accessories: list[ShortText] = Field(default_factory=list, max_length=12)
    overall_style: OverallStyle = "casual"
    must_keep: list[ShortText] = Field(default_factory=list, max_length=8)
    avoid: list[ShortText] = Field(default_factory=list, max_length=8)

    @field_validator("accessories", "must_keep", "avoid")
    @classmethod
    def _clean_lists(cls, value: list[str]) -> list[str]:
        return _clean_list(value)


# --------------------------------------------------------------------------
# Venue
# --------------------------------------------------------------------------


class VenueProfile(ProfileModel):
    """What a specific place looks like, from research or from the brief.

    `sources` is the audit trail: support must be able to tell whether a bad
    photo came from a bad source or a bad prompt (design.md §6.1).
    """

    venue_key: str = Field(min_length=1, max_length=200)
    name: str = Field(default="", max_length=200)
    kind: str = Field(default="", max_length=120)
    indoor_outdoor: IndoorOutdoor = "mixed"
    architecture: list[ShortText] = Field(default_factory=list, max_length=12)
    materials: list[ShortText] = Field(default_factory=list, max_length=12)
    palette: list[ShortText] = Field(default_factory=list, max_length=10)
    lighting: list[ShortText] = Field(default_factory=list, max_length=10)
    signature_details: list[DetailText] = Field(default_factory=list, max_length=12)
    layout: DetailText | None = None
    time_of_day_evidence: list[ShortText] = Field(default_factory=list, max_length=8)
    confidence: Confidence = 0.0
    sources: list[str] = Field(default_factory=list, max_length=20)

    @field_validator(
        "architecture",
        "materials",
        "palette",
        "lighting",
        "signature_details",
        "time_of_day_evidence",
        "sources",
    )
    @classmethod
    def _clean_lists(cls, value: list[str]) -> list[str]:
        return _clean_list(value)

    @model_validator(mode="after")
    def _confidence_consistent_with_evidence(self) -> "VenueProfile":
        """A confident profile with no distinguishing detail is a lie.

        Guards design.md §6.1: when research returns nothing, the profile must
        admit it rather than fabricate architecture from the event type.
        """
        if self.confidence >= 0.5 and not (self.architecture or self.signature_details):
            raise ValueError(
                "A venue profile with confidence >= 0.5 must include architecture or "
                "signature_details; lower the confidence instead of inventing detail."
            )
        return self

    def has_research(self) -> bool:
        """True when at least one *real* source backs this profile.

        The `no-sources` marker means "we looked and found nothing", so it must
        not count as evidence. Otherwise an honest low-confidence profile would
        advertise itself as researched.
        """
        real_sources = [source for source in self.sources if source and source != "no-sources"]
        return bool(real_sources) and self.confidence > 0.0


# --------------------------------------------------------------------------
# Scene hints and cast
# --------------------------------------------------------------------------


class GroupObservation(ProfileModel):
    """How the group relates in the supplied photos: closeness and hierarchy."""

    size: int | None = Field(default=None, ge=1, le=100)
    dynamics: list[ShortText] = Field(default_factory=list, max_length=8)
    relative_heights: list[ShortText] = Field(default_factory=list, max_length=8)

    @field_validator("dynamics", "relative_heights")
    @classmethod
    def _clean_lists(cls, value: list[str]) -> list[str]:
        return _clean_list(value)


class SceneHints(ProfileModel):
    """What the reference images revealed about the day beyond the brief."""

    group_size: int | None = Field(default=None, ge=1, le=100)
    group_dynamics: list[ShortText] = Field(default_factory=list, max_length=8)
    clothing_style: ShortText | None = None
    glimpsed_venue_features: list[ShortText] = Field(default_factory=list, max_length=10)
    mood: ShortText | None = None

    @field_validator("group_dynamics", "glimpsed_venue_features")
    @classmethod
    def _clean_lists(cls, value: list[str]) -> list[str]:
        return _clean_list(value)


CastRole = Literal["primary", "guest"]


class CastEntry(ProfileModel):
    """A person as they appear in the shoot: profile + wardrobe + role.

    Fusion output. `role` drives coverage priority in the plan: primaries get
    more shots than guests (design.md §7.2).
    """

    person: PersonProfile
    wardrobe: WardrobeProfile | None = None
    role: CastRole = "guest"
    position: int = Field(default=0, ge=0, le=99)

    @model_validator(mode="after")
    def _wardrobe_belongs_to_person(self) -> "CastEntry":
        if self.wardrobe is not None and self.wardrobe.person_key != self.person.person_key:
            raise ValueError(
                f"Wardrobe person_key {self.wardrobe.person_key!r} does not match "
                f"person_key {self.person.person_key!r}."
            )
        return self

    @property
    def person_key(self) -> str:
        return self.person.person_key


__all__ = [
    "AgeBand",
    "Build",
    "CastEntry",
    "CastRole",
    "Confidence",
    "FaceFeatures",
    "Garment",
    "GenderPresentation",
    "GroupObservation",
    "HairFeatures",
    "IndoorOutdoor",
    "OverallStyle",
    "PersonProfile",
    "ProfileModel",
    "SceneHints",
    "VenueProfile",
    "WardrobeProfile",
]
