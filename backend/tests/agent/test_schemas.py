"""Step 1: profile and plan schemas.

The important properties:
- a fully populated ShootPlan round-trips through JSON byte-stably
- unknown fields are rejected with a readable error
- every Literal value is exercised
- the plan's structural invariants (unique node ids, known subjects, coverage
  agreement, no duplicate framings) actually reject bad data
"""

from collections import Counter
from typing import Any

import pytest
from pydantic import ValidationError

from app.agent.plan import (
    CameraSpec,
    EditInstruction,
    ModelRef,
    ShootPlan,
    ShotSpec,
)
from app.agent.profiles import (
    CastEntry,
    FaceFeatures,
    Garment,
    GroupObservation,
    HairFeatures,
    PersonProfile,
    SceneHints,
    VenueProfile,
    WardrobeProfile,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def make_person(key: str = "p1", label: str = "the bride", **overrides: Any) -> PersonProfile:
    payload: dict[str, Any] = {
        "person_key": key,
        "label": label,
        "age_band": "adult",
        "gender_presentation": "feminine",
        "face": FaceFeatures(shape="oval", eyes="wide-set", mouth="full"),
        "hair": HairFeatures(length="long", style="braided updo", colour="black"),
        "build": "average",
        "skin_tone": "warm brown",
        "distinguishing": ["small scar above left brow", "gold studs"],
        "reference_media_ids": [f"media-{key}-a", f"media-{key}-b"],
        "confidence": 0.82,
    }
    payload.update(overrides)
    return PersonProfile(**payload)


def make_wardrobe(key: str = "p1") -> WardrobeProfile:
    return WardrobeProfile(
        person_key=key,
        garments=[
            Garment(name="floor-length gown", colour="ivory", fabric="silk", fit="fitted")
        ],
        accessories=["veil", "pearl earrings"],
        overall_style="formal",
        must_keep=["the ivory gown"],
        avoid=["hats"],
    )


def make_venue(**overrides: Any) -> VenueProfile:
    payload: dict[str, Any] = {
        "venue_key": "sarova stanley hotel nairobi|wedding",
        "name": "The Sarova Stanley",
        "kind": "heritage hotel",
        "indoor_outdoor": "mixed",
        "architecture": ["Georgian facade", "double-height atrium"],
        "materials": ["pale oak", "travertine"],
        "palette": ["ivory", "deep green"],
        "lighting": ["warm uplighting", "north-facing windows"],
        "signature_details": ["the long reflecting pool", "arched colonnade"],
        "layout": "lobby opens onto a courtyard with the pool at its centre",
        "time_of_day_evidence": ["late afternoon sunlight"],
        "confidence": 0.71,
        "sources": ["https://example.test/sarova", "https://example.test/sarova-photos"],
    }
    payload.update(overrides)
    return VenueProfile(**payload)


def make_cast() -> list[CastEntry]:
    return [
        CastEntry(person=make_person("p1", "the bride"), wardrobe=make_wardrobe("p1"), role="primary", position=0),
        CastEntry(
            person=make_person("p2", "the groom", gender_presentation="masculine"),
            wardrobe=make_wardrobe("p2"),
            role="primary",
            position=1,
        ),
    ]


def make_shot(node_id: str, **overrides: Any) -> ShotSpec:
    payload: dict[str, Any] = {
        "node_id": node_id,
        "phase": "ceremony",
        "moment": "vow exchange",
        "activity": "exchanging vows",
        "description": "The couple face each other as the celebrant speaks.",
        "subjects": ["p1", "p2"],
        "framing": "medium",
        "camera": CameraSpec(focal_length_mm=85, aperture_f=1.8, depth_of_field="shallow"),
        "lighting": "soft window light from camera left",
        "venue_anchors": ["arched colonnade"],
        "wardrobe_notes": ["ivory gown", "charcoal suit"],
        "must_include": ["the ring", "joined hands"],
        "must_avoid": ["guests blocking the couple"],
        "style_preset": "editorial",
        "caption": "The vows",
        "why": "The central moment of the ceremony.",
    }
    payload.update(overrides)
    return ShotSpec(**payload)


def make_plan(**overrides: Any) -> ShootPlan:
    shots = [
        make_shot("s00", moment="vow exchange", framing="medium"),
        make_shot("s01", moment="ring exchange", framing="close"),
        make_shot("s02", moment="confetti exit", framing="wide", phase="reception"),
    ]
    payload: dict[str, Any] = {
        "shoot_id": "shoot-123",
        "revision": 1,
        "cast": make_cast(),
        "venue": make_venue(),
        "style_preset": "editorial",
        "shots": shots,
        "coverage": dict(Counter(shot.moment for shot in shots)),
        "warnings": [],
        "compiler": ModelRef(provider="fixture", model="fixture-planner"),
        "confidence": 0.68,
    }
    payload.update(overrides)
    return ShootPlan(**payload)


# --------------------------------------------------------------------------
# Round-trip
# --------------------------------------------------------------------------


def test_fully_populated_plan_round_trips_byte_stably() -> None:
    plan = make_plan()

    first = plan.model_dump_json()
    reloaded = ShootPlan.model_validate_json(first)
    second = reloaded.model_dump_json()

    assert first == second
    # And the object compares equal after the trip.
    assert reloaded == plan


def test_round_trip_preserves_nested_detail() -> None:
    plan = ShootPlan.model_validate_json(make_plan().model_dump_json())

    assert plan.cast[0].person.face.eyes == "wide-set"
    assert plan.cast[0].wardrobe is not None
    assert plan.cast[0].wardrobe.garments[0].colour == "ivory"
    assert plan.venue.signature_details[0] == "the long reflecting pool"
    assert plan.shots[0].camera.aperture_f == 1.8


# --------------------------------------------------------------------------
# Unknown fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "payload", "bad_field"),
    [
        (VenueProfile, lambda: {"venue_key": "k", "roof_colour": "red"}, "roof_colour"),
        (PersonProfile, lambda: {"person_key": "p1", "height_cm": 170}, "height_cm"),
        (Garment, lambda: {"name": "gown", "brand": "acme"}, "brand"),
        (ShotSpec, lambda: {"node_id": "s00", "moment": "m", "activity": "a", "subjects": ["p1"], "lens": "85"}, "lens"),
        (CameraSpec, lambda: {"iso": 400}, "iso"),
        (WardrobeProfile, lambda: {"person_key": "p1", "size": "M"}, "size"),
    ],
)
def test_unknown_fields_are_rejected_with_the_field_name(
    model: type, payload: Any, bad_field: str
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        model(**payload())
    assert bad_field in str(excinfo.value)


# --------------------------------------------------------------------------
# Literal coverage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("age_band", ["child", "teen", "adult", "senior"])
def test_age_band_values(age_band: str) -> None:
    assert make_person(age_band=age_band).age_band == age_band


@pytest.mark.parametrize("presentation", ["masculine", "feminine", "androgynous", "unspecified"])
def test_gender_presentation_values(presentation: str) -> None:
    assert make_person(gender_presentation=presentation).gender_presentation == presentation


@pytest.mark.parametrize("build", ["slim", "average", "athletic", "curvy", "plus", "unspecified"])
def test_build_values(build: str) -> None:
    assert make_person(build=build).build == build


@pytest.mark.parametrize(
    "style", ["formal", "smart-casual", "casual", "traditional", "athletic", "festive"]
)
def test_overall_style_values(style: str) -> None:
    assert make_wardrobe().model_copy(update={"overall_style": style}).overall_style == style


@pytest.mark.parametrize("indoor_outdoor", ["indoor", "outdoor", "mixed"])
def test_indoor_outdoor_values(indoor_outdoor: str) -> None:
    assert make_venue(indoor_outdoor=indoor_outdoor).indoor_outdoor == indoor_outdoor


@pytest.mark.parametrize(
    "framing", ["wide", "medium", "close", "detail", "over-shoulder", "candid"]
)
def test_framing_values(framing: str) -> None:
    # Distinct moments so the duplicate-framing invariant is not tripped.
    shot = make_shot("s00", framing=framing, moment=f"moment-{framing}")
    assert shot.framing == framing


@pytest.mark.parametrize("angle", ["eye-level", "low", "high", "overhead", "shoulder"])
def test_camera_angle_values(angle: str) -> None:
    assert CameraSpec(angle=angle).angle == angle


@pytest.mark.parametrize("dof", ["shallow", "medium", "deep"])
def test_depth_of_field_values(dof: str) -> None:
    assert CameraSpec(depth_of_field=dof).depth_of_field == dof


@pytest.mark.parametrize("role", ["primary", "guest"])
def test_cast_role_values(role: str) -> None:
    assert CastEntry(person=make_person(), role=role).role == role


@pytest.mark.parametrize(
    "change",
    [
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
    ],
)
def test_edit_change_values(change: str) -> None:
    instruction = EditInstruction(base_node_id="s00", change=change, detail="darker")
    assert instruction.change == change


def test_invalid_literal_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_shot("s00", framing="extreme-closeup")


# --------------------------------------------------------------------------
# Confidence bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_confidence_is_bounded(bad: float) -> None:
    with pytest.raises(ValidationError):
        make_person(confidence=bad)
    with pytest.raises(ValidationError):
        make_venue(confidence=bad)
    with pytest.raises(ValidationError):
        make_plan(confidence=bad)


def test_confidence_boundaries_are_accepted() -> None:
    assert make_person(confidence=0.0).confidence == 0.0
    assert make_person(confidence=1.0).confidence == 1.0


# --------------------------------------------------------------------------
# Profiles: None means "not observed"
# --------------------------------------------------------------------------


def test_unobserved_fields_default_to_none_not_empty_string() -> None:
    person = PersonProfile(person_key="p1")
    assert person.face.eyes is None
    assert person.hair.colour is None
    assert person.skin_tone is None
    assert person.face.observed_field_count() == 0
    assert person.hair.observed_field_count() == 0


def test_observed_field_count_tracks_visible_fields() -> None:
    face = FaceFeatures(shape="oval", eyes="wide-set")
    assert face.observed_field_count() == 2


def test_person_without_reference_images_is_not_identity_anchored() -> None:
    assert PersonProfile(person_key="p1").is_identity_anchored() is False
    assert make_person("p1").is_identity_anchored() is True


def test_person_key_pattern_is_enforced() -> None:
    with pytest.raises(ValidationError):
        PersonProfile(person_key="person-1")
    with pytest.raises(ValidationError):
        PersonProfile(person_key="P1")


@pytest.mark.parametrize("raw", ["@bride", "  bride  "])
def test_label_is_normalised(raw: str) -> None:
    assert make_person(label=raw).label == "bride"


def test_lists_are_cleaned_and_deduplicated_case_insensitively() -> None:
    person = make_person(distinguishing=["glasses", "Glasses", " beard ", ""])
    assert person.distinguishing == ["glasses", "beard"]


def test_garment_requires_a_name() -> None:
    with pytest.raises(ValidationError):
        Garment(name="")


# --------------------------------------------------------------------------
# Venue consistency
# --------------------------------------------------------------------------


def test_confident_venue_without_detail_is_rejected() -> None:
    """A confident profile with no distinguishing detail must not exist."""
    with pytest.raises(ValidationError) as excinfo:
        make_venue(architecture=[], signature_details=[], confidence=0.5)
    assert "confidence" in str(excinfo.value)


def test_low_confidence_venue_may_have_no_detail() -> None:
    profile = make_venue(architecture=[], signature_details=[], confidence=0.2, sources=[])
    assert profile.has_research() is False


def test_venue_with_sources_has_research() -> None:
    assert make_venue().has_research() is True


# --------------------------------------------------------------------------
# Cast and wardrobe pairing
# --------------------------------------------------------------------------


def test_wardrobe_person_key_must_match_person() -> None:
    with pytest.raises(ValidationError) as excinfo:
        CastEntry(person=make_person("p1"), wardrobe=make_wardrobe("p2"))
    assert "person_key" in str(excinfo.value)


def test_cast_person_key_property() -> None:
    assert CastEntry(person=make_person("p7")).person_key == "p7"


# --------------------------------------------------------------------------
# ShotSpec invariants
# --------------------------------------------------------------------------


def test_shot_requires_at_least_one_subject() -> None:
    with pytest.raises(ValidationError):
        make_shot("s00", subjects=[])


def test_shot_subjects_must_be_person_keys() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_shot("s00", subjects=["alice"])
    assert "person key" in str(excinfo.value)


def test_shot_subjects_are_deduplicated() -> None:
    assert make_shot("s00", subjects=["p1", "p1", "p2"]).subjects == ["p1", "p2"]


def test_node_id_pattern_is_enforced() -> None:
    with pytest.raises(ValidationError):
        make_shot("node-1")


def test_shot_signature_is_order_insensitive_over_subjects() -> None:
    a = make_shot("s00", subjects=["p1", "p2"])
    b = make_shot("s00", subjects=["p2", "p1"])
    assert a.signature() == b.signature()


def test_camera_defaults_are_sensible() -> None:
    camera = CameraSpec()
    assert camera.focal_length_mm == 50
    assert camera.angle == "eye-level"
    assert camera.depth_of_field == "medium"


# --------------------------------------------------------------------------
# ShootPlan invariants
# --------------------------------------------------------------------------


def test_plan_rejects_duplicate_node_ids() -> None:
    shots = [make_shot("s00", moment="m1"), make_shot("s00", moment="m2")]
    with pytest.raises(ValidationError) as excinfo:
        make_plan(shots=shots, coverage={"m1": 1, "m2": 1})
    assert "duplicate node_id" in str(excinfo.value)


def test_plan_rejects_unknown_subject() -> None:
    shots = [make_shot("s00", subjects=["p1", "p9"])]
    with pytest.raises(ValidationError) as excinfo:
        make_plan(shots=shots, coverage={"vow exchange": 1})
    assert "unknown subject" in str(excinfo.value)


def test_plan_rejects_duplicate_cast_keys() -> None:
    # Shots must reference only p1, so the duplicate-cast check is what fires
    # rather than the earlier unknown-subject check.
    shots = [make_shot("s00", subjects=["p1"])]
    with pytest.raises(ValidationError) as excinfo:
        make_plan(
            cast=[CastEntry(person=make_person("p1")), CastEntry(person=make_person("p1"))],
            shots=shots,
            coverage={"vow exchange": 1},
        )
    assert "duplicate cast" in str(excinfo.value)


def test_plan_rejects_coverage_that_disagrees_with_shots() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_plan(coverage={"vow exchange": 99})
    assert "coverage" in str(excinfo.value)


def test_plan_rejects_duplicate_moment_framing_subject_combination() -> None:
    shots = [
        make_shot("s00", moment="first dance", framing="wide"),
        make_shot("s01", moment="first dance", framing="wide"),
    ]
    with pytest.raises(ValidationError) as excinfo:
        make_plan(shots=shots, coverage={"first dance": 2})
    assert "duplicates moment" in str(excinfo.value)


def test_same_moment_with_different_framing_is_allowed() -> None:
    shots = [
        make_shot("s00", moment="first dance", framing="wide"),
        make_shot("s01", moment="first dance", framing="close"),
    ]
    plan = make_plan(shots=shots, coverage={"first dance": 2})
    assert len(plan.shots) == 2


def test_plan_requires_a_cast_and_at_least_one_shot() -> None:
    with pytest.raises(ValidationError):
        make_plan(cast=[])
    with pytest.raises(ValidationError):
        make_plan(shots=[], coverage={})


# --------------------------------------------------------------------------
# Plan helpers
# --------------------------------------------------------------------------


def test_person_and_wardrobe_lookup() -> None:
    plan = make_plan()
    assert plan.person("p1").label == "the bride"
    assert plan.wardrobe("p1") is not None
    assert plan.wardrobe("p2").person_key == "p2"

    with pytest.raises(KeyError):
        plan.person("p9")
    assert plan.wardrobe("p9") is None


def test_person_keys_and_shot_lookup() -> None:
    plan = make_plan()
    assert plan.person_keys() == {"p1", "p2"}
    assert plan.shot("s01").moment == "ring exchange"
    with pytest.raises(KeyError):
        plan.shot("s99")


def test_coverage_helpers_count_case_insensitively() -> None:
    plan = make_plan()
    assert plan.activity_coverage()["exchanging vows"] == 3
    person_cov = plan.person_coverage()
    assert person_cov["p1"] == 3 and person_cov["p2"] == 3


def test_with_coverage_recomputes_from_shots() -> None:
    plan = make_plan()
    recalculated = plan.with_coverage()
    assert recalculated.coverage == dict(Counter(shot.moment for shot in plan.shots))


# --------------------------------------------------------------------------
# EditInstruction
# --------------------------------------------------------------------------


def test_edit_defaults_preserve_identity_wardrobe_and_venue() -> None:
    instruction = EditInstruction(base_node_id="s00", change="lighting", detail="warmer")
    assert instruction.preserves_field("identity") is True
    assert instruction.preserves_field("wardrobe") is True
    assert instruction.preserves_field("venue") is True


def test_person_changes_do_not_force_identity_preservation() -> None:
    add = EditInstruction(base_node_id="s00", change="add_person", detail="add Ada")
    remove = EditInstruction(base_node_id="s00", change="remove_person", detail="remove Ada")
    assert add.preserves == []
    assert remove.preserves == []


def test_explicit_preserves_are_kept_and_deduplicated() -> None:
    instruction = EditInstruction(
        base_node_id="s00",
        change="subject",
        detail="smile more",
        preserves=["identity", "identity", " venue "],
    )
    assert instruction.preserves == ["identity", "venue"]


def test_edit_requires_detail_and_valid_node() -> None:
    with pytest.raises(ValidationError):
        EditInstruction(base_node_id="s00", change="lighting", detail="")
    with pytest.raises(ValidationError):
        EditInstruction(base_node_id="node", change="lighting", detail="warmer")


def test_edit_reference_media_ids_are_cleaned() -> None:
    instruction = EditInstruction(
        base_node_id="s00",
        change="wardrobe",
        detail="same dress in green",
        reference_media_ids=["m1", " m1 ", "", "m2"],
    )
    assert instruction.reference_media_ids == ["m1", "m2"]


def test_edit_round_trips() -> None:
    instruction = EditInstruction(base_node_id="s01", change="framing", detail="tighter")
    assert EditInstruction.model_validate_json(instruction.model_dump_json()) == instruction


# --------------------------------------------------------------------------
# Scene hints and group observation
# --------------------------------------------------------------------------


def test_scene_hints_and_group_observation_clean_lists() -> None:
    hints = SceneHints(
        group_size=6,
        group_dynamics=["affectionate", "Affectionate", " playful "],
        glimpsed_venue_features=["tiled floor"],
        mood="warm",
    )
    assert hints.group_dynamics == ["affectionate", "playful"]

    group = GroupObservation(size=4, dynamics=["formal", "formal"])
    assert group.dynamics == ["formal"]


def test_group_size_bounds() -> None:
    with pytest.raises(ValidationError):
        SceneHints(group_size=0)
    with pytest.raises(ValidationError):
        GroupObservation(size=1000)
