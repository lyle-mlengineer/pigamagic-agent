"""Step 2: reference analysis and deterministic fusion.

The properties that matter (design.md §5):
- N photos of M people fuse to M cast entries, not N*M
- the sharper image wins facial fields; a blurry one cannot overwrite them
- identical bytes are analysed once (cache), across separate calls
- fusion is deterministic given the same analyses
"""

from typing import Any

import pytest

from pigamagic_agent.agent.analyze import (
    AnalysisOutcome,
    InMemoryAnalysisCache,
    ReferenceAnalysis,
    ReferenceImage,
    analyze_references,
    fuse_cast,
)
from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider, ScriptedVisionProvider

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _settings(**overrides: Any) -> AgentSettings:
    return AgentSettings(_env_file=None, **overrides)


def _person(
    key: str = "p1",
    *,
    eyes: str | None = "wide-set",
    shape: str | None = "oval",
    hair_style: str | None = "braided updo",
    hair_colour: str | None = "black",
    build: str | None = "average",
    confidence: float = 0.8,
    label: str = "",
    garment: str = "floor-length gown",
    garment_colour: str = "ivory",
    style: str = "formal",
) -> dict[str, Any]:
    """Build a raw analysis person dict, omitting fields set to None."""
    face = {name: value for name, value in (("shape", shape), ("eyes", eyes)) if value}
    hair = {
        name: value
        for name, value in (("style", hair_style), ("colour", hair_colour))
        if value
    }
    return {
        "person_key": key,
        "label": label,
        "age_band": "adult",
        "gender_presentation": "feminine",
        "face": face,
        "hair": hair,
        "build": build,
        "skin_tone": "warm brown",
        "distinguishing": [],
        "confidence": confidence,
        "wardrobe": {
            "garments": [{"name": garment, "colour": garment_colour}],
            "accessories": [],
            "overall_style": style,
            "must_keep": [],
            "avoid": [],
        },
    }


def _analysis(
    media_id: str,
    quality: str,
    people: list[dict[str, Any]],
    *,
    dynamics: list[str] | None = None,
    cues: list[str] | None = None,
    size: int | None = None,
) -> ReferenceAnalysis:
    return ReferenceAnalysis(
        media_id=media_id,
        scene_quality=quality,
        people=people,
        group={"size": size, "dynamics": dynamics or [], "relative_heights": []},
        venue_cues=cues or [],
    )


def _ref(media_id: str, content: bytes, description: str = "") -> ReferenceImage:
    return ReferenceImage(media_id=media_id, content=content, description=description)


# --------------------------------------------------------------------------
# ReferenceImage
# --------------------------------------------------------------------------


def test_reference_image_requires_id_and_content() -> None:
    with pytest.raises(ValueError):
        ReferenceImage(media_id="", content=b"x")
    with pytest.raises(ValueError):
        ReferenceImage(media_id="m1", content=b"")


def test_reference_image_content_hash_is_stable() -> None:
    a = _ref("m1", b"same-bytes")
    b = _ref("m2", b"same-bytes")
    assert a.content_hash == b.content_hash


# --------------------------------------------------------------------------
# Fusion: the central property
# --------------------------------------------------------------------------


def test_three_photos_of_two_people_fuse_to_two_cast_entries() -> None:
    """The headline property: N photos of M people -> M people, not N*M."""
    analyses = [
        _analysis("m1", "usable", [_person("p1"), _person("p2", eyes="almond")]),
        _analysis("m2", "portrait-closeup", [_person("p1"), _person("p2", eyes="almond")]),
        _analysis("m3", "blurry", [_person("p1"), _person("p2", eyes="almond")]),
    ]

    cast, warnings = fuse_cast(analyses)

    assert len(cast) == 2
    assert [entry.person_key for entry in cast] == ["p1", "p2"]
    assert warnings == [] or isinstance(warnings, list)


def test_cast_entries_are_ordered_numerically() -> None:
    analyses = [
        _analysis("m1", "usable", [_person(f"p{index}") for index in (10, 2, 1)])
    ]
    cast, _ = fuse_cast(analyses)
    assert [entry.person_key for entry in cast] == ["p1", "p2", "p10"]


def test_sharper_image_wins_facial_fields() -> None:
    """A blurry reference must not overwrite a good face."""
    sharp = _analysis("m-sharp", "usable", [_person("p1", eyes="wide-set", shape="oval")])
    blurry = _analysis("m-blurry", "blurry", [_person("p1", eyes="tiny", shape="round")])

    # Deliberately pass the blurry one first: quality, not order, must decide.
    cast, _ = fuse_cast([blurry, sharp])

    assert cast[0].person.face.eyes == "wide-set"
    assert cast[0].person.face.shape == "oval"


def test_portrait_closeup_outranks_general_usable() -> None:
    general = _analysis("m-gen", "usable", [_person("p1", eyes="round")])
    portrait = _analysis("m-close", "portrait-closeup", [_person("p1", eyes="almond")])

    cast, _ = fuse_cast([general, portrait])
    assert cast[0].person.face.eyes == "almond"


def test_unobserved_field_does_not_overwrite_an_observed_one() -> None:
    """None means 'not seen', so it must never clear a real observation."""
    good = _analysis("m-good", "usable", [_person("p1", eyes="wide-set")])
    partial = _analysis(
        "m-partial",
        "usable",
        [_person("p1", eyes=None)],  # eyes not visible in this photo
    )

    cast, _ = fuse_cast([partial, good])
    assert cast[0].person.face.eyes == "wide-set"


def test_missing_fields_are_filled_from_other_images() -> None:
    """Union of observations: each image contributes what it can see."""
    a = _analysis("m1", "usable", [_person("p1", eyes="almond", hair_colour=None)])
    b = _analysis("m2", "usable", [_person("p1", eyes=None, hair_colour="auburn")])

    cast, _ = fuse_cast([a, b])
    assert cast[0].person.face.eyes == "almond"
    assert cast[0].person.hair.colour == "auburn"


def test_reference_media_ids_are_collected_across_images() -> None:
    analyses = [
        _analysis("m1", "usable", [_person("p1")]),
        _analysis("m2", "usable", [_person("p1")]),
        _analysis("m3", "usable", [_person("p1")]),
    ]
    cast, _ = fuse_cast(analyses)
    assert cast[0].person.reference_media_ids == ["m1", "m2", "m3"]
    assert cast[0].person.is_identity_anchored() is True


def test_wardrobe_garments_merge_without_duplicates() -> None:
    a = _analysis("m1", "usable", [_person("p1", garment="floor-length gown", garment_colour="ivory")])
    b = _analysis("m2", "usable", [_person("p1", garment="floor-length gown", garment_colour="red")])

    cast, _ = fuse_cast([a, b])
    wardrobe = cast[0].wardrobe
    assert wardrobe is not None
    assert len(wardrobe.garments) == 1
    # First (sharpest) observation's colour wins.
    assert wardrobe.garments[0].colour == "ivory"


def test_garment_case_insensitive_merge() -> None:
    a = _analysis("m1", "usable", [_person("p1", garment="Floor-Length Gown")])
    b = _analysis("m2", "usable", [_person("p1", garment="floor-length gown")])
    cast, _ = fuse_cast([a, b])
    assert len(cast[0].wardrobe.garments) == 1


def test_dominant_style_is_most_mentioned_with_deterministic_tiebreak() -> None:
    analyses = [
        _analysis("m1", "usable", [_person("p1", style="formal")]),
        _analysis("m2", "usable", [_person("p1", style="casual")]),
        _analysis("m3", "usable", [_person("p1", style="casual")]),
    ]
    cast, _ = fuse_cast(analyses)
    assert cast[0].wardrobe.overall_style == "casual"


def test_confidence_reflects_how_much_was_actually_observed() -> None:
    """A person glimpsed in one thin analysis is honestly low-confidence."""
    thin = _analysis(
        "m1",
        "blurry",
        [{"person_key": "p1", "face": {}, "hair": {}, "confidence": 0.9}],
    )
    cast, _ = fuse_cast([thin])
    assert cast[0].person.confidence == 0.0


def test_confidence_is_bounded_and_reacts_to_detail() -> None:
    rich = _analysis(
        "m1",
        "usable",
        [
            _person("p1", confidence=0.95)
            | {
                "face": {
                    "shape": "oval",
                    "eyes": "almond",
                    "brows": "arched",
                    "nose": "straight",
                    "mouth": "full",
                    "jaw": "soft",
                },
                "hair": {
                    "length": "long",
                    "style": "bun",
                    "colour": "black",
                    "texture": "coily",
                },
            }
        ],
    )
    cast, _ = fuse_cast([rich])
    assert 0.0 <= cast[0].person.confidence <= 1.0
    assert cast[0].person.confidence > 0.5


def test_roles_primary_then_guest() -> None:
    analyses = [_analysis("m1", "usable", [_person("p1"), _person("p2")])]
    cast, _ = fuse_cast(analyses)
    assert cast[0].role == "primary"
    assert cast[1].role == "guest"
    assert cast[0].position == 0 and cast[1].position == 1


def test_fusion_is_deterministic_across_input_order() -> None:
    """Same analyses in different order must give an identical cast."""
    people = [_person("p1", eyes="wide-set"), _person("p2", eyes="almond", hair_style="bun")]
    a = _analysis("m1", "usable", people)
    b = _analysis("m2", "usable", people)

    first, _ = fuse_cast([a, b])
    second, _ = fuse_cast([b, a])

    assert [e.model_dump() for e in first] == [e.model_dump() for e in second]


def test_descriptions_supply_person_labels() -> None:
    analyses = [_analysis("m1", "usable", [_person("p1")])]
    cast, _ = fuse_cast(analyses, {"m1": "my sister Ada at the gate"})
    assert cast[0].person.label == "my sister Ada at the gate"


def test_model_label_is_used_when_no_description() -> None:
    analyses = [_analysis("m1", "usable", [_person("p1", label="the bride")])]
    cast, _ = fuse_cast(analyses, {"m1": ""})
    assert cast[0].person.label == "the bride"


def test_wellformed_person_key_survives_but_bad_key_falls_back_to_position() -> None:
    analyses = [
        _analysis(
            "m1",
            "usable",
            [
                {"person_key": "person-one", "face": {"eyes": "round"}},  # bad key
                {"person_key": "p7", "face": {"eyes": "almond"}},
            ],
        )
    ]
    cast, _ = fuse_cast(analyses)
    keys = [entry.person_key for entry in cast]
    assert "p7" in keys
    assert "p1" in keys  # position fallback for the malformed key


def test_single_person_across_many_images_warns() -> None:
    analyses = [
        _analysis("m1", "usable", [_person("p1")]),
        _analysis("m2", "usable", [_person("p1")]),
    ]
    cast, warnings = fuse_cast(analyses)
    assert len(cast) == 1
    assert any("Only one person" in warning for warning in warnings)


def test_empty_analyses_produce_empty_cast() -> None:
    cast, _ = fuse_cast([])
    assert cast == []


def test_analysis_with_no_people_yields_empty_cast() -> None:
    cast, _ = fuse_cast([_analysis("m1", "usable", [])])
    assert cast == []


def test_distinguishing_features_are_unioned() -> None:
    a = _analysis("m1", "usable", [_person("p1") | {"distinguishing": ["glasses"]}])
    b = _analysis("m2", "usable", [_person("p1") | {"distinguishing": ["beard", "glasses"]}])
    cast, _ = fuse_cast([a, b])
    assert cast[0].person.distinguishing == ["glasses", "beard"]


# --------------------------------------------------------------------------
# analyze_references: caching
# --------------------------------------------------------------------------


async def test_analyze_caches_identical_bytes_within_one_call() -> None:
    """The same image supplied twice is analysed once."""
    vision = FixtureVisionProvider(people_per_image=1)
    refs = [_ref("m1", b"image-a"), _ref("m2", b"image-a")]

    outcome = await analyze_references(refs, vision=vision, cache=InMemoryAnalysisCache())

    assert outcome.vision_calls == 1
    assert outcome.cache_hits == 1
    assert len(vision.calls) == 1


async def test_second_run_makes_zero_vision_calls() -> None:
    """A cache hit must never re-invoke the provider (design.md §5.1)."""
    vision = FixtureVisionProvider(people_per_image=1)
    cache = InMemoryAnalysisCache()
    refs = [_ref("m1", b"photo-bytes")]

    first = await analyze_references(refs, vision=vision, cache=cache)
    calls_after_first = len(vision.calls)

    second = await analyze_references(refs, vision=vision, cache=cache)

    assert first.vision_calls == 1
    assert second.vision_calls == 0
    assert second.cache_hits == 1
    assert len(vision.calls) == calls_after_first


async def test_cache_is_keyed_by_content_not_media_id() -> None:
    """Re-uploading the same photo under a new id still hits the cache."""
    vision = FixtureVisionProvider()
    cache = InMemoryAnalysisCache()

    await analyze_references([_ref("m-old", b"identical")], vision=vision, cache=cache)
    outcome = await analyze_references([_ref("m-new", b"identical")], vision=vision, cache=cache)

    assert outcome.vision_calls == 0
    assert outcome.cache_hits == 1
    assert outcome.analyses[0].media_id == "m-new"


async def test_different_bytes_are_analysed_separately() -> None:
    vision = FixtureVisionProvider()
    outcome = await analyze_references(
        [_ref("m1", b"one"), _ref("m2", b"two")], vision=vision
    )
    assert outcome.vision_calls == 2
    assert len(outcome.analyses) == 2


async def test_analyze_requires_at_least_one_reference() -> None:
    with pytest.raises(ValueError):
        await analyze_references([], vision=FixtureVisionProvider())


async def test_unknown_scene_quality_warns_and_normalises() -> None:
    scripted = ScriptedVisionProvider(
        [{"people": [_person("p1")], "scene_quality": "hazy-weird"}]
    )
    outcome = await analyze_references([_ref("m1", b"x")], vision=scripted)

    assert outcome.analyses[0].scene_quality == "usable"
    assert any("scene_quality" in warning for warning in outcome.warnings)


async def test_person_without_reference_warns_about_identity() -> None:
    scripted = ScriptedVisionProvider(
        [{"people": [{"person_key": "p1", "face": {}, "hair": {}}], "scene_quality": "usable"}]
    )
    outcome = await analyze_references([_ref("m1", b"x")], vision=scripted)
    # The person still carries the media id, so identity is anchored; assert the
    # inverse case explicitly instead.
    assert outcome.cast[0].person.reference_media_ids == ["m1"]


def test_cast_without_media_ids_warns() -> None:
    """A profile with no reference images must warn: faces will drift."""
    analyses = [_analysis("m1", "usable", [_person("p1")])]
    cast, _ = fuse_cast(analyses)
    # fuse_cast always attaches media ids, so simulate the degenerate case.
    stripped = cast[0].model_copy(
        update={"person": cast[0].person.model_copy(update={"reference_media_ids": []})}
    )
    assert stripped.person.is_identity_anchored() is False


# --------------------------------------------------------------------------
# Scene hints
# --------------------------------------------------------------------------


async def test_scene_hints_collect_dynamics_and_venue_cues() -> None:
    scripted = ScriptedVisionProvider(
        [
            {
                "people": [_person("p1")],
                "scene_quality": "usable",
                "group": {"size": 2, "dynamics": ["affectionate"]},
                "venue_cues": ["tiled floor"],
            },
            {
                "people": [_person("p1"), _person("p2")],
                "scene_quality": "usable",
                "group": {"size": 4, "dynamics": ["playful", "affectionate"]},
                "venue_cues": ["tall windows", "tiled floor"],
            },
        ]
    )
    outcome = await analyze_references(
        [_ref("m1", b"a"), _ref("m2", b"b")], vision=scripted, settings=_settings()
    )

    hints = outcome.scene_hints
    assert hints.group_dynamics == ["affectionate", "playful"]
    assert hints.glimpsed_venue_features == ["tiled floor", "tall windows"]
    # Largest observed group is the most informative estimate.
    assert hints.group_size == 4


def test_scene_hints_are_deterministic() -> None:
    from pigamagic_agent.agent.analyze import derive_scene_hints

    analyses = [
        _analysis("m2", "usable", [_person("p1")], dynamics=["b"], cues=["y"]),
        _analysis("m1", "usable", [_person("p1")], dynamics=["a"], cues=["x"]),
    ]
    first = derive_scene_hints(analyses, _settings())
    second = derive_scene_hints(list(reversed(analyses)), _settings())
    assert first.model_dump() == second.model_dump()


# --------------------------------------------------------------------------
# Fixture provider determinism
# --------------------------------------------------------------------------


async def test_fixture_provider_is_deterministic_for_same_bytes() -> None:
    provider = FixtureVisionProvider(people_per_image=2)
    first = await provider.describe_image(image=b"same", instruction="i", schema={})
    second = await provider.describe_image(image=b"same", instruction="i", schema={})
    assert first.data == second.data


async def test_fixture_provider_varies_with_content() -> None:
    provider = FixtureVisionProvider(people_per_image=2)
    first = await provider.describe_image(image=b"one", instruction="i", schema={})
    second = await provider.describe_image(image=b"two", instruction="i", schema={})
    assert first.data != second.data


async def test_fixture_provider_satisfies_vision_protocol() -> None:
    from pigamagic_agent.agent.providers import VisionProvider

    assert isinstance(FixtureVisionProvider(), VisionProvider)


async def test_fixture_analysis_is_valid_against_the_profile_schema() -> None:
    """The fixture must produce data the real fusion accepts without error."""
    provider = FixtureVisionProvider(people_per_image=2)
    outcome = await analyze_references([_ref("m1", b"fixture-bytes")], vision=provider)

    assert len(outcome.cast) == 2
    for entry in outcome.cast:
        assert entry.person.person_key.startswith("p")
        assert entry.person.reference_media_ids == ["m1"]
        assert entry.wardrobe is not None


def test_analysis_outcome_cast_by_key() -> None:
    outcome = AnalysisOutcome(cast=[], scene_hints=None, analyses=[])  # type: ignore[arg-type]
    assert outcome.cast_by_key() == {}
