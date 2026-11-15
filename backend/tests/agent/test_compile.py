"""Step 5: plan compiler and validator.

The properties that matter (design.md §7):
- a good planner response compiles untouched (repaired == False)
- an under-covered plan is REPAIRED by injecting shots, not rejected
- a policy-violating shot is dropped with a warning
- identical inputs give identical plans
- a planner outage still yields a valid plan with warnings
"""

from typing import Any

import pytest

from pigamagic_agent.agent.compile import (
    MIN_SHOTS_PER_PERSON,
    STYLE_PRESETS,
    ShootBrief,
    build_brief_payload,
    build_user_prompt,
    compile_plan,
    procedural_plan,
    validate_and_repair,
)
from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.conventions import get_pack
from pigamagic_agent.agent.plan import ShootPlan, ShotSpec
from pigamagic_agent.agent.profiles import CastEntry, Garment, PersonProfile, VenueProfile, WardrobeProfile
from pigamagic_agent.agent.providers.llm_fixture import (
    FailingLLMProvider,
    FixtureLLMProvider,
    ScriptedLLMProvider,
    extract_payload,
)


def _settings(**overrides: Any) -> AgentSettings:
    return AgentSettings(_env_file=None, **overrides)


def _person(key: str = "p1", label: str = "the bride") -> PersonProfile:
    return PersonProfile(
        person_key=key,
        label=label,
        hair={"style": "braided updo", "colour": "black"},  # type: ignore[arg-type]
        reference_media_ids=[f"media-{key}"],
        confidence=0.8,
    )


def _wardrobe(key: str = "p1") -> WardrobeProfile:
    return WardrobeProfile(person_key=key, garments=[Garment(name="ivory gown", colour="ivory")])


def _cast() -> list[CastEntry]:
    return [
        CastEntry(person=_person("p1", "the bride"), wardrobe=_wardrobe("p1"), role="primary", position=0),
        CastEntry(person=_person("p2", "the groom"), wardrobe=_wardrobe("p2"), role="primary", position=1),
    ]


def _venue(confidence: float = 0.7) -> VenueProfile:
    return VenueProfile(
        venue_key="sarova|wedding",
        name="The Sarova Stanley",
        kind="hotel",
        indoor_outdoor="indoor",
        architecture=["atrium", "colonnade"],
        signature_details=["the long reflecting pool"],
        lighting=["warm uplighting"],
        confidence=confidence,
        sources=["https://example.test/sarova"] if confidence > 0.5 else ["no-sources"],
    )


def _brief(**overrides: Any) -> ShootBrief:
    payload: dict[str, Any] = {
        "shoot_id": "shoot-1",
        "title": "Ada & Sam",
        "event": "wedding",
        "venue_name": "The Sarova Stanley",
        "activities": ("exchanging vows", "cutting the cake", "first dance"),
        "story": "A quiet ceremony that turned into a loud party.",
        "style_preset": "editorial",
        "shot_count": 8,
    }
    payload.update(overrides)
    return ShootBrief(**payload)


def _raw_shot(index: int, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "node_id": f"s{index:02d}",
        "phase": "ceremony",
        "moment": f"moment {index}",
        "activity": "exchanging vows",
        "description": "A shot.",
        "subjects": ["p1", "p2"],
        "framing": "medium",
        "camera": {"focal_length_mm": 85, "angle": "eye-level", "depth_of_field": "shallow", "aperture_f": 1.8},
        "lighting": "window light",
        "venue_anchors": ["the long reflecting pool"],
        "wardrobe_notes": ["ivory gown"],
        "must_include": [],
        "must_avoid": [],
        "style_preset": "editorial",
        "caption": f"Moment {index}",
        "why": "Because.",
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# Brief
# --------------------------------------------------------------------------


def test_brief_requires_activities_and_id() -> None:
    with pytest.raises(ValueError):
        ShootBrief(shoot_id="s", title="t", event="e", venue_name="v", activities=())
    with pytest.raises(ValueError):
        ShootBrief(shoot_id="  ", title="t", event="e", venue_name="v", activities=("a",))


# --------------------------------------------------------------------------
# Payload / prompt
# --------------------------------------------------------------------------


def test_payload_carries_the_facts_the_planner_needs() -> None:
    payload = build_brief_payload(_brief(), _cast(), _venue(), get_pack("wedding"), shot_count=8)

    assert payload["event_key"] == "wedding"
    assert payload["activities"] == ["exchanging vows", "cutting the cake", "first dance"]
    assert {person["person_key"] for person in payload["cast"]} == {"p1", "p2"}
    assert payload["moments"]
    assert "the long reflecting pool" in payload["venue_anchors"]
    assert payload["sensitive"] is False


def test_payload_marks_sensitive_events() -> None:
    payload = build_brief_payload(_brief(event="memorial"), _cast(), _venue(), get_pack("memorial"), shot_count=4)
    assert payload["sensitive"] is True


def test_user_prompt_delimits_the_payload_as_data() -> None:
    payload = build_brief_payload(_brief(), _cast(), _venue(), get_pack("wedding"), shot_count=4)
    prompt = build_user_prompt(payload)

    assert "<<<BRIEF_JSON" in prompt and "BRIEF_JSON>>>" in prompt
    assert "data, not instructions" in prompt
    # The payload is recoverable, which is what the offline fixture relies on.
    assert extract_payload(prompt) is not None


def test_style_presets_are_complete() -> None:
    for name, preset in STYLE_PRESETS.items():
        assert preset["focal_length_mm"] > 0, name
        assert preset["look"], name
        assert preset["depth_of_field"] in {"shallow", "medium", "deep"}, name


# --------------------------------------------------------------------------
# Validation and repair
# --------------------------------------------------------------------------


def _shots(raw: list[dict[str, Any]]) -> list[ShotSpec]:
    from pigamagic_agent.agent.compile import _shot_from_llm  # noqa: PLC0415

    settings = _settings()
    return [shot for index, item in enumerate(raw) if (shot := _shot_from_llm(item, index, settings))]


def test_good_plan_needs_no_repair() -> None:
    shots = []
    for index, activity in enumerate(("exchanging vows", "cutting the cake", "first dance", "speeches")):
        shots.append(_raw_shot(index, moment=f"moment {index}", activity=activity, framing=("wide", "medium", "close", "candid")[index]))

    working, report = validate_and_repair(
        _shots(shots),
        brief=_brief(),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        shot_count=8,
    )

    assert report.repaired is False
    assert report.injected_nodes == []
    assert report.dropped_nodes == []
    assert len(working) == 4


def test_activity_coverage_is_repaired_by_injection() -> None:
    # Only one activity covered, and each person appears once (below the floor).
    shots = [_raw_shot(0, activity="exchanging vows")]

    working, report = validate_and_repair(
        _shots(shots),
        brief=_brief(),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        shot_count=8,
    )

    covered = {shot.activity.casefold() for shot in working}
    for activity in _brief().activities:
        assert activity.casefold() in covered, f"{activity} still uncovered"

    assert report.repaired is True
    assert report.injected_nodes


def test_person_coverage_minimum_is_enforced() -> None:
    shots = [_raw_shot(0)]
    working, report = validate_and_repair(
        _shots(shots),
        brief=_brief(activities=("exchanging vows",)),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        shot_count=8,
    )

    for entry in _cast():
        count = sum(1 for shot in working if entry.person_key in shot.subjects)
        assert count >= MIN_SHOTS_PER_PERSON, f"{entry.person_key} covered only {count}x"
    assert report.repaired is True


def test_policy_violating_shot_is_dropped_with_a_warning() -> None:
    shots = [
        _raw_shot(0),
        _raw_shot(1, activity="nude photoshoot", moment="private"),
    ]
    working, report = validate_and_repair(
        _shots(shots),
        brief=_brief(activities=("exchanging vows", "cutting the cake", "first dance")),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        shot_count=8,
    )

    assert any("blocked activity" in warning for warning in report.warnings)
    assert not any("nude" in shot.activity for shot in working)


def test_unknown_subjects_are_dropped_not_crashed() -> None:
    shots = [_raw_shot(0, subjects=["p1", "p9"])]
    working, report = validate_and_repair(
        _shots(shots),
        brief=_brief(activities=("exchanging vows",)),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        shot_count=8,
    )

    assert any("unknown subject" in warning for warning in report.warnings)
    assert all(set(shot.subjects) <= {"p1", "p2"} for shot in working)


def test_duplicate_framings_are_deduplicated() -> None:
    shots = [
        _raw_shot(0, moment="first dance", framing="wide"),
        _raw_shot(1, moment="first dance", framing="wide"),
    ]
    working, report = validate_and_repair(
        _shots(shots),
        brief=_brief(activities=("exchanging vows",)),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        shot_count=8,
    )

    signatures = [shot.signature() for shot in working]
    assert len(signatures) == len(set(signatures))
    assert report.repaired is True


def test_over_long_plan_is_trimmed() -> None:
    shots = [_raw_shot(index, moment=f"m{index}", framing="wide") for index in range(12)]
    working, report = validate_and_repair(
        _shots(shots),
        brief=_brief(shot_count=4),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        shot_count=4,
    )
    assert len(working) <= 4
    assert any("Trimmed" in warning for warning in report.warnings)


def test_repair_never_exceeds_the_shot_limit() -> None:
    working, _ = validate_and_repair(
        _shots([_raw_shot(0)]),
        brief=_brief(shot_count=3),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        shot_count=3,
    )
    assert len(working) <= 3


def test_nodes_are_renumbered_contiguously() -> None:
    working, _ = validate_and_repair(
        _shots([_raw_shot(0), _raw_shot(5, moment="other")]),
        brief=_brief(activities=("exchanging vows",)),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        shot_count=8,
    )
    assert [shot.node_id for shot in working] == [f"s{index:02d}" for index in range(len(working))]


def test_thin_plan_warns() -> None:
    working, report = validate_and_repair(
        _shots([_raw_shot(0)]),
        brief=_brief(activities=("exchanging vows",)),
        cast=[CastEntry(person=_person("p1"), role="guest")],
        venue=_venue(),
        pack=get_pack("wedding"),
        shot_count=2,
    )
    assert any("thin" in warning.lower() or "usable shot" in warning.lower() for warning in report.warnings)


# --------------------------------------------------------------------------
# compile_plan: happy path
# --------------------------------------------------------------------------


async def test_compiles_a_valid_plan_from_the_fixture_planner() -> None:
    result = await compile_plan(
        _brief(),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        llm=FixtureLLMProvider(),
        settings=_settings(),
    )

    plan = result.plan
    assert isinstance(plan, ShootPlan)
    assert plan.shots
    assert result.used_fallback is False
    assert result.validation.repaired is False
    # Schema invariants must hold on the compiled plan.
    plan.model_validate(plan.model_dump())


async def test_compiled_plan_covers_every_activity() -> None:
    result = await compile_plan(
        _brief(),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        llm=FixtureLLMProvider(),
        settings=_settings(),
    )
    covered = {shot.activity.casefold() for shot in result.plan.shots}
    for activity in _brief().activities:
        assert activity.casefold() in covered


async def test_compiled_plan_round_trips_byte_stably() -> None:
    result = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=FixtureLLMProvider(), settings=_settings(),
    )
    first = result.plan.model_dump_json()
    second = ShootPlan.model_validate_json(first).model_dump_json()
    assert first == second


async def test_compilation_is_deterministic() -> None:
    kwargs = dict(cast=_cast(), venue=_venue(), pack=get_pack("wedding"), settings=_settings())
    first = await compile_plan(_brief(), llm=FixtureLLMProvider(), **kwargs)
    second = await compile_plan(_brief(), llm=FixtureLLMProvider(), **kwargs)
    assert first.plan.model_dump() == second.plan.model_dump()


async def test_shot_count_is_capped_by_settings() -> None:
    result = await compile_plan(
        _brief(shot_count=40), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=FixtureLLMProvider(), settings=_settings(max_shots=6),
    )
    assert len(result.plan.shots) <= 6


# --------------------------------------------------------------------------
# compile_plan: repair, fallback, policy
# --------------------------------------------------------------------------


async def test_short_plan_is_repaired_by_injection() -> None:
    result = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=FixtureLLMProvider(mode="short"), settings=_settings(),
    )

    assert result.validation.repaired is True
    assert result.validation.injected_nodes
    covered = {shot.activity.casefold() for shot in result.plan.shots}
    for activity in _brief().activities:
        assert activity.casefold() in covered


async def test_unsafe_plan_has_the_offending_shot_dropped() -> None:
    result = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=FixtureLLMProvider(mode="unsafe"), settings=_settings(),
    )

    assert result.validation.dropped_nodes
    assert any("blocked activity" in warning for warning in result.plan.warnings)


async def test_planner_failure_falls_back_to_a_procedural_plan() -> None:
    llm = FailingLLMProvider()
    result = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=llm, settings=_settings(),
    )

    assert result.used_fallback is True
    assert result.plan.shots, "a fallback plan must still contain shots"
    assert any("convention-based" in warning for warning in result.plan.warnings)
    assert llm.calls == 2, "should try twice before falling back"


async def test_unparseable_planner_output_falls_back() -> None:
    result = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=FixtureLLMProvider(mode="bad_json"), settings=_settings(),
    )
    assert result.used_fallback is True
    assert result.plan.shots


async def test_non_retryable_failure_does_not_retry() -> None:
    llm = FailingLLMProvider(retryable=False)
    result = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=llm, settings=_settings(),
    )
    assert llm.calls == 1
    assert result.used_fallback is True


async def test_fallback_plan_is_lower_confidence_than_a_good_plan() -> None:
    good = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=FixtureLLMProvider(), settings=_settings(),
    )
    fallback = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=FailingLLMProvider(), settings=_settings(),
    )
    assert fallback.plan.confidence < good.plan.confidence


async def test_unresearched_venue_lowers_plan_confidence() -> None:
    good = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(confidence=0.7), pack=get_pack("wedding"),
        llm=FixtureLLMProvider(), settings=_settings(),
    )
    vague = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(confidence=0.15), pack=get_pack("wedding"),
        llm=FixtureLLMProvider(), settings=_settings(),
    )
    assert vague.plan.confidence < good.plan.confidence


async def test_scripted_plan_that_drops_an_activity_is_repaired() -> None:
    """A model that half-follows instructions still produces a usable plan."""
    scripted = ScriptedLLMProvider([{"shots": [_raw_shot(0, activity="exchanging vows")]}])

    result = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=scripted, settings=_settings(),
    )

    covered = {shot.activity.casefold() for shot in result.plan.shots}
    for activity in _brief().activities:
        assert activity.casefold() in covered
    assert result.validation.repaired is True


# --------------------------------------------------------------------------
# Procedural plan
# --------------------------------------------------------------------------


def test_procedural_plan_is_deterministic_and_valid() -> None:
    pack = get_pack("wedding")
    first = procedural_plan(brief=_brief(), cast=_cast(), venue=_venue(), pack=pack, shot_count=6)
    second = procedural_plan(brief=_brief(), cast=_cast(), venue=_venue(), pack=pack, shot_count=6)

    assert [shot.model_dump() for shot in first] == [shot.model_dump() for shot in second]
    assert len(first) == 6
    assert all(shot.subjects for shot in first)


def test_procedural_plan_uses_convention_moments() -> None:
    pack = get_pack("graduation")
    shots = procedural_plan(
        brief=_brief(event="graduation", activities=("tossing caps",)),
        cast=_cast(), venue=_venue(), pack=pack, shot_count=4,
    )
    moments = {shot.moment for shot in shots}
    assert moments <= set(pack.canonical_moments)


# --------------------------------------------------------------------------
# Plan invariants survive compilation
# --------------------------------------------------------------------------


async def test_every_compiled_shot_names_a_real_person() -> None:
    result = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=FixtureLLMProvider(), settings=_settings(),
    )
    known = result.plan.person_keys()
    for shot in result.plan.shots:
        assert set(shot.subjects) <= known


async def test_coverage_matches_shots_after_compilation() -> None:
    result = await compile_plan(
        _brief(), cast=_cast(), venue=_venue(), pack=get_pack("wedding"),
        llm=FixtureLLMProvider(), settings=_settings(),
    )
    recomputed = result.plan.with_coverage()
    assert recomputed.coverage == result.plan.coverage


async def test_brief_with_one_activity_and_one_person_still_compiles() -> None:
    single = [CastEntry(person=_person("p1"), role="primary")]
    result = await compile_plan(
        _brief(activities=("cutting the cake",)),
        cast=single, venue=_venue(), pack=get_pack("wedding"),
        llm=FixtureLLMProvider(), settings=_settings(),
    )
    assert result.plan.shots
    assert all("p1" in shot.subjects for shot in result.plan.shots)


async def test_policy_violation_is_dropped_wherever_it_appears() -> None:
    """Regression: a violation at the tail must be dropped, not silently trimmed.

    Trimming to the shot limit happens after policy filtering, so a violating
    shot must be identified by content rather than by position.
    """
    shots = [_raw_shot(index, moment=f"m{index}", framing="wide") for index in range(3)]
    shots.append(_raw_shot(9, activity="a nude photoshoot", moment="tail"))

    result = await compile_plan(
        _brief(activities=("exchanging vows",)),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        llm=ScriptedLLMProvider([{"shots": shots}]),
        settings=_settings(),
    )

    assert result.validation.dropped_nodes, "the trailing violation must be recorded as dropped"
    assert not any("nude" in shot.activity.casefold() for shot in result.plan.shots)


# --------------------------------------------------------------------------
# Fixture planner: distinct shots (regression)
#
# The fixture must propose `shot_count` genuinely distinct shots. When it does
# not, the validator's dedupe stage silently drops the duplicates, so a plan
# comes back short and a real coverage bug hides behind a fixture artefact.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "activities",
    [
        # Five activities against a five-frame cycle: index%5 advances in lockstep
        # on both, so every repeat used to be byte-identical to its original.
        ("arriving", "greeting family", "a toast", "cutting a cake", "dancing"),
        # Three activities, so the wrap-around arrives quickly.
        ("arrivals", "eulogy", "candle lighting"),
        # Four activities, the wedding shape.
        ("getting ready", "exchanging vows", "cutting the cake", "first dance"),
    ],
)
async def test_fixture_shots_are_all_distinct(activities: tuple[str, ...]) -> None:
    result = await compile_plan(
        _brief(activities=activities, shot_count=9),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        llm=FixtureLLMProvider(),
        settings=_settings(),
    )
    signatures = [shot.signature() for shot in result.plan.shots]
    assert len(signatures) == len(set(signatures)), "fixture emitted duplicate shots"
    assert not [
        warning for warning in result.validation.warnings if "duplicates" in warning
    ], "validator had to drop fixture duplicates"


async def test_fixture_keeps_every_shot_it_proposed() -> None:
    """A short plan must mean a real coverage gap, not dropped duplicates."""
    result = await compile_plan(
        _brief(activities=("arriving", "a toast", "dancing"), shot_count=6),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        llm=FixtureLLMProvider(),
        settings=_settings(max_shots=6),
    )
    assert len(result.plan.shots) == 6


async def test_fixture_repeats_an_activity_with_a_new_frame() -> None:
    """The same activity shot twice must differ in framing or subjects."""
    result = await compile_plan(
        _brief(activities=("dancing",), shot_count=3),
        cast=_cast(),
        venue=_venue(),
        pack=get_pack("wedding"),
        llm=FixtureLLMProvider(),
        settings=_settings(),
    )
    shots = result.plan.shots
    assert len(shots) == 3
    assert len({shot.signature() for shot in shots}) == 3
    # Same activity throughout, so distinctness must come from framing/subjects.
    assert all(shot.activity == "dancing" for shot in shots)
    assert len({shot.framing for shot in shots}) > 1
