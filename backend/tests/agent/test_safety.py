"""Step 10: pre-generation content policy gate.

The gate must be right in both directions. A false refusal blocks a grieving
family from a memorial slideshow; a false allow spends money on a render we must
then destroy. So these tests check the refusals *and* the neighbouring cases that
must stay allowed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.agent.cli import _demo_venue, demo_cast
from app.agent.compile import ShootBrief
from app.agent.config import AgentSettings
from app.agent.pipeline import load_brief, run_pipeline
from app.agent.providers.image_fixture import FixtureImageProvider
from app.agent.providers.llm_fixture import FixtureLLMProvider
from app.agent.providers.vision_fixture import FixtureVisionProvider
from app.agent.providers.web_fixture import FixtureWebProvider
from app.agent.safety import (
    SAFETY_VERSION,
    SafetyGate,
    SafetyRefusal,
    SafetyVerdict,
    check_brief,
    require_allowed,
)
from app.agent.storage import LocalBlobStore


def _brief(**overrides: Any) -> ShootBrief:
    base: dict[str, Any] = {
        "shoot_id": "test-shoot",
        "title": "A party",
        "event": "wedding",
        "venue_name": "The Garden Pavilion",
        "activities": ("dancing", "toasting"),
    }
    base.update(overrides)
    return ShootBrief(**base)


def _settings() -> AgentSettings:
    return AgentSettings(_env_file=None)


def _artifact(brief: ShootBrief | None = None):
    async def build():
        target = brief or load_brief("wedding")
        settings = _settings()
        return await run_pipeline(
            target,
            llm=FixtureLLMProvider(),
            vision=FixtureVisionProvider(),
            web=FixtureWebProvider(),
            settings=settings,
            cast_override=demo_cast(target.event),
            venue_override=_demo_venue(target),
        )

    return asyncio.run(build())


# --------------------------------------------------------------------------
# Allowed: the happy path must not be obstructed
# --------------------------------------------------------------------------


def test_ordinary_event_is_allowed() -> None:
    verdict = check_brief(_brief())
    assert verdict.allowed
    assert verdict.codes == ()
    assert verdict.reason == ""


@pytest.mark.parametrize("name", ["wedding", "birthday", "memorial"])
def test_golden_briefs_are_allowed(name: str) -> None:
    verdict = check_brief(load_brief(name))
    assert verdict.allowed, f"{name} was refused: {verdict.reason}"


def test_ceremony_activities_are_allowed() -> None:
    """Words that look risky in isolation are ordinary wedding content."""
    verdict = check_brief(
        _brief(
            activities=(
                "underwater kiss",
                "first dance",
                "cutting the cake",
                "confetti throw",
            ),
            story="The couple had begun dancing when the rain stopped.",
        )
    )
    assert verdict.allowed, verdict.reason


def test_word_boundaries_prevent_substring_false_positives() -> None:
    """'gun' inside 'begun', 'nude' inside nothing — must not fire."""
    gate = SafetyGate()
    verdict = gate.check(_brief(story="The speeches had begun and the dancing continued."))
    assert verdict.allowed, "substring matching produced a false refusal"


# --------------------------------------------------------------------------
# Refused: intimate content
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "activity",
    ["nude portrait", "naked group photo", "lingerie shoot", "a sexual pose"],
)
def test_intimate_content_is_refused(activity: str) -> None:
    verdict = check_brief(_brief(activities=(activity,)))
    assert verdict.refused
    assert "sexual_content" in verdict.codes


def test_refusal_message_is_actionable() -> None:
    verdict = check_brief(_brief(activities=("nude portrait",)))
    reason = verdict.reason
    # Names the problem, and tells the user what would work instead.
    assert "nude" in reason.casefold()
    assert "for example" in reason.casefold()
    assert len(reason) > 80, "a refusal must explain itself, not just deny"


# --------------------------------------------------------------------------
# Refused: violence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("activity", ["weapon display", "gun range", "gore effects"])
def test_violence_is_refused(activity: str) -> None:
    verdict = check_brief(_brief(activities=(activity,)))
    assert verdict.refused
    assert "violence" in verdict.codes


def test_impersonation_is_refused() -> None:
    verdict = check_brief(_brief(story="Make the groom look like a famous actor."))
    assert verdict.refused
    assert "impersonation" in verdict.codes


# --------------------------------------------------------------------------
# Minors
# --------------------------------------------------------------------------


def test_child_in_private_context_is_refused() -> None:
    verdict = check_brief(
        _brief(activities=("pool party",)),
        cast=[{"age_band": "child"}],
    )
    assert verdict.refused
    assert "minor_unsupervised_context" in verdict.codes


def test_same_context_without_a_minor_is_allowed() -> None:
    """The context alone is ordinary; it is the combination that refuses."""
    verdict = check_brief(_brief(activities=("pool party",)), cast=[{"age_band": "adult"}])
    assert verdict.allowed, verdict.reason


def test_child_at_a_normal_family_event_is_allowed() -> None:
    verdict = check_brief(
        _brief(event="birthday", activities=("cake cutting", "party games")),
        cast=[{"age_band": "child"}],
    )
    assert verdict.allowed, "children at family events are the product, not the problem"


def test_minor_plus_intimate_content_is_refused() -> None:
    verdict = check_brief(
        _brief(activities=("nude beach",)),
        cast=[{"age_band": "teen"}],
    )
    assert verdict.refused
    assert "minor_sexual_context" in verdict.codes
    assert "teen" in verdict.reason.casefold()


def test_cast_accepts_objects_and_dicts() -> None:
    class Member:
        age_band = "child"

    by_object = check_brief(_brief(activities=("sleepover",)), cast=[Member()])
    by_dict = check_brief(_brief(activities=("sleepover",)), cast=[{"age_band": "child"}])
    assert by_object.refused and by_dict.refused


# --------------------------------------------------------------------------
# Care flags: allowed, but sensitive
# --------------------------------------------------------------------------


@pytest.mark.parametrize("event", ["memorial", "religious-cultural"])
def test_sensitive_events_are_flagged_not_refused(event: str) -> None:
    verdict = check_brief(_brief(event=event))
    assert verdict.allowed, "a memorial must be photographable"
    assert any(flag.code == "sensitive_event" for flag in verdict.care_flags)


def test_care_flags_do_not_count_as_refusals() -> None:
    verdict = check_brief(_brief(story="A quiet funeral with the family at the grave."))
    assert verdict.allowed
    assert verdict.care_flags
    assert verdict.codes == ()


def test_wedding_has_no_care_flags() -> None:
    verdict = check_brief(_brief())
    assert verdict.care_flags == ()


# --------------------------------------------------------------------------
# Verdict object
# --------------------------------------------------------------------------


def test_verdict_serialises() -> None:
    import json

    verdict = check_brief(_brief(activities=("nude portrait",)))
    payload = verdict.as_dict()
    json.dumps(payload)
    assert payload["verdict"] == "refuse"
    assert payload["safety_version"] == SAFETY_VERSION
    assert payload["findings"]


def test_verdict_is_frozen() -> None:
    verdict = check_brief(_brief())
    assert isinstance(verdict, SafetyVerdict)
    with pytest.raises(Exception):
        verdict.verdict = "refuse"  # type: ignore[misc]


def test_safety_refusal_carries_the_verdict() -> None:
    verdict = check_brief(_brief(activities=("nude portrait",)))
    error = SafetyRefusal(verdict)
    assert error.verdict is verdict
    assert "nude" in str(error).casefold()


def test_require_allowed_raises_only_on_refusal() -> None:
    allowed = require_allowed(_brief())
    assert allowed.allowed

    with pytest.raises(SafetyRefusal):
        require_allowed(_brief(activities=("gun range",)))


# --------------------------------------------------------------------------
# Plan-level check: model output is untrusted
# --------------------------------------------------------------------------


def test_plan_is_checked_not_just_the_brief() -> None:
    """A clean brief can still compile to a shot we must not render."""
    from app.agent.plan import ShootPlan

    artifact = _artifact()
    # Inject a bad shot as though the model had produced it.
    bad = artifact.plan.shots[0].model_copy(update={"activity": "nude portrait"})
    plan: ShootPlan = artifact.plan.model_copy(
        update={"shots": [bad, *artifact.plan.shots[1:]]}
    )

    clean = check_brief(artifact.brief)
    assert clean.allowed, "the brief itself is clean"

    dirty = check_brief(artifact.brief, plan)
    assert dirty.refused, "the plan must be gated independently"


# --------------------------------------------------------------------------
# Integration: the gate runs before money is spent
# --------------------------------------------------------------------------


def test_pipeline_refuses_before_compiling() -> None:
    """A refusal must cost nothing: no LLM call, no plan."""

    class CountingLLM(FixtureLLMProvider):
        calls = 0

        async def complete(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            CountingLLM.calls += 1
            return await super().complete(*args, **kwargs)

    brief = _brief(activities=("nude portrait",))
    llm = CountingLLM()

    async def run():
        return await run_pipeline(
            brief,
            llm=llm,
            vision=FixtureVisionProvider(),
            web=FixtureWebProvider(),
            settings=_settings(),
        )

    with pytest.raises(SafetyRefusal):
        asyncio.run(run())
    assert CountingLLM.calls == 0, "the gate must run before the LLM call"


def test_pipeline_records_the_safety_report() -> None:
    artifact = _artifact(load_brief("memorial"))
    assert artifact.safety["verdict"] == "allow"
    assert artifact.safety["care_flags"]
    assert artifact.as_dict()["safety"]["verdict"] == "allow"


def test_pipeline_flags_sensitive_events_as_warnings() -> None:
    artifact = _artifact(load_brief("memorial"))
    assert any("sensitive event" in warning.casefold() for warning in artifact.warnings)


def test_render_plan_refuses_a_polluted_artifact(tmp_path) -> None:
    """render_plan is public and artifacts can be loaded from disk: re-check."""
    artifact = _artifact()
    bad = artifact.plan.shots[0].model_copy(update={"activity": "graphic gore"})
    polluted = type(artifact)(
        brief=artifact.brief,
        plan=artifact.plan.model_copy(update={"shots": [bad, *artifact.plan.shots[1:]]}),
        prompts=artifact.prompts,
        validation=artifact.validation,
    )

    async def run():
        from app.agent.render import render_plan

        return await render_plan(
            polluted,
            provider=FixtureImageProvider(settings=_settings()),
            store=LocalBlobStore(root=tmp_path),
            settings=_settings(),
        )

    with pytest.raises(SafetyRefusal):
        asyncio.run(run())
    assert not list(tmp_path.rglob("*.png")), "nothing may be written on a refusal"


def test_render_plan_still_works_for_allowed_artifacts(tmp_path) -> None:
    from app.agent.render import render_plan

    artifact = _artifact()
    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=FixtureImageProvider(settings=_settings()),
            store=LocalBlobStore(root=tmp_path),
            settings=_settings(),
            max_shots=1,
        )
    )
    assert outcome.count == 1


def test_every_event_pack_has_a_golden_brief() -> None:
    """A convention pack with no brief is untested content."""
    from app.agent.conventions import pack_keys
    from app.agent.pipeline import list_briefs, load_brief

    covered = {load_brief(name).event for name in list_briefs()}
    missing = sorted(set(pack_keys()) - covered - {"generic"})
    assert not missing, f"convention packs with no golden brief: {missing}"
