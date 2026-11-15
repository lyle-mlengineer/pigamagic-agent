"""Step 7: end-to-end pipeline and CLI (Milestone 1 gate).

The gate is: the whole agent layer runs offline from a golden brief and produces
a reproducible artifact. These tests assert exactly that, plus the degraded path.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pigamagic_agent.agent.cli import main
from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.pipeline import (
    BRIEFS_DIR,
    PIPELINE_VERSION,
    BriefNotFound,
    list_briefs,
    load_brief,
    run_pipeline,
)
from pigamagic_agent.agent.plan import ShootPlan
from pigamagic_agent.agent.providers.llm_fixture import FailingLLMProvider, FixtureLLMProvider
from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider

BACKEND_DIR = Path(__file__).resolve().parents[2]
# Every shipped golden brief. Each one exercises a different convention pack, so
# the parametrized tests below run the whole pipeline once per event type.
GOLDEN_BRIEFS = (
    "anniversary",
    "baby-shower",
    "birthday",
    "corporate",
    "engagement",
    "farewell",
    "festival",
    "graduation",
    "memorial",
    "religious-cultural",
    "reunion",
    "road-trip",
    "wedding",
)


def _settings(**overrides) -> AgentSettings:
    return AgentSettings(_env_file=None, **overrides)


async def _run(brief_name: str, *, llm=None, budget=None):
    from pigamagic_agent.agent.cli import demo_cast, _demo_venue

    brief = load_brief(brief_name)
    return await run_pipeline(
        brief,
        llm=llm or FixtureLLMProvider(),
        vision=FixtureVisionProvider(),
        web=FixtureWebProvider(),
        settings=_settings(),
        cast_override=demo_cast(brief.event),
        venue_override=_demo_venue(brief),
        prompt_token_budget=budget,
    )


# --------------------------------------------------------------------------
# Briefs
# --------------------------------------------------------------------------


def test_golden_briefs_exist() -> None:
    names = list_briefs()
    for name in GOLDEN_BRIEFS:
        assert name in names, f"missing golden brief {name}"


@pytest.mark.parametrize("name", GOLDEN_BRIEFS)
def test_brief_loads_and_is_valid(name: str) -> None:
    brief = load_brief(name)
    assert brief.shoot_id
    assert brief.event
    assert brief.activities, "every golden brief must request at least one activity"
    assert brief.shot_count > 0


def test_unknown_brief_raises_with_helpful_message() -> None:
    with pytest.raises(BriefNotFound) as info:
        load_brief("no-such-brief")
    message = str(info.value)
    assert "no-such-brief" in message
    assert "wedding" in message, "the error should list what is available"


def test_briefs_dir_is_populated() -> None:
    assert BRIEFS_DIR.exists()
    assert len(list(BRIEFS_DIR.glob("*.json"))) >= len(GOLDEN_BRIEFS)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", GOLDEN_BRIEFS)
async def test_pipeline_produces_a_valid_plan(name: str) -> None:
    artifact = await _run(name)

    assert artifact.plan.shots
    assert artifact.prompts
    assert set(artifact.prompts) == {shot.node_id for shot in artifact.plan.shots}
    # Schema invariants must survive the full chain.
    ShootPlan.model_validate(artifact.plan.model_dump(mode="json"))


@pytest.mark.parametrize("name", GOLDEN_BRIEFS)
async def test_pipeline_covers_every_activity(name: str) -> None:
    artifact = await _run(name)
    brief = load_brief(name)
    covered = {shot.activity.casefold() for shot in artifact.plan.shots}
    for activity in brief.activities:
        assert activity.casefold() in covered, f"{activity} uncovered in {name}"


@pytest.mark.parametrize("name", GOLDEN_BRIEFS)
async def test_every_shot_has_a_nonempty_prompt(name: str) -> None:
    artifact = await _run(name)
    for node_id, assembled in artifact.prompts.items():
        assert assembled.prompt.strip(), f"empty prompt for {node_id}"
        assert "[SUBJECTS]" in assembled.prompt
        assert assembled.estimated_tokens > 0


async def test_artifact_is_deterministic() -> None:
    first = await _run("wedding")
    second = await _run("wedding")
    assert first.to_json() == second.to_json()
    assert first.digest() == second.digest()


async def test_artifact_differs_between_briefs() -> None:
    assert (await _run("wedding")).digest() != (await _run("birthday")).digest()


async def test_artifact_round_trips_through_json() -> None:
    artifact = await _run("wedding")
    text = artifact.to_json()
    reloaded = json.loads(text)
    assert reloaded["pipeline_version"] == PIPELINE_VERSION
    assert len(reloaded["plan"]["shots"]) == len(artifact.plan.shots)
    # Re-serialising canonical JSON must be byte-identical.
    from pigamagic_agent.agent.hashing import canonical_json

    assert canonical_json(reloaded) == text


async def test_sensitive_brief_is_treated_as_sensitive() -> None:
    artifact = await _run("memorial")
    assert all(item.metadata["sensitive"] for item in artifact.prompts.values())
    assert any("restrained" in item.prompt.casefold() for item in artifact.prompts.values())


async def test_planner_outage_still_produces_a_usable_artifact() -> None:
    artifact = await _run("wedding", llm=FailingLLMProvider())

    assert artifact.used_fallback is True
    assert artifact.plan.shots
    assert artifact.prompts
    assert artifact.plan.confidence < 0.8
    assert any("convention-based" in warning for warning in artifact.plan.warnings)


async def test_prompt_budget_is_respected_end_to_end() -> None:
    artifact = await _run("wedding", budget=350)
    for assembled in artifact.prompts.values():
        assert assembled.estimated_tokens <= 350


async def test_no_reference_images_is_reported_not_crashed() -> None:
    from pigamagic_agent.agent.cli import _demo_venue

    brief = load_brief("wedding")
    artifact = await run_pipeline(
        brief,
        llm=FixtureLLMProvider(),
        vision=FixtureVisionProvider(),
        web=FixtureWebProvider(),
        settings=_settings(),
        venue_override=_demo_venue(brief),
    )
    assert any("No reference images" in warning for warning in artifact.warnings)
    # Still compiles, using convention-based subjects.
    assert artifact.plan.shots


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_briefs_lists_golden_briefs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["briefs"]) == 0
    out = capsys.readouterr().out
    for name in GOLDEN_BRIEFS:
        assert name in out


def test_cli_plan_succeeds_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--brief", "wedding"]) == 0
    out = capsys.readouterr().out
    assert "Shots:" in out
    assert "Prompts:" in out


def test_cli_plan_writes_an_artifact(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    assert main(["plan", "--brief", "birthday", "--out", str(target), "--quiet"]) == 0
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["plan"]["shots"]


def test_cli_plan_unknown_brief_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--brief", "nope"]) == 2
    assert "nope" in capsys.readouterr().err


def test_cli_degraded_mode_uses_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--brief", "wedding", "--degraded"]) == 0
    out = capsys.readouterr().out
    assert "fallback plan" in out


def test_cli_show_prompt_prints_the_reference_block(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--brief", "wedding", "--show-prompt"]) == 0
    out = capsys.readouterr().out
    assert "[STYLE]" in out
    assert "[NEGATIVE]" in out


def test_cli_runs_as_a_module_subprocess() -> None:
    """The documented invocation must actually work, not just the in-process call."""
    result = subprocess.run(
        [sys.executable, "-m", "pigamagic_agent.agent.cli", "plan", "--brief", "wedding"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "Shots:" in result.stdout


def test_cli_output_is_identical_across_two_runs() -> None:
    def run() -> str:
        result = subprocess.run(
            [sys.executable, "-m", "pigamagic_agent.agent.cli", "plan", "--brief", "wedding"],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    assert run() == run()
