"""Step 19 gates: the eval harness must fail loudly on a degraded prompt.

steps.md: *"the harness fails loudly when a prompt change degrades identity score
below threshold."*

The risk with an eval harness is that it reports a comfortable number forever and
everyone learns to ignore it. So every test here either asserts a real property
or **injects the regression** and asserts the gate catches it. The identity tests
in particular construct the exact degradations that hurt face fidelity, rather
than trusting a number that happens to be 1.0.

`NFR-9` — no face embeddings persisted — is asserted directly: the score object
carries counts and flags, and its dict says `embedding_persisted: False`.
"""

from __future__ import annotations

import pytest

from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.eval import (
    IDENTITY_THRESHOLD,
    MAX_BRIEF_TOKENS,
    MAX_PROMPT_TOKENS,
    REQUIRED_SECTIONS,
    EvalError,
    IdentityScore,
    assert_eval,
    brief_plan_identity_refs,
    est_tokens,
    evaluate,
    score_cost,
    score_coverage,
    score_identity,
    score_prompt_stability,
)
from pigamagic_agent.agent.pipeline import list_briefs, load_brief, run_pipeline
from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider
from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider


def _settings() -> AgentSettings:
    return AgentSettings(target_shot_count=8, max_shots=16, image_budget_usd=50.0)


async def _artifact(name: str = "wedding"):
    return await run_pipeline(
        load_brief(name),
        llm=FixtureLLMProvider(),
        vision=FixtureVisionProvider(),
        web=FixtureWebProvider(),
        settings=_settings(),
    )


# --------------------------------------------------------------------------
# The whole harness over every golden brief
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("brief", list_briefs())
async def test_every_golden_brief_passes_every_metric(brief):
    """The CI gate: all 13 briefs, all four metrics, no exceptions."""
    artifact = await _artifact(brief)
    report = evaluate(
        artifact, references_by_node=brief_plan_identity_refs(artifact.plan)
    )
    assert report.passed, "\n".join(report.failures())
    assert report.shot_count >= 5
    assert {m.name for m in report.metrics} == {
        "identity",
        "coverage",
        "prompt_stability",
        "cost",
    }


@pytest.mark.asyncio
async def test_report_serialises_for_ci_output():
    artifact = await _artifact()
    report = evaluate(
        artifact, references_by_node=brief_plan_identity_refs(artifact.plan)
    )
    payload = report.as_dict()
    assert payload["passed"] is True
    assert payload["eval_version"] == "eval-v1"
    assert len(payload["metrics"]) == 4
    assert payload["failures"] == []
    assert payload["identity"]["score"] == 1.0


# --------------------------------------------------------------------------
# GATE: a degraded prompt must fail
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_dropping_identity_references_fails_the_harness():
    """The headline regression: a node with no identity refs cannot hold a face."""
    artifact = await _artifact()
    scoped = brief_plan_identity_refs(artifact.plan)
    stripped = {node_id: [] for node_id in scoped}

    report = evaluate(artifact, references_by_node=stripped)

    assert report.passed is False
    assert report.metric("identity").passed is False
    assert report.identity.score < IDENTITY_THRESHOLD
    assert any("no identity references" in failure for failure in report.failures())
    # The message must name the nodes, or nobody can act on it.
    assert any("s00" in failure for failure in report.failures())


@pytest.mark.asyncio
async def test_gate_truncated_reference_notes_fail_the_harness():
    """Identity data is dropped *first* under a token budget — a truncation event."""
    artifact = await _artifact()
    # A tiny cap forces the reference block to shrink.
    starved = await run_pipeline(
        load_brief("wedding"),
        llm=FixtureLLMProvider(),
        vision=FixtureVisionProvider(),
        web=FixtureWebProvider(),
        settings=_settings(),
        prompt_token_budget=120,
    )
    truncated = [node for node, prompt in starved.prompts.items() if prompt.truncated_reference]
    # Not a skip: if a future fixture stops producing reference notes, this gate
    # would quietly stop testing anything, which is the failure mode a harness
    # must not have.
    assert truncated, "no node truncated — the harness would be testing nothing"

    report = evaluate(
        starved, references_by_node=brief_plan_identity_refs(starved.plan)
    )
    assert report.metric("identity").passed is False
    assert any("truncated" in failure for failure in report.failures())


@pytest.mark.asyncio
async def test_gate_dropping_a_required_section_fails_prompt_stability():
    """Inject the regression: remove the whole [SUBJECTS] block, not just its header."""
    artifact = await _artifact()
    tampered = _clone_with(artifact)
    for assembled in tampered.prompts.values():
        kept: list[str] = []
        in_subjects = False
        for line in assembled.prompt.splitlines():
            if line.startswith("[SUBJECTS]"):
                in_subjects = True
                continue
            if in_subjects and line.startswith("["):
                in_subjects = False
            if not in_subjects:
                kept.append(line)
        assembled.prompt = "\n".join(kept)
        assert "[SUBJECTS]" not in assembled.prompt

    report = evaluate(tampered, references_by_node=brief_plan_identity_refs(tampered.plan))

    stability = report.metric("prompt_stability")
    assert stability.passed is False
    assert any("[SUBJECTS]" in failure for failure in stability.failures)
    # And identity falls too: without the cast description there is no detail.
    assert report.metric("identity").passed is False


@pytest.mark.asyncio
async def test_gate_a_bloated_prompt_fails_the_cost_ceiling():
    """A prompt edit that doubles the text must fail even if nothing else broke."""
    artifact = await _artifact()
    tampered = _clone_with(artifact)
    for assembled in tampered.prompts.values():
        assembled.prompt = assembled.prompt + (" padding" * 400)
        assembled.estimated_tokens = est_tokens(assembled.prompt)

    report = evaluate(tampered, references_by_node=brief_plan_identity_refs(tampered.plan))

    cost = report.metric("cost")
    assert cost.passed is False
    assert any("ceiling" in failure for failure in cost.failures)


@pytest.mark.asyncio
async def test_gate_dropped_activity_fails_coverage():
    artifact = await _artifact()
    tampered = _clone_with(artifact)
    # Remove every shot that photographs the first activity.
    target = artifact.brief.activities[0].casefold()
    tampered.plan.shots = [s for s in tampered.plan.shots if s.activity.casefold() != target]

    report = evaluate(tampered, references_by_node=brief_plan_identity_refs(tampered.plan))
    assert report.metric("coverage").passed is False
    assert any(target in failure for failure in report.metric("coverage").failures)


@pytest.mark.asyncio
async def test_assert_eval_lists_every_failure_at_once():
    """CI should show all problems in one run, not one per push."""
    artifact = await _artifact()
    tampered = _clone_with(artifact)
    for assembled in tampered.prompts.values():
        assembled.prompt = assembled.prompt + (" padding" * 400)
        assembled.estimated_tokens = est_tokens(assembled.prompt)
    tampered.plan.shots = tampered.plan.shots[:1]

    report = evaluate(tampered, references_by_node={})

    with pytest.raises(EvalError) as excinfo:
        assert_eval(report)
    message = str(excinfo.value)
    assert "eval FAILED" in message
    assert message.count("  - ") >= 2, "a CI failure must list every problem"


def _clone_with(artifact):
    """A mutable stand-in exposing the same attributes the metrics read.

    Both `ShootArtifact` and `AssembledPrompt` are frozen (prompt text is a cache
    key, so freezing is right). A test that wants to inject a regression therefore
    swaps in mutable stand-ins rather than fighting the immutability the product
    depends on. The metrics take the artifact structurally, so this is a faithful
    input, not a mock of the logic under test.
    """
    import copy

    class _MutablePrompt:
        def __init__(self, source):
            self.prompt = source.prompt
            self.negative_prompt = source.negative_prompt
            self.reference_notes = source.reference_notes
            self.prompt_version = source.prompt_version
            self.truncated_reference = source.truncated_reference
            self.estimated_tokens = source.estimated_tokens
            self.metadata = dict(source.metadata)

    class _MutableArtifact:
        def __init__(self, source):
            self.brief = source.brief
            self.plan = copy.deepcopy(source.plan)
            self.prompts = {
                node_id: _MutablePrompt(assembled)
                for node_id, assembled in source.prompts.items()
            }
            self.validation = source.validation
            self.venue_sources = source.venue_sources
            self.warnings = source.warnings
            self.used_fallback = source.used_fallback
            self.safety = source.safety

    return _MutableArtifact(artifact)


# --------------------------------------------------------------------------
# Identity scoring in detail
# --------------------------------------------------------------------------


def test_identity_score_penalties_scale_with_damage():
    clean = IdentityScore(anchored_nodes=10, described_nodes=10, total_nodes=10)
    assert clean.score == 1.0

    # Full dilution halves the score: conditioning on off-frame faces weakens
    # every real face, so this must land below the threshold.
    diluted = IdentityScore(
        anchored_nodes=10,
        described_nodes=10,
        total_nodes=10,
        overscoped_nodes=[f"s{i:02d}" for i in range(10)],
    )
    assert diluted.score == pytest.approx(0.5)
    assert diluted.score < IDENTITY_THRESHOLD

    partial = IdentityScore(
        anchored_nodes=10, described_nodes=10, total_nodes=10, overscoped_nodes=["s00", "s01"]
    )
    assert 0.5 < partial.score < 1.0, "partial damage must be partial"

    truncated = IdentityScore(
        anchored_nodes=10, described_nodes=10, total_nodes=10, truncated_nodes=["s00"]
    )
    assert truncated.score == pytest.approx(0.85)
    assert truncated.score < clean.score


def test_identity_score_is_zero_with_no_nodes():
    assert IdentityScore().score == 0.0


def test_identity_score_persists_no_embedding():
    """NFR-9: QA embeddings are ephemeral. The score is counts and flags only."""
    payload = IdentityScore(anchored_nodes=1, total_nodes=1).as_dict()
    assert payload["embedding_persisted"] is False
    assert set(payload) == {
        "score",
        "anchored_nodes",
        "described_nodes",
        "total_nodes",
        "truncated_nodes",
        "unreferenced_nodes",
        "overscoped_nodes",
        "embedding_persisted",
    }
    # No vector, no blob, nothing that could reconstruct a face.
    for value in payload.values():
        assert not isinstance(value, (bytes, bytearray))


@pytest.mark.asyncio
async def test_identity_references_are_scoped_to_the_people_in_frame():
    """`brief_plan_identity_refs` must mirror what the renderer actually sends."""
    from pigamagic_agent.agent.render import identity_refs_for

    artifact = await _artifact()
    scoped = brief_plan_identity_refs(artifact.plan)
    for shot in artifact.plan.shots:
        assert set(scoped[shot.node_id]) == set(shot.subjects)


@pytest.mark.asyncio
async def test_real_prompts_carry_subject_descriptions():
    artifact = await _artifact()
    identity = score_identity(
        artifact, references_by_node=brief_plan_identity_refs(artifact.plan)
    )
    assert identity.described_nodes == identity.total_nodes
    assert identity.anchored_nodes == identity.total_nodes


# --------------------------------------------------------------------------
# Individual metrics
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_reports_real_token_numbers():
    artifact = await _artifact()
    cost = score_cost(artifact)
    assert cost.passed
    total = sum(p.estimated_tokens for p in artifact.prompts.values())
    assert str(total) in cost.detail
    assert total < MAX_BRIEF_TOKENS
    assert max(p.estimated_tokens for p in artifact.prompts.values()) < MAX_PROMPT_TOKENS


@pytest.mark.asyncio
async def test_coverage_verifies_every_person_and_activity():
    artifact = await _artifact()
    coverage = score_coverage(artifact)
    assert coverage.passed
    assert coverage.score == 1.0
    for entry in artifact.plan.cast:
        assert any(entry.person_key in shot.subjects for shot in artifact.plan.shots)


@pytest.mark.asyncio
async def test_prompt_stability_requires_every_anchor_section():
    artifact = await _artifact()
    stability = score_prompt_stability(artifact)
    assert stability.passed
    for assembled in artifact.prompts.values():
        for section in REQUIRED_SECTIONS:
            assert f"[{section}]" in assembled.prompt


@pytest.mark.asyncio
async def test_est_tokens_matches_the_assembly_estimator():
    from pigamagic_agent.agent.prompts import estimate_tokens

    text = "a" * 700
    assert est_tokens(text) == estimate_tokens(text)


# --------------------------------------------------------------------------
# Report surface
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_is_readable():
    artifact = await _artifact("memorial")
    report = evaluate(
        artifact, references_by_node=brief_plan_identity_refs(artifact.plan)
    )
    summary = report.summary()
    assert report.brief in summary
    assert "PASS" in summary
    for metric in ("identity", "coverage", "prompt_stability", "cost"):
        assert metric in summary


@pytest.mark.asyncio
async def test_a_failing_report_says_fail_and_names_whats_wrong():
    artifact = await _artifact()
    report = evaluate(artifact, references_by_node={})
    assert "FAIL" in report.summary()
    assert report.failures()
    assert report.passed is False


# --------------------------------------------------------------------------
# The CI entry point
# --------------------------------------------------------------------------


def test_cli_eval_exits_zero_when_every_brief_passes():
    """`python -m pigamagic_agent.agent.cli eval` is the whole CI gate."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pigamagic_agent.agent.cli", "eval", "--brief", "wedding", "--quiet"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr


def test_cli_eval_exits_nonzero_on_a_degraded_prompt():
    """The gate must fail the build, not warn and continue."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pigamagic_agent.agent.cli",
            "eval",
            "--brief",
            "wedding",
            "--max-prompt-tokens",
            "120",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 1
    assert "FAILED" in result.stderr
