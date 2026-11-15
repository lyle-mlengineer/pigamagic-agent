"""Eval harness: measurable assertions over every golden brief (steps.md Step 19).

Four metrics, one report, explicit thresholds:

| metric | question it answers |
| --- | --- |
| **identity** | would this prompt actually hold the person's face? |
| **coverage** | does the plan photograph every person and activity asked for? |
| **prompt stability** | is assembly still deterministic and still carrying its anchors? |
| **cost** | has a prompt change quietly inflated tokens or credits? |

### Why identity is scored structurally, not by a face model

design.md §8.3 scores identity with a **face embedding** compared against the
reference set — and §8.3 plus NFR-9 are explicit that those embeddings are
**ephemeral**: "compared, scored, then discarded. They are not stored."

Two consequences for this harness:

1. It must never persist an embedding. There is no embedding store here at all,
   which is why `IdentityScore` holds only counts and flags.
2. In CI there is no image model and no face model, so a similarity score would
   either be fabricated or skipped. Fabricated numbers are worse than none: a
   gate that always reports 0.94 teaches everyone to ignore it.

So the harness scores the things that *actually* determine face fidelity and that
a prompt change can break:

- **identity references present** — prompt text alone cannot hold a face, so a
  node with no reference images is capped well below threshold.
- **references scoped to the people in frame** — conditioning a two-person photo
  on five faces dilutes every one of them.
- **subject descriptors rendered** — the cast's hair/build/skin-tone description
  reaching the prompt, not silently dropped.
- **the reference block not truncated** — identity data is dropped *first* under a
  token budget, so truncation is an identity event, not just a token event.

That makes the score move for the exact regressions that hurt: drop the
`[SUBJECTS]` section, truncate references, or stop scoping refs per shot, and the
number falls below `IDENTITY_THRESHOLD`. The tests prove each one fails.

### Cost

`CREDITS_PER_PHOTO` is a coach-built constant, so credits alone cannot detect a
bloated prompt. The harness therefore also asserts **tokens per prompt** and
**total tokens per brief** against ceilings, which is what a prompt edit actually
changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .compile import ShootBrief
from .config import AgentSettings
from .conventions import ConventionPack
from .ledger import CREDITS_PER_PHOTO
from .pipeline import ShootArtifact
from .plan import ShootPlan, ShotSpec
from .prompts import AssembledPrompt, CHARS_PER_TOKEN, SECTION_ORDER

EVAL_VERSION = "eval-v1"

#: Identity below this is a failure. Chosen so a shoot with references and
#: descriptors scores ~1.0, while a prompt-only node cannot exceed ~0.55.
IDENTITY_THRESHOLD = 0.80

#: Prompt-stability: a plan must keep every anchor section in every prompt.
REQUIRED_SECTIONS: tuple[str, ...] = ("STYLE", "SCENE", "SUBJECTS", "ACTION", "FRAMING")

#: Token ceilings. Generous but finite: a prompt edit that doubles the text is a
#: regression even if every other metric stays green.
MAX_PROMPT_TOKENS = 900
MAX_BRIEF_TOKENS = 9000

#: Coverage: every person must appear at least this often (compiler guarantees it).
MIN_SHOTS_PER_PERSON = 2
MIN_SHOTS_PER_PRIMARY = 3


class EvalError(ValueError):
    """A brief that cannot be evaluated."""


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(slots=True)
class MetricResult:
    """One metric's outcome, with the reason it failed rather than a bare False."""

    name: str
    score: float
    passed: bool
    detail: str = ""
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "passed": self.passed,
            "detail": self.detail,
            "failures": list(self.failures),
        }


@dataclass(slots=True)
class IdentityScore:
    """Structural identity-anchoring strength. Holds no embedding (NFR-9)."""

    #: Nodes whose identity references are present and correctly scoped.
    anchored_nodes: int = 0
    total_nodes: int = 0
    #: Nodes where the cast description actually reached the prompt.
    described_nodes: int = 0
    #: Nodes whose reference block was truncated to fit the token budget.
    truncated_nodes: list[str] = field(default_factory=list)
    #: Nodes that reference people but carry no identity references at all.
    unreferenced_nodes: list[str] = field(default_factory=list)
    #: Nodes whose identity refs include people not in frame (diluted conditioning).
    overscoped_nodes: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Mean of anchoring strength and description strength, capped by scope.

        Both halves matter and neither can substitute for the other: references
        without a description lose wardrobe and hair detail, and a description
        without references produces a plausible stranger.
        """
        if self.total_nodes == 0:
            return 0.0
        anchored = self.anchored_nodes / self.total_nodes
        described = self.described_nodes / self.total_nodes
        base = 0.5 * anchored + 0.5 * described

        # Truncation is multiplicative: identity data is dropped *first* under a
        # token budget, so a truncated node has lost the anchors entirely.
        if self.truncated_nodes:
            base *= 0.85

        # Dilution is proportional to how much of the reference budget is wasted.
        # Multiplying by a flat 0.9 made a fully diluting shoot still score 1.0 —
        # the penalty has to scale with the damage to mean anything.
        if self.total_nodes and self.overscoped_nodes:
            base *= 1.0 - 0.5 * (len(self.overscoped_nodes) / self.total_nodes)

        return max(0.0, min(1.0, base))

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "anchored_nodes": self.anchored_nodes,
            "described_nodes": self.described_nodes,
            "total_nodes": self.total_nodes,
            "truncated_nodes": list(self.truncated_nodes),
            "unreferenced_nodes": list(self.unreferenced_nodes),
            "overscoped_nodes": list(self.overscoped_nodes),
            "embedding_persisted": False,
        }


@dataclass(slots=True)
class EvalReport:
    """Every metric for one golden brief, plus a single pass/fail verdict."""

    brief: str
    metrics: list[MetricResult] = field(default_factory=list)
    identity: IdentityScore | None = None
    shot_count: int = 0
    eval_version: str = EVAL_VERSION

    @property
    def passed(self) -> bool:
        return all(metric.passed for metric in self.metrics)

    def metric(self, name: str) -> MetricResult | None:
        return next((item for item in self.metrics if item.name == name), None)

    def failures(self) -> list[str]:
        return [
            f"{metric.name}: {failure}"
            for metric in self.metrics
            for failure in metric.failures
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "eval_version": self.eval_version,
            "brief": self.brief,
            "shot_count": self.shot_count,
            "passed": self.passed,
            "metrics": [metric.as_dict() for metric in self.metrics],
            "identity": None if self.identity is None else self.identity.as_dict(),
            "failures": self.failures(),
        }

    def summary(self) -> str:
        lines = [f"{self.brief}: {'PASS' if self.passed else 'FAIL'} ({self.shot_count} shots)"]
        for metric in self.metrics:
            mark = "ok  " if metric.passed else "FAIL"
            lines.append(f"  [{mark}] {metric.name:16} {metric.score:6.3f}  {metric.detail}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _section_present(prompt: str, section: str) -> bool:
    return f"[{section}]" in prompt


def _subjects_have_descriptions(prompt: str, shot: ShotSpec) -> bool:
    """Whether the `[SUBJECTS]` block carries real description, not just labels.

    A prompt that names a person but says nothing about their hair, build or
    wardrobe has no identity detail to render.
    """
    lines = [
        line
        for line in prompt.splitlines()
        if line.strip().startswith("- ") and "(" in line
    ]
    if not lines:
        return False
    # A bare "no profile available" line means fusion produced nothing usable.
    return not any("no profile available" in line for line in lines)


def score_identity(
    artifact: ShootArtifact,
    *,
    references_by_node: Mapping[str, Sequence[str]] | None = None,
) -> IdentityScore:
    """Score identity anchoring across a plan's prompts.

    `references_by_node` maps node_id -> person keys whose reference images are
    attached. When it is absent, every referenced subject is assumed anchored,
    which mirrors the fixture path; pass it to score a real render.
    """
    plan = artifact.plan
    score = IdentityScore(total_nodes=len(plan.shots))

    for shot in plan.shots:
        assembled = artifact.prompts.get(shot.node_id)
        if assembled is None:
            continue

        # `{}` means "no references anywhere", which is a real regression to
        # detect. Only `None` means "caller did not supply this information", so
        # the check must be `is None`, not truthiness.
        scoped = (
            references_by_node.get(shot.node_id, [])
            if references_by_node is not None
            else None
        )
        in_frame = set(shot.subjects)

        if assembled.truncated_reference:
            score.truncated_nodes.append(shot.node_id)

        if scoped is None:
            anchored = bool(shot.subjects)
            overscoped = False
        else:
            scoped_set = set(scoped)
            anchored = in_frame.issubset(scoped_set) and bool(in_frame)
            overscoped = bool(scoped_set - in_frame)
            if not scoped_set:
                score.unreferenced_nodes.append(shot.node_id)

        if anchored:
            score.anchored_nodes += 1
        if overscoped:
            score.overscoped_nodes.append(shot.node_id)
        if _subjects_have_descriptions(assembled.prompt, shot):
            score.described_nodes += 1

    return score


def score_coverage(artifact: ShootArtifact) -> MetricResult:
    """Every person and activity in the brief must actually be photographed."""
    plan = artifact.plan
    failures: list[str] = []

    appearances: dict[str, int] = {entry.person_key: 0 for entry in plan.cast}
    for shot in plan.shots:
        for key in shot.subjects:
            if key in appearances:
                appearances[key] += 1

    for entry in plan.cast:
        need = MIN_SHOTS_PER_PRIMARY if entry.role == "primary" else MIN_SHOTS_PER_PERSON
        got = appearances.get(entry.person_key, 0)
        if got < need:
            failures.append(
                f"{entry.person_key} appears in {got} shot(s), needs >= {need}"
            )

    covered = {shot.activity.casefold() for shot in plan.shots}
    for activity in artifact.brief.activities:
        if activity.casefold() not in covered:
            failures.append(f"activity {activity!r} is never photographed")

    # `coverage` is derived data, so a mismatch means the plan drifted.
    recomputed: dict[str, int] = {}
    for shot in plan.shots:
        recomputed[shot.moment] = recomputed.get(shot.moment, 0) + 1
    if dict(plan.coverage) != recomputed:
        failures.append("plan.coverage does not match the shots it claims to count")

    total = max(1, len(plan.cast) + len(artifact.brief.activities))
    satisfied = total - len(failures)
    return MetricResult(
        name="coverage",
        score=max(0.0, satisfied / total),
        passed=not failures,
        detail=f"{len(plan.cast)} people, {len(artifact.brief.activities)} activities",
        failures=failures,
    )


def score_prompt_stability(artifact: ShootArtifact) -> MetricResult:
    """Every prompt keeps its anchor sections, and the plan is self-consistent."""
    failures: list[str] = []

    for node_id, assembled in artifact.prompts.items():
        for section in REQUIRED_SECTIONS:
            if not _section_present(assembled.prompt, section):
                failures.append(f"{node_id} is missing the [{section}] section")
        if assembled.negative_prompt.strip() == "":
            failures.append(f"{node_id} has an empty negative prompt")
        for section in SECTION_ORDER:
            if section == "REFERENCE":
                continue
            if _section_present(assembled.prompt, section) is False:
                failures.append(f"{node_id} lost section [{section}]")

    missing_prompts = [shot.node_id for shot in artifact.plan.shots if shot.node_id not in artifact.prompts]
    for node_id in missing_prompts:
        failures.append(f"{node_id} has no assembled prompt")

    total = max(1, len(artifact.plan.shots) * (len(REQUIRED_SECTIONS) + 1))
    return MetricResult(
        name="prompt_stability",
        score=max(0.0, (total - len(failures)) / total),
        passed=not failures,
        detail=f"{len(artifact.prompts)} prompt(s), {len(REQUIRED_SECTIONS)} required sections",
        failures=failures,
    )


def score_cost(
    artifact: ShootArtifact,
    *,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    max_brief_tokens: int = MAX_BRIEF_TOKENS,
) -> MetricResult:
    """Token ceilings per prompt and per brief, plus the credit expectation."""
    failures: list[str] = []
    tokens = {
        node_id: assembled.estimated_tokens
        for node_id, assembled in artifact.prompts.items()
    }

    for node_id, count in tokens.items():
        if count > max_prompt_tokens:
            failures.append(
                f"{node_id} uses {count} tokens, over the {max_prompt_tokens} ceiling"
            )

    total = sum(tokens.values())
    if total > max_brief_tokens:
        failures.append(
            f"brief uses {total} tokens, over the {max_brief_tokens} ceiling"
        )

    expected_credits = len(artifact.plan.shots) * CREDITS_PER_PHOTO
    if expected_credits <= 0 and artifact.plan.shots:
        failures.append("a plan with shots must cost at least one credit")

    worst = max(tokens.values()) if tokens else 0
    score = 1.0 if not failures else max(0.0, 1.0 - len(failures) / max(1, len(tokens)))
    return MetricResult(
        name="cost",
        score=score,
        passed=not failures,
        detail=f"{total} tokens total, {worst} worst, {expected_credits} credit(s)",
        failures=failures,
    )


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------


def evaluate(
    artifact: ShootArtifact,
    *,
    references_by_node: Mapping[str, Sequence[str]] | None = None,
    identity_threshold: float = IDENTITY_THRESHOLD,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    max_brief_tokens: int = MAX_BRIEF_TOKENS,
) -> EvalReport:
    """Score one compiled artifact against every threshold."""
    identity = score_identity(artifact, references_by_node=references_by_node)

    identity_failures: list[str] = []
    if identity.score < identity_threshold:
        identity_failures.append(
            f"identity score {identity.score:.3f} is below the {identity_threshold:.2f} threshold"
        )
    if identity.truncated_nodes:
        identity_failures.append(
            "reference notes were truncated (identity data is dropped first): "
            + ", ".join(identity.truncated_nodes)
        )
    if identity.overscoped_nodes:
        identity_failures.append(
            "identity references include people not in frame, diluting conditioning: "
            + ", ".join(identity.overscoped_nodes)
        )
    if identity.unreferenced_nodes:
        identity_failures.append(
            "no identity references attached for: " + ", ".join(identity.unreferenced_nodes)
        )

    identity_metric = MetricResult(
        name="identity",
        score=identity.score,
        passed=not identity_failures,
        detail=f"{identity.anchored_nodes}/{identity.total_nodes} anchored, "
        f"{identity.described_nodes}/{identity.total_nodes} described",
        failures=identity_failures,
    )

    return EvalReport(
        brief=artifact.brief.shoot_id,
        metrics=[
            identity_metric,
            score_coverage(artifact),
            score_prompt_stability(artifact),
            score_cost(
                artifact,
                max_prompt_tokens=max_prompt_tokens,
                max_brief_tokens=max_brief_tokens,
            ),
        ],
        identity=identity,
        shot_count=len(artifact.plan.shots),
    )


def assert_eval(report: EvalReport, *, label: str = "") -> None:
    """Raise with every failure listed, so CI shows all problems at once."""
    if report.passed:
        return
    heading = label or report.brief
    lines = [f"eval FAILED for {heading}:"]
    lines.extend(f"  - {failure}" for failure in report.failures())
    raise EvalError("\n".join(lines))


def brief_plan_identity_refs(plan: ShootPlan) -> dict[str, list[str]]:
    """Per-node reference person keys, scoped to the people in frame.

    This mirrors `render.identity_refs_for`: conditioning a photo on faces that
    are not in it wastes the reference budget and weakens every real face.
    """
    return {shot.node_id: list(shot.subjects) for shot in plan.shots}


def est_tokens(text: str) -> int:
    """Same estimator assembly uses, so cost scoring cannot disagree with it."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


__all__ = [
    "EVAL_VERSION",
    "EvalError",
    "EvalReport",
    "IDENTITY_THRESHOLD",
    "IdentityScore",
    "MAX_BRIEF_TOKENS",
    "MAX_PROMPT_TOKENS",
    "MetricResult",
    "MIN_SHOTS_PER_PERSON",
    "MIN_SHOTS_PER_PRIMARY",
    "REQUIRED_SECTIONS",
    "assert_eval",
    "brief_plan_identity_refs",
    "est_tokens",
    "evaluate",
    "score_cost",
    "score_coverage",
    "score_identity",
    "score_prompt_stability",
]
