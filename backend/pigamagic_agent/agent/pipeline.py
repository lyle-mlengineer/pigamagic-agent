"""End-to-end agent pipeline (design.md §3).

Chains the stages that development so far has proven separately:

    reference images -> analyse -> cast + scene hints
                     -> venue research
                     -> compile plan
                     -> assemble prompts

The pipeline is deliberately synchronous-in, artifact-out: one call produces a
single JSON document containing the plan, the prompts, and the provenance. That
document is the unit of reproducibility — it is what the shell stores, what the
CLI prints, and what tests diff.

Everything runs against fixture providers by default (`uses_fixtures()`), so the
whole thing is provable offline and free. Real adapters are injected at Steps 8,
9 and 15 without changing this signature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pigamagic_agent.agent import PROMPT_VERSION
from pigamagic_agent.agent.analyze import AnalysisOutcome, ReferenceImage, analyze_references
from pigamagic_agent.agent.compile import CompileResult, ShootBrief, compile_plan
from pigamagic_agent.agent.config import AgentSettings, get_settings
from pigamagic_agent.agent.conventions import ConventionPack, resolve_event
from pigamagic_agent.agent.hashing import canonical_json, sha256_bytes
from pigamagic_agent.agent.plan import ShootPlan
from pigamagic_agent.agent.profiles import VenueProfile
from pigamagic_agent.agent.prompts import AssembledPrompt, assemble_for_plan
from pigamagic_agent.agent.providers.base import (
    LLMProvider,
    ProviderError,
    VisionProvider,
    WebProvider,
)
from pigamagic_agent.agent.research import research_venue, unreachable_venue_outcome
from pigamagic_agent.agent.safety import SafetyRefusal, check_brief

BRIEFS_DIR = Path(__file__).parent / "data" / "briefs"

PIPELINE_VERSION = "1"


class BriefNotFound(KeyError):
    """Raised when a golden brief name has no matching JSON file."""


def load_brief(name_or_path: str) -> ShootBrief:
    """Load a golden brief by name (`wedding`) or by file path."""
    candidate = BRIEFS_DIR / f"{name_or_path}.json"
    if not candidate.exists():
        candidate = Path(name_or_path)
    if not candidate.exists():
        available = sorted(p.stem for p in BRIEFS_DIR.glob("*.json"))
        raise BriefNotFound(
            f"No brief named {name_or_path!r}. Available: {', '.join(available) or 'none'}"
        )
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["activities"] = tuple(payload.get("activities") or ())
    return ShootBrief(**payload)


def list_briefs() -> list[str]:
    return sorted(path.stem for path in BRIEFS_DIR.glob("*.json"))


@dataclass(frozen=True, slots=True)
class ShootArtifact:
    """Everything one planning run produced, ready to persist or print."""

    brief: ShootBrief
    plan: ShootPlan
    prompts: dict[str, AssembledPrompt]
    validation: dict[str, Any]
    venue_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    used_fallback: bool = False
    pipeline_version: str = PIPELINE_VERSION
    prompt_version: str = PROMPT_VERSION
    #: Content-policy findings. Empty when the gate had nothing to say.
    safety: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "prompt_version": self.prompt_version,
            "brief": {
                "shoot_id": self.brief.shoot_id,
                "title": self.brief.title,
                "event": self.brief.event,
                "venue_name": self.brief.venue_name,
                "activities": list(self.brief.activities),
                "style_preset": self.brief.style_preset,
                "shot_count": self.brief.shot_count,
            },
            "plan": self.plan.model_dump(mode="json"),
            "prompts": {key: value.as_dict() for key, value in self.prompts.items()},
            "validation": self.validation,
            "venue_sources": list(self.venue_sources),
            "warnings": list(self.warnings),
            "used_fallback": self.used_fallback,
            "safety": dict(self.safety),
        }

    def to_json(self) -> str:
        return canonical_json(self.as_dict())

    def digest(self) -> str:
        """Stable fingerprint of the artifact. Two identical runs must match."""
        return sha256_bytes(self.to_json().encode("utf-8"))

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    def summary(self) -> str:
        """Human-readable summary for the CLI."""
        lines = [
            f"Shoot:     {self.brief.title}  ({self.brief.shoot_id})",
            f"Event:     {self.brief.event} at {self.brief.venue_name}",
            f"Shots:     {len(self.plan.shots)}",
        ]
        if self.plan.cast:
            labels = ", ".join(entry.person.label for entry in self.plan.cast)
            lines.append(f"Cast:      {len(self.plan.cast)} ({labels})")
        else:
            lines.append("Cast:      none")

        confidence = f"Confidence:{self.plan.confidence:.2f}"
        if self.used_fallback:
            confidence += "  [fallback plan]"
        lines.append(confidence)

        injected = self.validation.get("injected_nodes") or []
        repaired = f"Repaired:  {self.validation.get('repaired')}"
        if injected:
            repaired += f"  (+{len(injected)} injected)"
        lines.extend([repaired, ""])

        for shot in self.plan.shots:
            subjects = ",".join(shot.subjects)
            lines.append(f"  {shot.node_id} [{shot.framing:<7}] {shot.moment[:34]:<34} <{subjects}>")
        if self.plan.warnings:
            lines.append("")
            lines.append("Plan warnings:")
            lines.extend(f"  - {warning}" for warning in self.plan.warnings)
        if self.warnings:
            lines.append("")
            lines.append("Pipeline warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        lines.append("")
        lines.append(f"Prompts:   {len(self.prompts)}")
        lines.append(f"Digest:    {self.digest()[:16]}")
        return "\n".join(lines)


async def run_pipeline(
    brief: ShootBrief,
    *,
    references: Sequence[ReferenceImage] = (),
    llm: LLMProvider,
    vision: VisionProvider,
    web: WebProvider,
    settings: AgentSettings | None = None,
    pack: ConventionPack | None = None,
    cast_override: Sequence[Any] | None = None,
    venue_override: VenueProfile | None = None,
    prompt_token_budget: int | None = None,
) -> ShootArtifact:
    """Run analyse -> research -> compile -> assemble and return the artifact.

    `cast_override` and `venue_override` exist so tests and the CLI can supply a
    known cast without synthesising reference images. With neither supplied, the
    cast comes from the vision pass and the venue from web research.
    """
    active = settings or get_settings()
    warnings: list[str] = []

    # `brief.event` is whatever a host or a person typed — "Birthday", "my 40th",
    # "Njeri's wedding" — not necessarily a pack key. `get_pack` is exact-key by
    # design, so calling it here made a free-text event kill the whole shoot with
    # `KeyError: no convention pack for 'Birthday'`. `resolve_event` is the
    # entry point that matches aliases and degrades to `generic` with a warning,
    # which is what every host needs.
    if pack is not None:
        conventions = pack
    else:
        resolution = resolve_event(brief.event)
        conventions = resolution.pack
        warnings.extend(resolution.warnings)

    # --- analyse references ------------------------------------------------
    if cast_override is not None:
        cast = list(cast_override)
    elif references:
        try:
            analysis = await analyze_references(references, vision=vision, settings=active)
            cast = analysis.cast
            warnings.extend(analysis.warnings)
            if not cast:
                warnings.append("No people detected in the reference images.")
        except ProviderError as exc:
            # A vision provider that is down or keyless must not turn a shoot
            # into an error page. The cast is then unknown, which the planner
            # already knows how to work without.
            cast = []
            warnings.append(
                f"Reference photos could not be analysed ({exc}); planning without "
                "a known cast, so face consistency will be weaker."
            )
    else:
        cast = []
        warnings.append("No reference images supplied; planning without a known cast.")

    # --- research the venue ------------------------------------------------
    if venue_override is not None:
        venue = venue_override
        venue_sources: list[str] = list(venue.sources)
    else:
        try:
            outcome = await research_venue(
                brief.venue_name, brief.event, web=web, vision=vision, settings=active, pack=conventions
            )
        except ProviderError as exc:
            # Same rule: an unresearchable venue is a warning, not a failure.
            # The honest-ignorance profile is exactly what we would have got had
            # the search simply returned nothing.
            outcome = unreachable_venue_outcome(
                brief.venue_name, brief.event, conventions, reason=str(exc)
            )
        venue = outcome.profile
        venue_sources = list(outcome.sources)
        warnings.extend(outcome.warnings)

    # --- safety gate (before compile: never spend an LLM call on a refusal) --
    verdict = check_brief(brief, cast=cast)
    if verdict.refused:
        raise SafetyRefusal(verdict)
    if verdict.care_flags:
        warnings.extend(item.message for item in verdict.care_flags)

    # --- compile ----------------------------------------------------------
    result: CompileResult = await compile_plan(
        brief, cast=cast, venue=venue, pack=conventions, llm=llm, settings=active
    )

    # The plan is model output, so re-check it. A brief that was clean can still
    # yield a shot we must not render, and this runs before any prompt exists.
    plan_verdict = check_brief(brief, result.plan, cast=cast)
    if plan_verdict.refused:
        raise SafetyRefusal(plan_verdict)
    safety_report = plan_verdict.as_dict()

    # --- assemble ---------------------------------------------------------
    prompts = assemble_for_plan(
        result.plan,
        pack=conventions,
        settings=active,
        max_prompt_tokens=prompt_token_budget,
    )

    return ShootArtifact(
        brief=brief,
        plan=result.plan,
        prompts=prompts,
        validation=result.validation.as_dict(),
        venue_sources=venue_sources,
        warnings=warnings,
        used_fallback=result.used_fallback,
        safety=safety_report,
    )


__all__ = [
    "BRIEFS_DIR",
    "BriefNotFound",
    "PIPELINE_VERSION",
    "ShootArtifact",
    "list_briefs",
    "load_brief",
    "run_pipeline",
]
