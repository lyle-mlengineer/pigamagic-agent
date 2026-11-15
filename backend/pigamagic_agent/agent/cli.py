"""Offline CLI for the agent (design.md §12, Milestone 1 gate).

    python -m pigamagic_agent.agent.cli briefs
    python -m pigamagic_agent.agent.cli plan --brief wedding
    python -m pigamagic_agent.agent.cli plan --brief wedding --out artifact.json --quiet

Runs entirely on fixture providers: no network, no credentials, no cost. This is
the demonstration surface for Milestone 1 — if `plan --brief wedding` prints a
valid shot list with prompts, the agent layer works end to end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pigamagic_agent.agent.config import get_settings
from pigamagic_agent.agent.conventions import get_pack
from pigamagic_agent.agent.pipeline import ShootArtifact, list_briefs, load_brief, run_pipeline
from pigamagic_agent.agent.profiles import (
    CastEntry,
    Garment,
    PersonProfile,
    VenueProfile,
    WardrobeProfile,
)
from pigamagic_agent.agent.providers.llm_fixture import FailingLLMProvider, FixtureLLMProvider
from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider
from pigamagic_agent.agent.safety import SafetyRefusal


def demo_cast(event: str) -> list[CastEntry]:
    """A small stand-in cast, used when no reference photos are supplied.

    Real casts come from the vision pass over uploaded references. The CLI needs
    *something* so the golden briefs exercise multi-subject logic offline.
    """
    people = [
        ("p1", "the guest of honour", "braided updo", "black", "ivory gown", "ivory"),
        ("p2", "their partner", "short fade", "black", "three-piece suit", "charcoal"),
        ("p3", "the youngest sibling", "curly crop", "dark brown", "floral dress", "sage"),
    ]
    cast: list[CastEntry] = []
    for index, (key, label, hair, colour, garment, garment_colour) in enumerate(people):
        cast.append(
            CastEntry(
                person=PersonProfile(
                    person_key=key,
                    label=label,
                    hair={"style": hair, "colour": colour},  # type: ignore[arg-type]
                    confidence=0.78,
                    reference_media_ids=[f"demo-{key}"],
                ),
                wardrobe=WardrobeProfile(
                    person_key=key,
                    garments=[Garment(name=garment, colour=garment_colour)],
                ),
                role="primary" if index < 2 else "guest",
                position=index,
            )
        )
    return cast


def _demo_venue(brief) -> VenueProfile:
    pack = get_pack(brief.event)
    return VenueProfile(
        venue_key=f"{brief.venue_name.casefold()}|{brief.event}",
        name=brief.venue_name,
        kind="venue",
        indoor_outdoor="indoor",
        signature_details=[],
        lighting=list(pack.lighting_notes[:2]),
        confidence=0.3,
        sources=["demo-fixture"],
    )


async def _build(brief_name: str, *, degraded: bool, prompt_budget: int | None) -> ShootArtifact:
    settings = get_settings()
    brief = load_brief(brief_name)

    llm = FailingLLMProvider() if degraded else FixtureLLMProvider()

    return await run_pipeline(
        brief,
        llm=llm,
        vision=FixtureVisionProvider(),
        web=FixtureWebProvider(),
        settings=settings,
        cast_override=demo_cast(brief.event),
        venue_override=_demo_venue(brief),
        prompt_token_budget=prompt_budget,
    )


def cmd_briefs(_: argparse.Namespace) -> int:
    names = list_briefs()
    if not names:
        print("No golden briefs found.", file=sys.stderr)
        return 1
    print("Available briefs:")
    for name in names:
        brief = load_brief(name)
        print(f"  {name:<12} {brief.event:<10} {len(brief.activities)} activities")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the eval harness over every golden brief. Exit 1 if any brief fails.

    This is the CI entry point: `python -m pigamagic_agent.agent.cli eval` is the whole gate,
    and a non-zero exit is what fails the build.
    """
    from pigamagic_agent.agent.eval import brief_plan_identity_refs, evaluate

    names = [args.brief] if args.brief else list_briefs()
    if not names:
        print("No golden briefs found.", file=sys.stderr)
        return 1

    failed = 0
    for name in names:
        try:
            artifact = asyncio.run(
                _build(name, degraded=args.degraded, prompt_budget=args.max_prompt_tokens)
            )
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        report = evaluate(
            artifact,
            references_by_node=brief_plan_identity_refs(artifact.plan),
        )
        if not args.quiet:
            print(report.summary())
        if not report.passed:
            failed += 1

    if failed:
        print(f"\n{failed} of {len(names)} brief(s) FAILED the eval gate.", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"\nAll {len(names)} brief(s) passed.")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    try:
        artifact = asyncio.run(
            _build(args.brief, degraded=args.degraded, prompt_budget=args.max_prompt_tokens)
        )
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.out:
        path = artifact.write(Path(args.out))
        if not args.quiet:
            print(artifact.summary())
            print()
        print(f"Artifact written to {path}")
    elif not args.quiet:
        print(artifact.summary())

    if args.show_prompt:
        first = next(iter(artifact.prompts.values()), None)
        if first is not None:
            print()
            print("--- prompt for the first shot ---")
            print(first.prompt)

    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from pigamagic_agent.agent.providers.registry import build_image_provider  # noqa: PLC0415
    from pigamagic_agent.agent.render import render_plan  # noqa: PLC0415
    from pigamagic_agent.agent.storage import StorageError, build_blob_store  # noqa: PLC0415

    try:
        artifact = asyncio.run(
            _build(args.brief, degraded=args.degraded, prompt_budget=args.max_prompt_tokens)
        )
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except SafetyRefusal as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 4

    settings = get_settings()
    try:
        store = build_blob_store(settings, name=args.store)
    except StorageError as exc:
        print(f"Storage setup failed: {exc}", file=sys.stderr)
        return 3

    try:
        provider = build_image_provider(settings, name=args.provider)
    except Exception as exc:  # noqa: BLE001
        print(f"Image provider setup failed: {exc}", file=sys.stderr)
        return 3

    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=provider,
            store=store,
            settings=settings,
            width=args.width,
            height=args.height,
            max_shots=args.max_shots,
        )
    )

    for warning in outcome.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for failure in outcome.failures:
        print(f"failed {failure['node_id']}: {failure['reason']}", file=sys.stderr)

    if not args.quiet:
        print(f"Rendered {outcome.count} photo(s) via {provider.name} into {store.name}.")
        print(f"Cost: ${outcome.total_cost.usd:.4f} ({outcome.total_cost.credits} credits)")
        for photo in outcome.photos:
            print(f"  {photo.photo_id}  {photo.node_id}  {photo.blob.key}")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(outcome.as_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"Manifest written to {target}")

    return 0 if outcome.complete else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pigamagic_agent.agent.cli",
        description="PlusOne agent — offline planner (fixtures only)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    briefs = sub.add_parser("briefs", help="list the golden briefs")
    briefs.set_defaults(func=cmd_briefs)

    plan = sub.add_parser("plan", help="compile a shoot plan from a golden brief")
    plan.add_argument("--brief", required=True, help="brief name or path to a JSON brief")
    plan.add_argument("--out", help="write the artifact JSON to this path")
    plan.add_argument("--quiet", action="store_true", help="suppress the summary")
    plan.add_argument(
        "--degraded",
        action="store_true",
        help="simulate a planner outage to exercise the fallback path",
    )
    plan.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="enforce a per-prompt token budget (reference notes shrink first)",
    )
    plan.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the assembled prompt for the first shot",
    )
    plan.set_defaults(func=cmd_plan)

    render = sub.add_parser("render", help="render a brief's plan to stored photos")
    render.add_argument("--brief", required=True, help="brief name or path to a JSON brief")
    render.add_argument("--store", default=None, help="local | s3 | backblaze (default: config)")
    render.add_argument("--provider", default=None, help="image provider name (default: config)")
    render.add_argument("--out", help="write the render manifest JSON to this path")
    render.add_argument("--max-shots", type=int, default=None, help="render at most N shots")
    render.add_argument("--width", type=int, default=1024)
    render.add_argument("--height", type=int, default=1024)
    render.add_argument("--quiet", action="store_true", help="suppress the summary")
    render.add_argument(
        "--degraded",
        action="store_true",
        help="simulate a planner outage to exercise the fallback path",
    )
    render.add_argument("--max-prompt-tokens", type=int, default=None)
    render.set_defaults(func=cmd_render)

    eval_parser = sub.add_parser(
        "eval", help="run the eval harness over golden briefs (CI gate)"
    )
    eval_parser.add_argument("--brief", default=None, help="one brief (default: all)")
    eval_parser.add_argument("--quiet", action="store_true", help="only report failures")
    eval_parser.add_argument("--max-prompt-tokens", type=int, default=None)
    eval_parser.add_argument(
        "--degraded",
        action="store_true",
        help="simulate a planner outage to exercise the fallback path",
    )
    eval_parser.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
