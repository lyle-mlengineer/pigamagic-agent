"""Render a compiled plan into persisted photographs (design.md §10).

Responsibilities, kept deliberately narrow:

- generate one image per plan node, in order
- persist through a `BlobStore` (content-addressed, immutable)
- record a per-photo entry with lineage, prompt hash and cost
- enforce the budget across the whole run
- never lose a partial run: if shot 7 of 10 fails, the first 6 are still returned

**Delivered photos are immutable.** Nothing here mutates a stored photo. An edit
creates a *new* photo whose `parent_id` points at the original, so a user's feed
can never change under them and every descendant is traceable (SR-4).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pigamagic_agent.agent.config import AgentSettings, get_settings
from pigamagic_agent.agent.hashing import prompt_hash, sha256_bytes
from pigamagic_agent.agent.pipeline import ShootArtifact
from pigamagic_agent.agent.plan import ShotSpec
from pigamagic_agent.agent.prompts import AssembledPrompt
from pigamagic_agent.agent.providers.base import (
    AgentError,
    CostEstimate,
    GeneratedImage,
    ImageProvider,
    ProviderError,
)
from pigamagic_agent.agent.providers.http_base import BudgetExceeded, BudgetLedger
from pigamagic_agent.agent.safety import require_allowed
from pigamagic_agent.agent.storage import BlobStore, StorageError, StoredBlob, utc_now

RENDER_VERSION = "1"

MAX_ATTEMPTS_PER_SHOT = 2


@dataclass(frozen=True, slots=True)
class RenderedPhoto:
    """One persisted photo. The unit the feed shows and the user can act on."""

    photo_id: str
    shoot_id: str
    node_id: str
    blob: StoredBlob
    moment: str
    activity: str
    subjects: tuple[str, ...]
    framing: str
    caption: str
    seed: int | None
    provider: str
    model: str
    prompt_hash: str
    prompt_version: str
    cost: CostEstimate
    width: int
    height: int
    parent_id: str | None = None
    created_at: str = ""
    attempts: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "photo_id": self.photo_id,
            "shoot_id": self.shoot_id,
            "node_id": self.node_id,
            "blob": self.blob.as_dict(),
            "moment": self.moment,
            "activity": self.activity,
            "subjects": list(self.subjects),
            "framing": self.framing,
            "caption": self.caption,
            "seed": self.seed,
            "provider": self.provider,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "prompt_version": self.prompt_version,
            "cost": {"credits": self.cost.credits, "usd": self.cost.usd},
            "width": self.width,
            "height": self.height,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "attempts": self.attempts,
        }


@dataclass(frozen=True, slots=True)
class RenderOutcome:
    """Everything one render run produced, including what failed and why."""

    shoot_id: str
    photos: list[RenderedPhoto] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_cost: CostEstimate = field(default_factory=CostEstimate.zero)
    ledger: dict[str, Any] = field(default_factory=dict)
    render_version: str = RENDER_VERSION
    started_at: str = ""
    finished_at: str = ""

    @property
    def count(self) -> int:
        return len(self.photos)

    @property
    def complete(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "render_version": self.render_version,
            "shoot_id": self.shoot_id,
            "photos": [photo.as_dict() for photo in self.photos],
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "total_cost": {
                "credits": self.total_cost.credits,
                "usd": self.total_cost.usd,
            },
            "ledger": dict(self.ledger),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "complete": self.complete,
        }

    def photo_for(self, node_id: str) -> RenderedPhoto | None:
        return next((photo for photo in self.photos if photo.node_id == node_id), None)

    def digest(self) -> str:
        """Stable fingerprint of the rendered set (content, not timestamps)."""
        material = "|".join(
            f"{photo.node_id}:{photo.blob.sha256}" for photo in self.photos
        )
        return sha256_bytes(material.encode("utf-8"))


def default_seed(shot: ShotSpec, brief_shoot_id: str) -> int:
    """Deterministic per-shot seed.

    Derived from the shot identity, so re-rendering the same plan reproduces the
    same images and a retry is not a lottery.
    """
    digest = sha256_bytes(f"{brief_shoot_id}:{shot.node_id}".encode("utf-8"))
    return int(digest[:8], 16)


def identity_refs_for(
    shot: ShotSpec, reference_images: Mapping[str, Sequence[bytes]] | None
) -> dict[str, list[bytes]] | None:
    """Select the reference photos for the people in this specific shot.

    Only subjects actually in frame are sent, which keeps requests small and
    avoids conditioning a two-person photo on five people's faces.
    """
    if not reference_images:
        return None
    selected: dict[str, list[bytes]] = {}
    for key in shot.subjects:
        images = reference_images.get(key)
        if images:
            selected[key] = list(images)
    return selected or None


async def render_plan(
    artifact: ShootArtifact,
    *,
    provider: ImageProvider,
    store: BlobStore,
    settings: AgentSettings | None = None,
    reference_images: Mapping[str, Sequence[bytes]] | None = None,
    width: int = 1024,
    height: int = 1024,
    max_shots: int | None = None,
    parent_id: str | None = None,
    budget: BudgetLedger | None = None,
    node_ids: Sequence[str] | None = None,
) -> RenderOutcome:
    """Render every shot in a plan to storage.

    `node_ids` restricts the run to specific shots (used for a single-photo
    edit/regenerate). `parent_id` records lineage when re-rendering one photo.
    """
    active = settings or get_settings()

    # Defence in depth. The pipeline already gated this, but render_plan is a
    # public entry point and an artifact can be loaded from disk, so re-check
    # immediately before the first provider call — the expensive step.
    require_allowed(artifact.brief, artifact.plan)

    started = utc_now()

    ledger = budget or BudgetLedger(
        limit_usd=float(getattr(active, "image_budget_usd", 0.0) or 0.0)
    )

    shots = list(artifact.plan.shots)
    if node_ids is not None:
        wanted = set(node_ids)
        shots = [shot for shot in shots if shot.node_id in wanted]
    if max_shots is not None:
        shots = shots[: max(0, max_shots)]

    photos: list[RenderedPhoto] = []
    failures: list[dict[str, Any]] = []
    warnings: list[str] = []
    total = CostEstimate.zero()

    for shot in shots:
        assembled = artifact.prompts.get(shot.node_id)
        if assembled is None:
            failures.append(
                {
                    "node_id": shot.node_id,
                    "reason": "no prompt assembled for this node",
                }
            )
            continue

        try:
            photo, cost, attempts = await _render_one(
                artifact=artifact,
                shot=shot,
                assembled=assembled,
                provider=provider,
                store=store,
                ledger=ledger,
                reference_images=reference_images,
                width=width,
                height=height,
                parent_id=parent_id,
            )
        except BudgetExceeded as exc:
            # Stop the whole run: continuing would only fail the same way.
            warnings.append(
                f"Stopped after {len(photos)} photo(s): {exc}"
            )
            failures.append(
                {
                    "node_id": shot.node_id,
                    "reason": "budget exceeded",
                    "detail": str(exc),
                }
            )
            break
        except (ProviderError, StorageError, AgentError) as exc:
            failures.append(
                {
                    "node_id": shot.node_id,
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
            )
            continue

        photos.append(photo)
        total = total + cost

    finished = utc_now()

    if failures:
        warnings.append(
            f"{len(failures)} of {len(shots)} shot(s) failed; "
            f"{len(photos)} photo(s) are still available."
        )

    return RenderOutcome(
        shoot_id=artifact.brief.shoot_id,
        photos=photos,
        failures=failures,
        warnings=warnings,
        total_cost=total,
        ledger=ledger.as_dict(),
        started_at=started,
        finished_at=finished,
    )


async def _render_one(
    *,
    artifact: ShootArtifact,
    shot: ShotSpec,
    assembled: AssembledPrompt,
    provider: ImageProvider,
    store: BlobStore,
    ledger: BudgetLedger,
    reference_images: Mapping[str, Sequence[bytes]] | None,
    width: int,
    height: int,
    parent_id: str | None,
) -> tuple[RenderedPhoto, CostEstimate, int]:
    """Generate and persist a single photo, with one retry on a transient error."""
    refs = identity_refs_for(shot, reference_images)
    seed = default_seed(shot, artifact.brief.shoot_id)

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS_PER_SHOT + 1):
        try:
            image: GeneratedImage = await provider.generate(
                prompt=assembled.prompt,
                negative_prompt=assembled.negative_prompt,
                identity_refs=refs,
                width=width,
                height=height,
                seed=seed,
            )
        except ProviderError as exc:
            last_error = exc
            # Only retry when the provider itself says it is worth retrying.
            if exc.retryable and attempt < MAX_ATTEMPTS_PER_SHOT:
                await asyncio.sleep(0.05 * attempt)
                continue
            raise
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"unexpected provider failure: {exc}", provider=provider.name
            ) from exc

        ledger.spend(image.cost)

        blob = await store.put(
            image.content,
            content_type=image.content_type,
            prefix=artifact.brief.event or "photos",
        )

        photo = RenderedPhoto(
            photo_id=_photo_id(image.content),
            shoot_id=artifact.brief.shoot_id,
            node_id=shot.node_id,
            blob=blob,
            moment=shot.moment,
            activity=shot.activity,
            subjects=tuple(shot.subjects),
            framing=shot.framing,
            caption=shot.caption,
            seed=image.seed if image.seed is not None else seed,
            provider=image.provider or provider.name,
            model=image.model,
            prompt_hash=prompt_hash(
                resolved_prompt=assembled.prompt,
                negative_prompt=assembled.negative_prompt,
                identity_refs=refs,
                plan_node_id=shot.node_id,
            ),
            prompt_version=assembled.prompt_version,
            cost=image.cost,
            width=image.width,
            height=image.height,
            parent_id=parent_id,
            created_at=utc_now(),
            attempts=attempt,
        )
        return photo, image.cost, attempt

    raise last_error or ProviderError("generation failed", provider=provider.name)


def _photo_id(content: bytes) -> str:
    """Photo id derived from content, so identical bytes share an id.

    A photo is its pixels; an id that changed on every run would create phantom
    duplicates in the feed.
    """
    return f"ph_{sha256_bytes(content)[:24]}"


async def render_edit(
    original: RenderedPhoto,
    artifact: ShootArtifact,
    *,
    instruction: str,
    provider: ImageProvider,
    store: BlobStore,
    settings: AgentSettings | None = None,
    reference_images: Mapping[str, Sequence[bytes]] | None = None,
    width: int = 1024,
    height: int = 1024,
) -> RenderOutcome:
    """Create a child photo from a comment-style edit instruction.

    The original photo is never touched. The child records `parent_id`, so the
    feed can show lineage and a rejected edit can be discarded without harming
    the delivered image (SR-4).
    """
    shot = next(
        (item for item in artifact.plan.shots if item.node_id == original.node_id), None
    )
    if shot is None:
        raise AgentError(f"No plan node {original.node_id!r} to edit from.")

    base_prompt = artifact.prompts.get(original.node_id)
    if base_prompt is None:
        raise AgentError(f"No assembled prompt for node {original.node_id!r}.")

    edited = _apply_instruction(base_prompt, instruction)

    scoped = ShootArtifact(
        brief=artifact.brief,
        plan=artifact.plan,
        prompts={**artifact.prompts, original.node_id: edited},
        validation=artifact.validation,
        venue_sources=artifact.venue_sources,
        warnings=artifact.warnings,
        used_fallback=artifact.used_fallback,
    )

    return await render_plan(
        scoped,
        provider=provider,
        store=store,
        settings=settings,
        reference_images=reference_images,
        width=width,
        height=height,
        node_ids=[original.node_id],
        parent_id=original.photo_id,
    )


def _apply_instruction(assembled: AssembledPrompt, instruction: str) -> AssembledPrompt:
    """Fold an edit instruction into a prompt as a new ACTION addendum.

    Kept as data in a fixed section rather than appended blindly, so the prompt
    shape stays predictable and the instruction cannot masquerade as system text.
    """
    text = " ".join(instruction.split())
    if not text:
        raise AgentError("An edit instruction must not be empty.")

    lines = assembled.prompt.splitlines()
    marker = "[ACTION]"
    for index, line in enumerate(lines):
        if line.startswith(marker):
            lines[index] = f"{line} Edit: {text}"
            break
    else:
        lines.append(f"{marker} Edit: {text}")

    return AssembledPrompt(
        prompt="\n".join(lines),
        negative_prompt=assembled.negative_prompt,
        reference_notes=assembled.reference_notes,
        prompt_version=assembled.prompt_version,
        truncated_reference=assembled.truncated_reference,
        estimated_tokens=assembled.estimated_tokens,
        metadata={**assembled.metadata, "edited": True, "instruction": text},
    )


def new_shoot_id(prefix: str = "shoot") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


__all__ = [
    "MAX_ATTEMPTS_PER_SHOT",
    "RENDER_VERSION",
    "RenderOutcome",
    "RenderedPhoto",
    "default_seed",
    "identity_refs_for",
    "new_shoot_id",
    "render_edit",
    "render_plan",
]
