"""Run orchestration: brief -> plan -> photos, with credits and a real state machine.

A run is the unit the user watches. It outlives any HTTP connection (design.md
§10.1), so state is held in a store and the event stream is a notification
channel over it — not the transport for pixels.

State machine (§9.1):

    queued -> researching -> compiling -> rendering -> done | partial | failed

Transitions are validated, never assumed. A run that jumps `queued -> done` is a
bug, and this module fails loudly rather than recording a state the UI cannot
render. Terminal states are terminal: `partial` and `failed` cannot drift to
`done` later, so a settled credit balance stays correct.

Ordering is deliberate and is the heart of the design:

1. safety gate  — a refusal costs zero credits and zero provider calls
2. reserve      — an unaffordable run is rejected before any work begins
3. research/compile
4. render
5. settle       — charge delivered, refund failed, refund cache hits

Reserving *after* the safety gate matters: refusing a brief must not leave a
hold on the user's balance. Reserving *before* rendering matters: it is what
stops concurrent runs from overspending the same balance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from .compile import ShootBrief
from .config import AgentSettings, get_settings
from .hashing import sha256_bytes, stable_digest
from .ledger import (
    CREDITS_PER_PHOTO,
    CreditLedger,
    InsufficientCredits,
    LedgerError,
    Reservation,
)
from .pipeline import ShootArtifact, run_pipeline
from .providers.base import AgentError, LLMProvider, VisionProvider, WebProvider
from .providers.image_base import BaseImageProvider
from .providers.http_base import BudgetLedger
from .render import RenderOutcome, render_plan
from .safety import SafetyRefusal, check_brief, require_allowed
from .storage import BlobStore

RUN_VERSION = "run-v1"

RunStatus = Literal[
    "queued",
    "researching",
    "compiling",
    "rendering",
    "done",
    "partial",
    "failed",
]

TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "partial", "failed"})
ACTIVE_STATUSES: frozenset[str] = frozenset(
    {"queued", "researching", "compiling", "rendering"}
)

#: Legal transitions. Everything else is a bug, not a state to tolerate.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"researching", "failed", "done", "partial"}),
    "researching": frozenset({"compiling", "failed"}),
    "compiling": frozenset({"rendering", "failed"}),
    "rendering": frozenset({"done", "partial", "failed"}),
    "done": frozenset(),
    "partial": frozenset(),
    "failed": frozenset(),
}


class RunError(AgentError):
    """Base class for run problems."""


class RunStateError(RunError):
    """A transition the state machine does not permit."""


class RunNotFound(RunError):
    """No run with that id."""


class RunInProgress(RunError):
    """A live run already exists for this shoot."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class GenerationRun:
    """One generation run. Mutable only through `RunStore.transition`."""

    run_id: str
    shoot_id: str
    account_id: str
    status: RunStatus = "queued"
    requested: int = 0
    reserved_credits: int = 0
    spent_credits: int = 0
    refunded_credits: int = 0
    idempotency_key: str = ""
    plan_id: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    photo_ids: list[str] = field(default_factory=list)
    cache_hits: int = 0
    created_at: str = _now()
    finished_at: str = ""
    #: Ordered log of every transition, so a support ticket can be answered by
    #: reading what actually happened instead of inferring it.
    history: list[dict[str, str]] = field(default_factory=list)
    run_version: str = RUN_VERSION

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_version": self.run_version,
            "run_id": self.run_id,
            "shoot_id": self.shoot_id,
            "account_id": self.account_id,
            "status": self.status,
            "requested": self.requested,
            "reserved_credits": self.reserved_credits,
            "spent_credits": self.spent_credits,
            "refunded_credits": self.refunded_credits,
            "plan_id": self.plan_id,
            "error": self.error,
            "warnings": list(self.warnings),
            "photo_ids": list(self.photo_ids),
            "cache_hits": self.cache_hits,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "history": list(self.history),
        }


class RunStore(Protocol):
    """Where runs live. In-memory here, Postgres in the product shell."""

    def put(self, run: GenerationRun) -> None: ...

    def get(self, run_id: str) -> GenerationRun | None: ...

    def active_for_shoot(self, shoot_id: str) -> GenerationRun | None: ...

    def by_idempotency_key(self, key: str) -> GenerationRun | None: ...

    def link_idempotency(self, key: str, run_id: str) -> None: ...


class InMemoryRunStore:
    """Default store. Correct semantics, no durability."""

    def __init__(self) -> None:
        self._runs: dict[str, GenerationRun] = {}
        self._by_shoot: dict[str, str] = {}
        self._by_key: dict[str, str] = {}

    def put(self, run: GenerationRun) -> None:
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> GenerationRun | None:
        return self._runs.get(run_id)

    def active_for_shoot(self, shoot_id: str) -> GenerationRun | None:
        run_id = self._by_shoot.get(shoot_id)
        if run_id is None:
            return None
        run = self._runs.get(run_id)
        if run is None or not run.active:
            return None
        return run

    def by_idempotency_key(self, key: str) -> GenerationRun | None:
        run_id = self._by_key.get(key)
        return None if run_id is None else self._runs.get(run_id)

    def link_idempotency(self, key: str, run_id: str) -> None:
        self._by_key[key] = run_id

    def track_shoot(self, run: GenerationRun) -> None:
        self._by_shoot[run.shoot_id] = run.run_id

    def list_runs(self) -> list[GenerationRun]:
        """All runs, newest first. Used by the feed to enumerate cards."""
        return sorted(
            self._runs.values(),
            key=lambda run: (run.created_at, run.run_id),
            reverse=True,
        )


def new_run_id(brief: ShootBrief, *, attempt: str = "") -> str:
    """Content-derived run id, so the same brief + key is the same run."""
    material = stable_digest(RUN_VERSION, brief.shoot_id, brief.title, attempt)
    return "run_" + material[:20]


def transition(run: GenerationRun, status: RunStatus, *, note: str = "") -> GenerationRun:
    """Move a run to `status`, or raise. The only way status ever changes."""
    if status == run.status:
        return run
    if status not in ALLOWED_TRANSITIONS.get(run.status, frozenset()):
        raise RunStateError(
            f"run {run.run_id} cannot go from {run.status!r} to {status!r}; "
            f"legal next states are {sorted(ALLOWED_TRANSITIONS.get(run.status, frozenset())) or 'none (terminal)'}."
        )
    run.status = status
    run.history.append({"status": status, "at": _now(), "note": note})
    if status in TERMINAL_STATUSES:
        run.finished_at = run.history[-1]["at"]
    return run


@dataclass(frozen=True, slots=True)
class RunResult:
    """What one orchestrated run produced, in the terms the API reports."""

    run: GenerationRun
    artifact: ShootArtifact | None = None
    outcome: RenderOutcome | None = None
    reservation: Reservation | None = None

    @property
    def status(self) -> str:
        return self.run.status

    def as_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.as_dict(),
            "artifact_digest": None if self.artifact is None else self.artifact.digest(),
            "render": None if self.outcome is None else self.outcome.as_dict(),
            "reservation": None if self.reservation is None else self.reservation.as_dict(),
        }


async def start_run(
    brief: ShootBrief,
    *,
    account_id: str,
    ledger: CreditLedger,
    store: RunStore,
    llm: LLMProvider,
    vision: VisionProvider,
    web: WebProvider,
    image: BaseImageProvider,
    blobs: BlobStore,
    settings: AgentSettings | None = None,
    cast: Sequence[Any] | None = None,
    reference_images: Mapping[str, Sequence[bytes]] | None = None,
    shot_count: int | None = None,
    idempotency_key: str = "",
    budget: BudgetLedger | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> RunResult:
    """Run the whole pipeline for one brief under a credit reservation.

    Never raises for provider or credit problems: a failure is a recorded,
    terminal run. The caller gets a run it can show, with the reason on it.
    """
    active = settings or get_settings()
    emit = on_event or (lambda _kind, _payload: None)

    # --- idempotent replay --------------------------------------------------
    if idempotency_key:
        existing = store.by_idempotency_key(idempotency_key)
        if existing is not None:
            return RunResult(
                run=existing,
                reservation=ledger.reservation_for_run(existing.run_id),
            )

    # --- one live run per shoot --------------------------------------------
    live = store.active_for_shoot(brief.shoot_id)
    if live is not None:
        raise RunInProgress(
            f"shoot {brief.shoot_id} already has an active run ({live.run_id}, "
            f"status {live.status!r}). Wait for it to finish or cancel it."
        )

    run = GenerationRun(
        run_id=new_run_id(brief, attempt=idempotency_key),
        shoot_id=brief.shoot_id,
        account_id=account_id,
        requested=shot_count if shot_count is not None else len(brief.activities),
        idempotency_key=idempotency_key,
    )
    store.put(run)
    if isinstance(store, InMemoryRunStore):
        store.track_shoot(run)
    if idempotency_key:
        store.link_idempotency(idempotency_key, run.run_id)
    emit("run", run.as_dict())

    # --- 1. safety gate: zero cost, zero provider calls ---------------------
    verdict = check_brief(brief)
    if verdict.refused:
        run.error = verdict.reason
        run.warnings.extend(flag.message for flag in verdict.care_flags)
        transition(run, "failed", note="refused by safety gate")
        emit("run_done", run.as_dict())
        return RunResult(run=run)

    # --- 2. reserve: affordable requests only ------------------------------
    wanted = shot_count if shot_count is not None else 0
    try:
        reservation = ledger.reserve(
            account_id,
            max(0, wanted) * CREDITS_PER_PHOTO,
            run_id=run.run_id,
            idempotency_key=f"reserve:{run.run_id}",
        )
    except InsufficientCredits as exc:
        run.error = str(exc)
        transition(run, "failed", note="insufficient credits")
        emit("run_done", run.as_dict())
        return RunResult(run=run)
    except LedgerError as exc:
        run.error = f"credit reservation failed: {exc}"
        transition(run, "failed", note="ledger error")
        emit("run_done", run.as_dict())
        return RunResult(run=run)

    run.reserved_credits = reservation.reserved
    emit("run", run.as_dict())

    # --- 3. research + compile ---------------------------------------------
    transition(run, "researching", note="analysing references and researching venue")
    emit("run", run.as_dict())

    try:
        artifact = await run_pipeline(
            brief,
            llm=llm,
            vision=vision,
            web=web,
            settings=active,
            cast_override=cast,
        )
    except SafetyRefusal as exc:
        run.error = str(exc)
        ledger.cancel(reservation, reason="run refused")
        run.refunded_credits = reservation.refunded
        transition(run, "failed", note="refused by safety gate")
        emit("run_done", run.as_dict())
        return RunResult(run=run, reservation=reservation)
    except (AgentError, ValueError) as exc:
        run.error = f"planning failed: {exc}"
        ledger.cancel(reservation, reason="planning failed")
        run.refunded_credits = reservation.refunded
        transition(run, "failed", note="planning failed")
        emit("run_done", run.as_dict())
        return RunResult(run=run, reservation=reservation)

    run.plan_id = artifact.digest()[:16]
    run.warnings.extend(artifact.warnings)

    transition(run, "compiling", note=f"plan {run.plan_id}")
    emit("run", run.as_dict())

    # --- 4. render ---------------------------------------------------------
    transition(run, "rendering", note=f"{len(artifact.plan.shots)} node(s)")
    emit("run", run.as_dict())

    try:
        outcome = await render_plan(
            artifact,
            provider=image,
            store=blobs,
            settings=active,
            reference_images=reference_images,
            max_shots=shot_count,
            budget=budget,
        )
    except SafetyRefusal as exc:
        run.error = str(exc)
        ledger.cancel(reservation, reason="run refused at render")
        run.refunded_credits = reservation.refunded
        transition(run, "failed", note="refused by safety gate")
        emit("run_done", run.as_dict())
        return RunResult(run=run, artifact=artifact, reservation=reservation)
    except (AgentError, ValueError) as exc:
        run.error = f"rendering failed: {exc}"
        ledger.cancel(reservation, reason="rendering failed")
        run.refunded_credits = reservation.refunded
        transition(run, "failed", note="rendering failed")
        emit("run_done", run.as_dict())
        return RunResult(run=run, artifact=artifact, reservation=reservation)

    run.photo_ids = [photo.photo_id for photo in outcome.photos]
    run.warnings.extend(outcome.warnings)
    failed_nodes = [item.get("node_id", "") for item in outcome.failures]

    # --- 5. settle ---------------------------------------------------------
    ledger.settle(reservation, delivered=outcome.count, file_failed=failed_nodes)
    run.spent_credits = reservation.charged
    run.refunded_credits = reservation.refunded

    if not outcome.photos:
        run.error = "no photos were produced"
        transition(run, "failed", note="nothing rendered")
    elif outcome.complete and len(outcome.photos) >= run.requested:
        transition(run, "done", note=f"{outcome.count} photo(s)")
    elif outcome.complete:
        # Everything asked for in the plan rendered, but fewer than requested:
        # report it honestly rather than as a clean success.
        run.warnings.append(
            f"{run.requested} photo(s) requested, {outcome.count} produced."
        )
        transition(run, "done", note=f"{outcome.count} of {run.requested} requested")
    else:
        transition(run, "partial", note=f"{outcome.count} ok, {len(outcome.failures)} failed")

    emit("run_done", run.as_dict())
    return RunResult(run=run, artifact=artifact, outcome=outcome, reservation=reservation)


def run_digest(result: RunResult) -> str:
    """Stable fingerprint of a run's outcome (content, not timestamps)."""
    parts = [RUN_VERSION, result.run.run_id, result.run.status]
    if result.artifact is not None:
        parts.append(result.artifact.digest())
    if result.outcome is not None:
        parts.append(result.outcome.digest())
    return sha256_bytes("|".join(parts).encode("utf-8"))


__all__ = [
    "ACTIVE_STATUSES",
    "ALLOWED_TRANSITIONS",
    "GenerationRun",
    "InMemoryRunStore",
    "RUN_VERSION",
    "RunError",
    "RunInProgress",
    "RunNotFound",
    "RunResult",
    "RunStateError",
    "RunStatus",
    "RunStore",
    "TERMINAL_STATUSES",
    "new_run_id",
    "run_digest",
    "start_run",
    "transition",
]
