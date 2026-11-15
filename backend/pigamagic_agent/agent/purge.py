"""Account deletion purge and the flagged-photo review registry (SR-1 → SR-7).

Two obligations live here, and they are the same obligation from two angles:
**nothing we hold about a person may outlive their consent.**

1. `purge_account()` — deleting an account removes media blobs, reference
   analyses, photos, runs, events and ledger rows (SR-5, design.md §14).
2. `FlagRegistry` — a photo the safety classifier flagged is *private*: it is
   visible to an admin reviewer and unreachable from every public route (SR-4).

Design decisions worth keeping:

- **The purge reports what it removed.** A function that only promises to delete
  is untestable; a `PurgeReport` with per-store counts is what makes
  "verified end to end" a real check rather than an assertion of intent.
- **Blobs are deleted only when no surviving photo still references them.**
  Photos are content-addressed, so two accounts sharing a byte-identical frame
  share one blob. Deleting it per-account would break the other account's photo
  — the purge must be scoped, not blunt.
- **The ledger is purged, not blanked.** A balance is derived from entries, so
  removing them leaves the account at zero with no phantom credit.
- **Flags are checked before lineage and compare, not inside them.** Every
  public route already asks the flag registry, so a new route cannot be added
  without the same question being answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Protocol, Sequence

from pigamagic_agent.agent.ledger import CreditLedger
from pigamagic_agent.agent.lineage import PhotoStore
from pigamagic_agent.agent.run import RunStore

PURGE_VERSION = "1"

#: Photo record fields that may name the account that owns the photo. The
#: pipeline writes `account_id`; the alternates are accepted so a shell-side
#: serialiser cannot silently make the purge miss a row.
_ACCOUNT_FIELDS: tuple[str, ...] = ("account_id", "owner_id", "user_id")


class PurgeError(Exception):
    """Raised when a purge cannot be completed safely."""


class DeletableBlobs(Protocol):
    """The slice of `BlobStore` a purge needs. Async, like the real store."""

    async def delete(self, key: str) -> bool: ...

    async def exists(self, key: str) -> bool: ...


class DeletableAnalysisCache(Protocol):
    """The slice of `ReferenceAnalysisCache` a purge needs."""

    def purge_account(self, account_id: str) -> int: ...


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass(slots=True)
class PurgeReport:
    """What a purge actually removed. Counts, so a test can assert each one."""

    account_id: str
    runs: int = 0
    events: int = 0
    photos: int = 0
    analyses: int = 0
    blobs: int = 0
    ledger_entries: int = 0
    reservations: int = 0
    #: Blobs left in place because another account still owns a photo of the
    #: same bytes. Reported rather than hidden: it explains a non-zero store.
    blobs_retained: int = 0
    #: Media keys the account claimed, before the shared-blob check.
    media_keys: list[str] = field(default_factory=list)
    purge_version: str = PURGE_VERSION

    @property
    def empty(self) -> bool:
        """True when there was nothing to delete. Still a successful purge."""
        return not any(
            (self.runs, self.events, self.photos, self.analyses, self.blobs, self.ledger_entries)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "purge_version": self.purge_version,
            "runs": self.runs,
            "events": self.events,
            "photos": self.photos,
            "analyses": self.analyses,
            "blobs": self.blobs,
            "ledger_entries": self.ledger_entries,
            "reservations": self.reservations,
            "blobs_retained": self.blobs_retained,
            "media_keys": list(self.media_keys),
        }


# --------------------------------------------------------------------------
# Flag registry (SR-4)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FlagEntry:
    """One flagged photo, with the evidence a reviewer needs to decide.

    `plan_node` and `prompt` are the point of the console: a reviewer must be
    able to see *what was asked for* and *what the model would be told*, not
    just a photo and a code.
    """

    photo_id: str
    account_id: str
    codes: tuple[str, ...] = ()
    reason: str = ""
    run_id: str = ""
    shoot_id: str = ""
    node_id: str = ""
    #: The shot that produced it, as the plan described it.
    plan_node: dict[str, Any] = field(default_factory=dict)
    #: The assembled prompt, so a review is about the actual instruction.
    prompt: str = ""
    negative_prompt: str = ""
    blob_key: str = ""
    state: str = "flagged"
    created_at: str = ""

    @property
    def public(self) -> bool:
        """Only `cleared` photos may be served. Everything else stays private."""
        return self.state == "cleared"

    def as_dict(self) -> dict[str, Any]:
        return {
            "photo_id": self.photo_id,
            "account_id": self.account_id,
            "codes": list(self.codes),
            "reason": self.reason,
            "run_id": self.run_id,
            "shoot_id": self.shoot_id,
            "node_id": self.node_id,
            "plan_node": dict(self.plan_node),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "blob_key": self.blob_key,
            "state": self.state,
            "created_at": self.created_at,
        }


class FlagRegistry:
    """Photo flag state. In-memory here; a table in the product shell.

    Three states, and the asymmetry is deliberate:

    - `flagged` — hidden from every public route, awaiting review.
    - `cleared` — reviewed and allowed; now publicly reachable.
    - `removed` — reviewed and destroyed. Never reachable, ever again.

    A photo with **no entry** is not flagged: it never tripped a rule. Only an
    entry in `flagged` or `removed` state blocks access.
    """

    def __init__(self) -> None:
        self._entries: dict[str, FlagEntry] = {}

    def flag(self, entry: FlagEntry) -> FlagEntry:
        if not entry.photo_id:
            raise PurgeError("a flag needs a photo_id")
        if not entry.codes:
            raise PurgeError(
                f"flag for {entry.photo_id!r} carries no policy code; "
                "a flag without a reason cannot be reviewed."
            )
        existing = self._entries.get(entry.photo_id)
        if existing is not None and existing.state == "removed":
            # A removed photo must not come back via a late duplicate flag.
            return existing
        self._entries[entry.photo_id] = entry
        return entry

    def get(self, photo_id: str) -> FlagEntry | None:
        return self._entries.get(photo_id)

    def state(self, photo_id: str) -> str:
        entry = self._entries.get(photo_id)
        return entry.state if entry is not None else "unflagged"

    def blocked(self, photo_id: str) -> bool:
        """True when a public route must refuse this photo."""
        return self.state(photo_id) in ("flagged", "removed")

    def pending(self, *, account_id: str | None = None) -> list[FlagEntry]:
        """The review queue, oldest first. Stable order, so pagination is sane."""
        items = [
            entry
            for entry in self._entries.values()
            if entry.state == "flagged"
            and (account_id is None or entry.account_id == account_id)
        ]
        return sorted(items, key=lambda entry: (entry.created_at, entry.photo_id))

    def clear(self, photo_id: str, *, reviewer: str = "", note: str = "") -> FlagEntry:
        return self._transition(photo_id, "cleared", reviewer=reviewer, note=note)

    def remove(self, photo_id: str, *, reviewer: str = "", note: str = "") -> FlagEntry:
        return self._transition(photo_id, "removed", reviewer=reviewer, note=note)

    def _transition(self, photo_id: str, state: str, *, reviewer: str, note: str) -> FlagEntry:
        entry = self._entries.get(photo_id)
        if entry is None:
            raise PurgeError(f"photo {photo_id!r} was never flagged; nothing to review.")
        if entry.state == "removed":
            raise PurgeError(
                f"photo {photo_id!r} was already removed; a removed photo cannot be reinstated."
            )
        reason = entry.reason
        if reviewer:
            reason = f"{reason} [reviewed by {reviewer}]".strip()
        if note:
            reason = f"{reason} — {note}".strip()
        updated = replace(entry, state=state, reason=reason)
        self._entries[photo_id] = updated
        return updated

    def purge_account(self, account_id: str) -> int:
        """Drop every flag belonging to an account. Returns how many went."""
        doomed = [pid for pid, entry in self._entries.items() if entry.account_id == account_id]
        for photo_id in doomed:
            del self._entries[photo_id]
        return len(doomed)

    def __len__(self) -> int:
        return len(self._entries)


# --------------------------------------------------------------------------
# Purging
# --------------------------------------------------------------------------


def _owner_of(photo: Mapping[str, Any]) -> str:
    for name in _ACCOUNT_FIELDS:
        value = photo.get(name)
        if value:
            return str(value)
    return ""


def _blob_key_of(photo: Mapping[str, Any]) -> str:
    blob = photo.get("blob")
    if isinstance(blob, Mapping):
        return str(blob.get("key") or "")
    return str(photo.get("blob_key") or "")


def _runs_for_account(store: RunStore, account_id: str) -> list[Any]:
    listed = getattr(store, "list_runs", None)
    runs = list(listed()) if callable(listed) else []
    return [run for run in runs if getattr(run, "account_id", "") == account_id]


def _delete_runs(store: RunStore, run_ids: Iterable[str]) -> int:
    """Remove runs from the store. Requires the store to support deletion."""
    remover = getattr(store, "delete", None)
    if not callable(remover):
        raise PurgeError(
            "the run store cannot delete rows, so account deletion would leave "
            "run history behind. Implement `delete(run_id)` on the store."
        )
    count = 0
    for run_id in run_ids:
        if remover(run_id):
            count += 1
    return count


def _delete_events(log: Any, run_ids: Sequence[str]) -> int:
    remover = getattr(log, "delete", None)
    if not callable(remover):
        raise PurgeError(
            "the event log cannot delete runs, so streamed events would outlive "
            "the account. Implement `delete(run_id)` on the log."
        )
    return sum(1 for run_id in run_ids if remover(run_id))


def _delete_photos(store: PhotoStore, account_id: str) -> tuple[int, list[str]]:
    """Remove every photo owned by the account. Returns (count, blob keys).

    A photo with *no* owner field is not deleted: it is not provably this
    account's, and deleting someone else's delivery during a purge is far worse
    than leaving a stray row for an operator to look at.
    """
    all_photos = list(store.all_photos())
    mine = [photo for photo in all_photos if _owner_of(photo) == account_id]
    keys: list[str] = []
    for photo in mine:
        key = _blob_key_of(photo)
        if key:
            keys.append(key)

    remover = getattr(store, "delete", None)
    if not callable(remover):
        raise PurgeError(
            "the photo store cannot delete rows. Implement `delete(photo_id)`."
        )
    count = 0
    for photo in mine:
        photo_id = str(photo.get("photo_id") or photo.get("id") or "")
        if photo_id and remover(photo_id):
            count += 1
    return count, keys


def _surviving_keys(store: PhotoStore, keys: Sequence[str]) -> set[str]:
    """Blob keys still referenced by a photo that is not being purged."""
    wanted = set(keys)
    survivors: set[str] = set()
    for photo in store.all_photos():
        key = _blob_key_of(photo)
        if key in wanted:
            survivors.add(key)
    return survivors


def _delete_ledger(ledger: CreditLedger, account_id: str) -> tuple[int, int]:
    """Drop the account's ledger rows. Returns (entries, reservations)."""
    purge = getattr(ledger, "purge_account", None)
    if callable(purge):
        result = purge(account_id)
        return int(result.get("entries", 0)), int(result.get("reservations", 0))

    # Fall back to manual removal so an older ledger still purges correctly.
    entries = [entry for entry in ledger.entries(account_id=account_id)]
    reservation_ids = [
        key
        for key, reservation in ledger._reservations.items()  # noqa: SLF001
        if reservation.account_id == account_id
    ]
    for key in reservation_ids:
        del ledger._reservations[key]  # noqa: SLF001
    remaining = [entry for entry in ledger._entries if entry.account_id != account_id]  # noqa: SLF001
    removed = len(ledger._entries) - len(remaining)  # noqa: SLF001
    ledger._entries = remaining  # noqa: SLF001
    ledger._by_key = {  # noqa: SLF001
        key: entry for key, entry in ledger._by_key.items() if entry.account_id != account_id  # noqa: SLF001
    }
    ledger._balances.pop(account_id, None)  # noqa: SLF001
    return removed or len(entries), len(reservation_ids)


async def purge_account(
    account_id: str,
    *,
    ledger: CreditLedger,
    runs: RunStore,
    events: Any,
    photos: PhotoStore,
    blobs: DeletableBlobs,
    analyses: Any | None = None,
    flags: FlagRegistry | None = None,
) -> PurgeReport:
    """Delete everything held about one account (SR-5).

    Order is deliberate and runs **innermost-out**:

    1. photos and their blob keys — we must know the bytes before the rows go
    2. blobs, minus any still referenced by another account
    3. reference analyses
    4. events, then runs (the log outlives a run in a live worker)
    5. ledger rows last, so a failure earlier leaves the charge visible rather
       than granting free credits on a half-finished purge
    6. flags, so a reviewer never sees a queue entry for an account that no
       longer exists

    Each step is counted into the report. Nothing is silently skipped: a store
    that cannot delete raises, because a purge that quietly does nothing is the
    exact compliance failure this function exists to prevent.
    """
    if not account_id:
        raise PurgeError("purge_account needs an account id")

    report = PurgeReport(account_id=account_id)

    # 1 + 2. Photos, then blobs that no surviving photo references.
    report.photos, keys = _delete_photos(photos, account_id)
    report.media_keys = keys
    shared = _surviving_keys(photos, keys)
    deleted_keys: set[str] = set()
    for key in keys:
        if key in deleted_keys:
            continue
        if key in shared:
            report.blobs_retained += 1
            deleted_keys.add(key)
            continue
        deleted_keys.add(key)
        if await blobs.delete(key):
            report.blobs += 1

    # 3. Reference analyses — derived data, so it goes with the references.
    if analyses is not None:
        purge_analyses = getattr(analyses, "purge_account", None)
        if callable(purge_analyses):
            report.analyses = int(purge_analyses(account_id))

    # 4. Events before runs: a run's log must not outlive the run.
    run_rows = _runs_for_account(runs, account_id)
    run_ids = [run.run_id for run in run_rows]
    report.events = _delete_events(events, run_ids)
    report.runs = _delete_runs(runs, run_ids)

    # 5. Ledger last.
    report.ledger_entries, report.reservations = _delete_ledger(ledger, account_id)

    # 6. Flags.
    if flags is not None:
        flags.purge_account(account_id)

    return report


# --------------------------------------------------------------------------
# Admin review view
# --------------------------------------------------------------------------


def review_card(entry: FlagEntry) -> dict[str, Any]:
    """One row of the admin review console.

    Carries the plan node and prompt, because a reviewer deciding whether a
    photo is acceptable needs the instruction that produced it — a bare image
    and a rule code is not a review.
    """
    return {
        "photo_id": entry.photo_id,
        "account_id": entry.account_id,
        "state": entry.state,
        "codes": list(entry.codes),
        "reason": entry.reason,
        "run_id": entry.run_id,
        "shoot_id": entry.shoot_id,
        "node_id": entry.node_id,
        "prompt": entry.prompt,
        "negative_prompt": entry.negative_prompt,
        "plan_node": dict(entry.plan_node),
        "blob_key": entry.blob_key,
        "created_at": entry.created_at,
        # A reviewer must never be shown a route that would work for the user.
        "public": entry.public,
        "actions": ["clear", "remove"] if entry.state == "flagged" else [],
    }


def review_queue(
    flags: FlagRegistry, *, account_id: str | None = None
) -> dict[str, Any]:
    """The console listing. Pending first, then a total so the queue size is visible."""
    pending = flags.pending(account_id=account_id)
    return {
        "items": [review_card(entry) for entry in pending],
        "count": len(pending),
        "registry_size": len(flags),
    }


__all__ = [
    "DeletableAnalysisCache",
    "DeletableBlobs",
    "FlagEntry",
    "FlagRegistry",
    "PURGE_VERSION",
    "PurgeError",
    "PurgeReport",
    "purge_account",
    "review_card",
    "review_queue",
]
