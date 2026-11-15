"""Lineage, compare, and per-edit credits (design.md §11.2, steps.md Step 18).

Three ideas, and the third is the one that costs money:

1. **Lineage is a tree, not a list.** A photo may be edited more than once, and
   an edited take may itself be edited. `parent_id` therefore forms a forest, and
   "siblings" means every take sharing a parent — which is exactly what
   FR-7.6's "compare and delete takes" needs.

2. **Compare is a diff of the plan node, not of the pixels.** The useful question
   is "what did this edit change, and what did it hold constant?", so the compare
   payload reports the instruction delta and the `preserves` that were honoured.

3. **Each edit reserves its own credits** under its own run id, keyed on
   `instruction_key()`. An edit is a separate charge from the shoot that produced
   the parent; folding it into the parent's run would make the ledger unable to
   answer "what did this one edit cost me?" and would let one retry charge twice.

The invariant everything here protects is §11.2's hard rule: **never mutate a
delivered photo.** A child is additive, and every lookup refuses to invent a
parent that is not in the store rather than producing a plausible-looking tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .comments import instruction_key
from .ledger import CREDITS_PER_PHOTO, CreditLedger, Reservation
from .plan import EditInstruction

LINEAGE_VERSION = "lineage-v1"

#: Fields compared between a parent and a child. Deliberately *not* the prompt
#: hash alone: a hash difference tells you something changed but not what, which
#: is useless in a compare view.
COMPARE_FIELDS: tuple[str, ...] = (
    "framing",
    "moment",
    "activity",
    "seed",
    "prompt_hash",
    "provider",
    "model",
)


class LineageError(ValueError):
    """A lineage request that cannot be honoured."""


# --------------------------------------------------------------------------
# Photo registry
# --------------------------------------------------------------------------


class PhotoStore(Protocol):
    """Where delivered photos live. In-memory here, Postgres in the shell."""

    def put(self, photo: Mapping[str, Any]) -> None: ...

    def get(self, photo_id: str) -> dict[str, Any] | None: ...

    def children_of(self, photo_id: str) -> list[dict[str, Any]]: ...

    def all_photos(self) -> list[dict[str, Any]]: ...

    def delete(self, photo_id: str) -> bool:
        """Remove one photo row. Required for account deletion (SR-5)."""
        ...


class InMemoryPhotoStore:
    """Default store. Appends only — a delivered photo is never replaced."""

    def __init__(self) -> None:
        self._photos: dict[str, dict[str, Any]] = {}
        self._by_parent: dict[str, list[str]] = {}

    def put(self, photo: Mapping[str, Any]) -> None:
        photo_id = str(photo.get("photo_id") or photo.get("id") or "")
        if not photo_id:
            raise LineageError("a stored photo needs a photo_id")
        existing = self._photos.get(photo_id)
        if existing is not None:
            # Idempotent when identical, loud when not: silently overwriting a
            # delivered photo is the trust violation §11.2 forbids.
            if existing.get("prompt_hash") != photo.get("prompt_hash"):
                raise LineageError(
                    f"photo {photo_id!r} already exists with different content; "
                    "delivered photos are immutable — create a child instead."
                )
            return
        record = dict(photo)
        record["photo_id"] = photo_id
        self._photos[photo_id] = record
        parent_id = str(record.get("parent_id") or "")
        if parent_id:
            self._by_parent.setdefault(parent_id, []).append(photo_id)

    def get(self, photo_id: str) -> dict[str, Any] | None:
        return self._photos.get(photo_id)

    def children_of(self, photo_id: str) -> list[dict[str, Any]]:
        ids = self._by_parent.get(photo_id, [])
        return [self._photos[child] for child in ids if child in self._photos]

    def all_photos(self) -> list[dict[str, Any]]:
        return list(self._photos.values())

    def delete(self, photo_id: str) -> bool:
        """Remove one photo row and its parent index entry.

        Deliberately *not* the tree-aware `delete_take`: a purge removes a whole
        account, so orphaning rules that protect a live family do not apply. The
        caller (`purge_account`) deletes every photo the account owns.
        """
        record = self._photos.pop(photo_id, None)
        if record is None:
            return False
        parent_id = str(record.get("parent_id") or "")
        if parent_id and parent_id in self._by_parent:
            siblings = [child for child in self._by_parent[parent_id] if child != photo_id]
            if siblings:
                self._by_parent[parent_id] = siblings
            else:
                del self._by_parent[parent_id]
        self._by_parent.pop(photo_id, None)
        return True


# --------------------------------------------------------------------------
# Lineage
# --------------------------------------------------------------------------


def root_of(store: PhotoStore, photo_id: str) -> dict[str, Any]:
    """Walk `parent_id` up to the original. Detects cycles instead of hanging."""
    seen: set[str] = set()
    current = store.get(photo_id)
    if current is None:
        raise LineageError(f"no photo {photo_id!r}")
    while True:
        parent_id = str(current.get("parent_id") or "")
        if not parent_id:
            return current
        if parent_id in seen or parent_id == photo_id:
            raise LineageError(
                f"lineage for {photo_id!r} is cyclic at {parent_id!r}"
            )
        seen.add(parent_id)
        parent = store.get(parent_id)
        if parent is None:
            # A missing parent is a data problem, not a root: pretending it is
            # the root would show a child as an original.
            raise LineageError(
                f"photo {photo_id!r} references missing parent {parent_id!r}"
            )
        current = parent


def siblings_of(store: PhotoStore, photo_id: str) -> list[dict[str, Any]]:
    """Every take sharing this photo's parent, plus this photo, in stable order.

    A root photo's "siblings" are its children: the root is the only original, so
    its alternate takes are the edits made from it.
    """
    photo = store.get(photo_id)
    if photo is None:
        raise LineageError(f"no photo {photo_id!r}")

    parent_id = str(photo.get("parent_id") or "")
    if parent_id:
        pool = [store.get(parent_id), *store.children_of(parent_id)]
    else:
        pool = [photo, *store.children_of(photo_id)]

    unique: dict[str, dict[str, Any]] = {}
    for item in pool:
        if item is None:
            continue
        pid = str(item.get("photo_id") or item.get("id") or "")
        if pid:
            unique[pid] = item
    # Stable, ascending by creation then id: a compare view must not reorder
    # between two identical requests.
    return sorted(
        unique.values(),
        key=lambda p: (str(p.get("created_at") or ""), str(p.get("photo_id") or "")),
    )


def lineage_tree(
    store: PhotoStore, photo_id: str, *, max_depth: int = 8
) -> dict[str, Any]:
    """The whole take-set as a tree, rooted at the original."""
    root = root_of(store, photo_id)

    def build(node: Mapping[str, Any], depth: int) -> dict[str, Any]:
        node_id = str(node.get("photo_id") or "")
        children = []
        if depth < max_depth:
            children = [
                build(child, depth + 1) for child in store.children_of(node_id)
            ]
        return {
            "photo_id": node_id,
            "node_id": str(node.get("node_id") or ""),
            "parent_id": node.get("parent_id"),
            "caption": str(node.get("caption") or ""),
            "created_at": str(node.get("created_at") or ""),
            "edit": node.get("edit_instruction"),
            "children": children,
        }

    tree = build(root, 0)
    siblings = siblings_of(store, photo_id)
    return {
        "lineage_version": LINEAGE_VERSION,
        "photo_id": photo_id,
        "root_id": tree["photo_id"],
        "depth": _depth_of(store, photo_id),
        "sibling_count": len(siblings),
        "siblings": [str(item.get("photo_id") or "") for item in siblings],
        "tree": tree,
    }


def _depth_of(store: PhotoStore, photo_id: str) -> int:
    depth = 0
    seen: set[str] = set()
    current = store.get(photo_id)
    while current is not None:
        parent_id = str(current.get("parent_id") or "")
        if not parent_id or parent_id in seen:
            return depth
        seen.add(parent_id)
        current = store.get(parent_id)
        if current is not None:
            depth += 1
    return depth


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------


def compare(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """What changed between a parent and a child, and what was preserved.

    Reports only fields that actually differ, so a compare view does not have to
    diff two full payloads in the client.
    """
    parent_id = str(parent.get("photo_id") or "")
    child_parent_id = str(child.get("parent_id") or "")
    if child_parent_id != parent_id:
        raise LineageError(
            f"cannot compare {child.get('photo_id')!r} against {parent_id!r}: "
            f"it is not its parent (its parent is {child_parent_id or 'none'!r})."
        )

    changed: dict[str, Any] = {}
    for name in COMPARE_FIELDS:
        before = parent.get(name)
        after = child.get(name)
        if before != after:
            changed[name] = {"from": before, "to": after}

    return {
        "lineage_version": LINEAGE_VERSION,
        "parent_id": parent_id,
        "child_id": str(child.get("photo_id") or ""),
        "node_id": str(child.get("node_id") or ""),
        "changed": changed,
        # The delta that produced the child, so the view can say *why* it differs.
        "instruction": child.get("edit_instruction"),
        "preserves": list(child.get("preserves") or []),
        "parent_unchanged": True,
    }


def compare_with_parent(
    store: PhotoStore, child_id: str
) -> dict[str, Any]:
    child = store.get(child_id)
    if child is None:
        raise LineageError(f"no photo {child_id!r}")
    parent_id = str(child.get("parent_id") or "")
    if not parent_id:
        raise LineageError(
            f"photo {child_id!r} is an original; there is no parent to compare against."
        )
    parent = store.get(parent_id)
    if parent is None:
        raise LineageError(f"photo {child_id!r} references missing parent {parent_id!r}")
    return compare(parent, child)


# --------------------------------------------------------------------------
# Per-edit credits
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EditCharge:
    """The result of reserving and settling credits for one edit."""

    run_id: str
    instruction_key: str
    reserved: int
    charged: int
    refunded: int
    reservation: Reservation | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "instruction_key": self.instruction_key,
            "reserved": self.reserved,
            "charged": self.charged,
            "refunded": self.refunded,
        }


def edit_run_id(instruction: EditInstruction) -> str:
    """Run id for one edit, derived from the instruction's content.

    Content-derived so a retried edit reconciles with the original charge
    instead of creating a second one — the same reasoning as `run_id` for shoots.
    """
    return "edit_" + instruction_key(instruction).split("_", 1)[-1]


def reserve_edit(
    ledger: CreditLedger,
    *,
    account_id: str,
    instruction: EditInstruction,
    credits: int = CREDITS_PER_PHOTO,
) -> EditCharge:
    """Hold credits for one edit under its own run id.

    Each edit is its own run, so the ledger can answer "what did this edit cost"
    and a per-edit refund cannot disturb the parent shoot's settled balance.
    """
    if credits < 0:
        raise ValueError("credits must be >= 0")
    run_id = edit_run_id(instruction)
    key = f"edit:{instruction_key(instruction)}"
    reservation = ledger.reserve(
        account_id,
        credits,
        run_id=run_id,
        reason=f"edit {instruction.change}",
        idempotency_key=key,
    )
    return EditCharge(
        run_id=run_id,
        instruction_key=instruction_key(instruction),
        reserved=reservation.reserved,
        charged=reservation.charged,
        refunded=reservation.refunded,
        reservation=reservation,
    )


def settle_edit(
    ledger: CreditLedger,
    charge: EditCharge,
    *,
    delivered: bool,
    node_id: str = "",
) -> EditCharge:
    """Settle one edit's reservation: charge if it rendered, refund if not."""
    if charge.reservation is None:
        raise LineageError("cannot settle an edit that was never reserved")
    ledger.settle(
        charge.reservation,
        delivered=1 if delivered else 0,
        file_failed=[] if delivered else [node_id or charge.run_id],
    )
    reservation = charge.reservation
    return EditCharge(
        run_id=charge.run_id,
        instruction_key=charge.instruction_key,
        reserved=reservation.reserved,
        charged=reservation.charged,
        refunded=reservation.refunded,
        reservation=reservation,
    )


def edit_ledger_lines(
    ledger: CreditLedger, *, account_id: str, instruction: EditInstruction
) -> list[dict[str, Any]]:
    """The ledger lines belonging to one edit, for "what did this cost me"."""
    run_id = edit_run_id(instruction)
    return [
        entry.as_dict()
        for entry in ledger.entries(account_id=account_id, run_id=run_id)
    ]


# --------------------------------------------------------------------------
# Delete takes (FR-7.6)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class DeleteResult:
    deleted: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "deleted": list(self.deleted),
            "refused": list(self.refused),
            "reason": self.reason,
        }


def delete_take(
    store: InMemoryPhotoStore, photo_id: str
) -> DeleteResult:
    """Remove one *take*, never cascading into its children.

    Deleting a take that has children would orphan them, so the caller is told to
    delete the children first. The original is deletable only when it has none —
    otherwise the whole take-set loses its root and lineage lookups start raising.
    """
    photo = store.get(photo_id)
    if photo is None:
        return DeleteResult(refused=[photo_id], reason=f"no photo {photo_id!r}")

    children = store.children_of(photo_id)
    if children:
        return DeleteResult(
            refused=[photo_id],
            reason=(
                f"photo {photo_id!r} has {len(children)} edit(s); delete those "
                "first so the take does not lose its lineage."
            ),
        )

    store._photos.pop(photo_id, None)  # noqa: SLF001 - single-owner store
    parent_id = str(photo.get("parent_id") or "")
    if parent_id:
        ids = store._by_parent.get(parent_id, [])  # noqa: SLF001
        if photo_id in ids:
            ids.remove(photo_id)
    store._by_parent.pop(photo_id, None)  # noqa: SLF001
    return DeleteResult(deleted=[photo_id])


__all__ = [
    "COMPARE_FIELDS",
    "DeleteResult",
    "EditCharge",
    "InMemoryPhotoStore",
    "LINEAGE_VERSION",
    "LineageError",
    "PhotoStore",
    "compare",
    "compare_with_parent",
    "delete_take",
    "edit_ledger_lines",
    "edit_run_id",
    "lineage_tree",
    "reserve_edit",
    "root_of",
    "settle_edit",
    "siblings_of",
]
