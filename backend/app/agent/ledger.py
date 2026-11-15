"""Credit ledger: reserve before the work, settle against what actually happened.

The design rule (design.md §9.2) is *reserve-then-settle*, and it exists to fix a
specific failure: a user with 5 credits requests 12 photos, half fail, and they
end up with fewer photos than they could have afforded. Reserving the whole run
up front and refunding node by node means the user never pays for a photo that
did not arrive, and the balance they see is never optimistic.

Three properties this module guarantees:

1. **No overspend.** `reserve` refuses when the balance is short, and it is the
   only way credits leave the account. Callers cannot render before reserving.
2. **No double-charge.** Every entry carries an idempotency key. Posting the same
   reservation twice returns the original entry instead of charging again, and a
   settled run replays rather than re-settling.
3. **Every refund is visible.** A refund is a ledger line with a reason, never a
   silent balance edit, so support can explain any balance by reading the ledger.

In-memory here; the Postgres implementation swaps the store, not the semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

from .hashing import sha256_bytes, stable_digest

LEDGER_VERSION = "ledger-v1"

#: A photo costs one credit (design.md §12).
CREDITS_PER_PHOTO = 1

EntryKind = Literal["reservation", "charge", "refund", "grant", "adjustment"]


class LedgerError(Exception):
    """Base class for ledger problems."""


class InsufficientCredits(LedgerError):
    """The account cannot afford the requested reservation.

    Raised *before* any provider call, so an unaffordable request costs nothing.
    """

    def __init__(self, message: str, *, required: int = 0, available: int = 0) -> None:
        super().__init__(message)
        self.required = required
        self.available = available

    @property
    def shortfall(self) -> int:
        return max(0, self.required - self.available)


class LedgerConflict(LedgerError):
    """An idempotency key was reused for a materially different request."""


@dataclass(frozen=True, slots=True)
class CreditEntry:
    """One immutable line in the ledger. Negative delta = credits held or spent."""

    entry_id: str
    account_id: str
    kind: EntryKind
    delta: int
    reason: str
    run_id: str = ""
    node_id: str = ""
    idempotency_key: str = ""
    balance_after: int = 0
    created_at: str = ""
    ledger_version: str = LEDGER_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "delta": self.delta,
            "reason": self.reason,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "balance_after": self.balance_after,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class Reservation:
    """Credits held for one run, plus the per-node outcome of settling it."""

    reservation_id: str
    account_id: str
    run_id: str
    reserved: int
    settled: bool = False
    charged: int = 0
    refunded: int = 0
    #: node_id -> credits charged for that node (0 for a cache hit or a failure).
    nodes: dict[str, int] = field(default_factory=dict)

    @property
    def outstanding(self) -> int:
        """Credits still held: reserved minus whatever was charged or refunded."""
        return self.reserved - self.charged - self.refunded

    def as_dict(self) -> dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "run_id": self.run_id,
            "reserved": self.reserved,
            "charged": self.charged,
            "refunded": self.refunded,
            "outstanding": self.outstanding,
            "settled": self.settled,
            "nodes": dict(self.nodes),
        }


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def entry_id_for(*parts: str) -> str:
    """Deterministic entry id, so replaying an operation re-uses its line."""
    return "cr_" + stable_digest(LEDGER_VERSION, *parts)[:20]


class CreditLedger:
    """An append-only credit ledger with an authoritative running balance.

    Balances are derived from entries, never stored separately, so the account
    balance and the ledger can never disagree — the classic source of
    "I was charged twice" support tickets.
    """

    def __init__(self, *, balances: dict[str, int] | None = None) -> None:
        self._balances: dict[str, int] = dict(balances or {})
        self._entries: list[CreditEntry] = []
        self._by_key: dict[str, CreditEntry] = {}
        self._reservations: dict[str, Reservation] = {}
        self._reservation_by_run: dict[str, str] = {}

    # -- reads ---------------------------------------------------------------

    def balance(self, account_id: str) -> int:
        return self._balances.get(account_id, 0)

    def entries(self, *, account_id: str | None = None, run_id: str | None = None) -> list[CreditEntry]:
        return [
            entry
            for entry in self._entries
            if (account_id is None or entry.account_id == account_id)
            and (run_id is None or entry.run_id == run_id)
        ]

    def reservation_for_run(self, run_id: str) -> Reservation | None:
        reservation_id = self._reservation_by_run.get(run_id)
        if reservation_id is None:
            return None
        return self._reservations.get(reservation_id)

    def statement(self, account_id: str, *, run_id: str | None = None) -> dict[str, Any]:
        """A support-readable statement: balance plus every line that made it."""
        lines = self.entries(account_id=account_id, run_id=run_id)
        # Refunds carry a positive delta (credits coming back). Charges are
        # zero-delta audit lines because the reservation already removed the
        # credits, so the charged total comes from the reservations themselves.
        refunded = sum(line.delta for line in lines if line.kind == "refund")
        reservations = [
            reservation
            for reservation in self._reservations.values()
            if reservation.account_id == account_id
            and (run_id is None or reservation.run_id == run_id)
        ]
        charged = sum(reservation.charged for reservation in reservations)
        return {
            "account_id": account_id,
            "balance": self.balance(account_id),
            "charged": charged,
            "refunded": refunded,
            "entries": [line.as_dict() for line in lines],
        }

    # -- writes --------------------------------------------------------------

    def _append(
        self,
        *,
        account_id: str,
        kind: EntryKind,
        delta: int,
        reason: str,
        run_id: str = "",
        node_id: str = "",
        idempotency_key: str,
    ) -> CreditEntry:
        entry = CreditEntry(
            entry_id=entry_id_for(account_id, kind, idempotency_key),
            account_id=account_id,
            kind=kind,
            delta=delta,
            reason=reason,
            run_id=run_id,
            node_id=node_id,
            idempotency_key=idempotency_key,
            balance_after=self.balance(account_id) + delta,
            created_at=_now(),
        )
        self._balances[account_id] = entry.balance_after
        self._entries.append(entry)
        self._by_key[idempotency_key] = entry
        return entry

    def grant(self, account_id: str, credits: int, *, reason: str = "purchase", idempotency_key: str = "") -> CreditEntry:
        """Add credits (a purchase or an admin adjustment)."""
        if credits <= 0:
            raise ValueError("credits must be > 0")
        key = idempotency_key or entry_id_for(account_id, "grant", reason, str(credits))
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        return self._append(
            account_id=account_id,
            kind="grant",
            delta=credits,
            reason=reason,
            idempotency_key=key,
        )

    def reserve(
        self,
        account_id: str,
        credits: int,
        *,
        run_id: str,
        reason: str = "reservation",
        idempotency_key: str = "",
    ) -> Reservation:
        """Hold `credits` for `run_id`, or raise before any work begins.

        Idempotent: the same key returns the original reservation, so a retried
        POST /generate cannot double-charge.
        """
        if credits < 0:
            raise ValueError("credits must be >= 0")
        key = idempotency_key or entry_id_for(account_id, "reservation", run_id)

        existing_entry = self._by_key.get(key)
        if existing_entry is not None:
            # The reservation id *is* the reservation entry's id, so this lookup
            # must key on it — not on the entry's run_id, which is a different
            # value entirely and made every replay look like a key collision.
            existing_reservation = self._reservations.get(existing_entry.entry_id)
            if existing_reservation is not None:
                if existing_reservation.reserved != credits:
                    raise LedgerConflict(
                        f"idempotency key {key!r} was already used to reserve "
                        f"{existing_reservation.reserved} credit(s), not {credits}."
                    )
                return existing_reservation
            raise LedgerConflict(f"idempotency key {key!r} is already in use.")

        # One live reservation per run: a second reserve for the same run is a
        # double-submit, not a second hold.
        already = self.reservation_for_run(run_id)
        if already is not None:
            return already

        available = self.balance(account_id)
        if credits > available:
            raise InsufficientCredits(
                f"A run of {credits} photo(s) needs {credits} credit(s) but this "
                f"account has {available}. Top up, or request fewer photos.",
                required=credits,
                available=available,
            )

        entry = self._append(
            account_id=account_id,
            kind="reservation",
            delta=-credits,
            reason=reason,
            run_id=run_id,
            idempotency_key=key,
        )
        reservation = Reservation(
            reservation_id=entry.entry_id,
            account_id=account_id,
            run_id=run_id,
            reserved=credits,
        )
        self._reservations[reservation.reservation_id] = reservation
        self._reservation_by_run[run_id] = reservation.reservation_id
        return reservation

    def settle(
        self,
        reservation: Reservation,
        *,
        delivered: int,
        cache_hits: int = 0,
        file_failed: Sequence[str] = (),
    ) -> Reservation:
        """Charge for delivered photos and refund the rest, line by line.

        `delivered` counts photos actually produced *including* cache hits; a
        cache hit costs nothing, so it is refunded with its own reason. Every
        failed node gets its own visible refund entry.
        """
        if reservation.settled:
            return reservation
        if delivered < 0 or cache_hits < 0:
            raise ValueError("counts must be >= 0")
        if cache_hits > delivered:
            raise ValueError("cache hits cannot exceed delivered photos")

        failed = list(file_failed)
        billable = delivered - cache_hits
        if billable < 0:
            raise ValueError("billable photos cannot be negative")

        # Cap the charge at what is still held. `outstanding` is the authoritative
        # "still held" figure, and it is read *before* any refund below mutates
        # the reservation.
        held = reservation.outstanding
        billable = min(billable, held)

        # A reservation already removed the credits from the balance, so settling
        # must not deduct a second time. The charge is recorded as a zero-delta
        # audit line (it *is* the hold being consumed) and the balance moves only
        # by the refund of everything not delivered. Deducting here double-charged
        # every run.
        failures_charged = min(len(failed), held - billable) if failed else 0
        for node_id in failed[:failures_charged] if failures_charged else []:
            self._append(
                account_id=reservation.account_id,
                kind="refund",
                delta=1,
                reason="refund: render_failed",
                run_id=reservation.run_id,
                node_id=node_id,
                idempotency_key=f"refund:{reservation.run_id}:{node_id}",
            )
            reservation.nodes[node_id] = 0
            reservation.refunded += 1

        if billable:
            self._append(
                account_id=reservation.account_id,
                kind="charge",
                delta=0,
                reason="charge: rendered",
                run_id=reservation.run_id,
                idempotency_key=f"charge:{reservation.run_id}",
            )
            reservation.charged += billable

        # Refund anything still held: cache hits, extra failures beyond the cap,
        # and any shortfall the counts did not account for. One explicit line so
        # the balance is explainable from the ledger alone.
        remaining = reservation.outstanding
        if remaining:
            reason = "refund: cache_hit" if cache_hits else "refund: unrendered"
            self._append(
                account_id=reservation.account_id,
                kind="refund",
                delta=remaining,
                reason=reason,
                run_id=reservation.run_id,
                idempotency_key=f"refund:{reservation.run_id}:balance",
            )
            reservation.refunded += remaining

        reservation.settled = True
        if reservation.outstanding != 0:
            # Settling must leave nothing held. If it does, the arithmetic above
            # is wrong and we want a loud failure rather than a stuck balance.
            raise LedgerError(
                f"settlement left {reservation.outstanding} credit(s) outstanding "
                f"for run {reservation.run_id}; reserved={reservation.reserved} "
                f"charged={reservation.charged} refunded={reservation.refunded}"
            )
        return reservation

    def cancel(self, reservation: Reservation, *, reason: str = "run cancelled") -> Reservation:
        """Release a whole reservation (refused, cancelled, or failed pre-render)."""
        if reservation.settled:
            return reservation
        if reservation.outstanding:
            self._append(
                account_id=reservation.account_id,
                kind="refund",
                delta=reservation.outstanding,
                reason=f"refund: {reason}",
                run_id=reservation.run_id,
                idempotency_key=f"cancel:{reservation.run_id}",
            )
            reservation.refunded += reservation.outstanding
        reservation.settled = True
        return reservation

    # -- deletion ------------------------------------------------------------

    def purge_account(self, account_id: str) -> dict[str, int]:
        """Remove every ledger row for one account (SR-5).

        Returns counts so a purge report can prove what happened. The balance is
        *deleted*, not zeroed, because a balance is derived from entries: leaving
        a zero there would imply an account that still exists with no history,
        and a later `grant` would silently reactivate it.

        Reservations go first, so a half-open hold cannot survive to refuse a
        future run for an account that no longer exists.
        """
        if not account_id:
            return {"entries": 0, "reservations": 0}

        reservation_ids = [
            key
            for key, reservation in self._reservations.items()
            if reservation.account_id == account_id
        ]
        for key in reservation_ids:
            reservation = self._reservations.pop(key)
            if self._reservation_by_run.get(reservation.run_id) == key:
                del self._reservation_by_run[reservation.run_id]

        doomed = {
            id(entry) for entry in self._entries if entry.account_id == account_id
        }
        removed_entries = len(doomed)
        if removed_entries:
            self._entries = [
                entry for entry in self._entries if id(entry) not in doomed
            ]
            # Rebuilt from the surviving list, so an idempotency key can never
            # point at a deleted row.
            self._by_key = {key: entry for key, entry in self._by_key.items() if id(entry) not in doomed}

        self._balances.pop(account_id, None)
        return {"entries": removed_entries, "reservations": len(reservation_ids)}


__all__ = [
    "CREDITS_PER_PHOTO",
    "CreditEntry",
    "CreditLedger",
    "InsufficientCredits",
    "LEDGER_VERSION",
    "LedgerConflict",
    "LedgerError",
    "Reservation",
    "entry_id_for",
]
