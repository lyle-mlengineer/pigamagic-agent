"""Run event log: the replayable record an SSE stream is served from.

The rule that shapes this module (design.md §10.1): the stream is a
**notification channel over persisted state**, never the transport for pixels.
So events are appended to a log with monotonic ids, and a reconnecting client
asks for "everything after id N". That is what makes a page reload or a dropped
connection harmless — the client replays from the log instead of hoping the
socket survived.

Two properties matter and are easy to get wrong:

1. **Ids are per-run and strictly increasing.** A global counter would let a
   reconnecting client skip another run's events and receive ids it cannot
   order. Ids are contiguous (`1, 2, 3...`) so `Last-Event-ID: 7` means exactly
   "everything from 8".

2. **Replay is exclusive of the cursor.** SSE's `Last-Event-ID` names the last
   event *received*, so replay must start strictly after it. Re-sending the
   cursor event would duplicate a feed card — the visible failure the whole
   replay mechanism exists to prevent.

In-memory here; the Redis + `generated_photos` implementation swaps the store,
not the semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Protocol

from .feed import photo_card

EVENTS_VERSION = "events-v1"

EventKind = Literal["run", "photo", "photo_failed", "run_done", "warning"]

#: Event kinds a client must never miss. Used to assert ordering in tests.
TERMINAL_KINDS: frozenset[str] = frozenset({"run_done"})


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One event in a run's stream. `seq` is the SSE `id:` field."""

    seq: int
    run_id: str
    kind: EventKind
    data: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "run_id": self.run_id,
            "kind": self.kind,
            "data": dict(self.data),
            "created_at": self.created_at,
        }

    def to_sse(self) -> str:
        """Render as a standards-compliant SSE frame."""
        return sse_frame(event=self.kind, data=self.data, event_id=self.seq)


def sse_frame(
    *,
    event: str,
    data: dict[str, Any],
    event_id: int | str | None = None,
) -> str:
    """One SSE frame: optional `id:`, `event:`, then a single `data:` line.

    `data` is canonical JSON with newlines collapsed, because a raw newline in a
    `data:` value would be read as a second `data:` line and silently corrupt
    the frame.
    """
    from .hashing import canonical_json

    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    payload = canonical_json(data).replace("\n", " ").replace("\r", " ")
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


class EventLog(Protocol):
    """Append + replay. The seam the SSE endpoint depends on."""

    def append(self, run_id: str, kind: EventKind, data: dict[str, Any]) -> RunEvent: ...

    def events_after(self, run_id: str, seq: int = 0) -> list[RunEvent]: ...

    def has_run(self, run_id: str) -> bool: ...


class InMemoryEventLog:
    """Default log. Per-run contiguous ids, replay strictly after a cursor."""

    def __init__(self) -> None:
        self._events: dict[str, list[RunEvent]] = {}

    def append(self, run_id: str, kind: EventKind, data: dict[str, Any]) -> RunEvent:
        if not run_id:
            raise ValueError("run_id must not be empty")
        stream = self._events.setdefault(run_id, [])
        event = RunEvent(
            seq=len(stream) + 1,
            run_id=run_id,
            kind=kind,
            data=dict(data),
            created_at=_now(),
        )
        stream.append(event)
        return event

    def events_after(self, run_id: str, seq: int = 0) -> list[RunEvent]:
        """Everything with `seq > cursor`. Exclusive, so replay never repeats."""
        stream = self._events.get(run_id, [])
        if seq <= 0:
            return list(stream)
        return [event for event in stream if event.seq > seq]

    def has_run(self, run_id: str) -> bool:
        return run_id in self._events

    def all_events(self, run_id: str) -> list[RunEvent]:
        return list(self._events.get(run_id, []))

    def latest_seq(self, run_id: str) -> int:
        stream = self._events.get(run_id, [])
        return stream[-1].seq if stream else 0

    def run_ids(self) -> list[str]:
        return sorted(self._events)

    def delete(self, run_id: str) -> bool:
        """Drop one run's stream. Used by the account purge (SR-5).

        A purged run's events must go with it: an SSE replay of a deleted run is
        the same privacy failure as serving the deleted photo.
        """
        return self._events.pop(run_id, None) is not None


def parse_last_event_id(raw: str | int | None) -> int:
    """Parse a `Last-Event-ID` header.

    `None`, empty, and unparseable values all mean "from the beginning". A
    garbage header must not 400 the stream: a client that cannot remember where
    it was should get every event, not an error.
    """
    if raw is None:
        return 0
    if isinstance(raw, int):
        return max(0, raw)
    text = str(raw).strip()
    if not text:
        return 0
    try:
        return max(0, int(text))
    except (TypeError, ValueError):
        return 0


class EventRecorder:
    """Adapts `start_run`'s `on_event(kind, payload)` hook onto an `EventLog`.

    `start_run` reports `run` and `run_done`; photo events are derived here from
    the payload so the run module stays unaware of the wire protocol.
    """

    def __init__(self, log: EventLog, run_id: str) -> None:
        self.log = log
        self.run_id = run_id
        self.events: list[RunEvent] = []
        self._last_run_brief: dict[str, Any] | None = None

    def __call__(self, kind: str, payload: dict[str, Any]) -> None:
        run_id = str(payload.get("run_id") or self.run_id)

        if kind == "run":
            brief = _run_brief(payload)
            # `start_run` emits `run` once on creation and once after the credit
            # reservation. When nothing the client can see changed, a second
            # frame is pure noise on the wire (and a duplicate state card), so
            # collapse consecutive identical ones.
            if self._last_run_brief == brief:
                return
            self._last_run_brief = brief
            event = self.log.append(
                run_id,
                "run",
                brief,
            )
            self.events.append(event)
            # A transition that carries photos means the previous step produced
            # them; publish each new photo once, in node order.
            return

        if kind == "run_done":
            event = self.log.append(run_id, "run_done", _done_brief(payload))
            self.events.append(event)
            return

        # Anything else is passed through keyed by its own kind.
        event = self.log.append(run_id, kind, dict(payload))  # type: ignore[arg-type]
        self.events.append(event)

    def publish_photos(self, photos: Iterable[dict[str, Any]]) -> list[RunEvent]:
        """Append one `photo` event per rendered photo, deduped by photo id.

        The dedupe set is read from the **log**, not from `self.events`.
        Photos published here were never passed through `__call__`, so a set
        built from `self.events` would be empty on every call and re-publish
        every photo on a retry — the exact duplicate-card failure this
        mechanism exists to prevent.
        """
        seen = {
            str(event.data.get("photo_id"))
            for event in self.log.events_after(self.run_id, 0)
            if event.kind == "photo"
        }
        appended: list[RunEvent] = []
        for photo in photos:
            photo_id = str(photo.get("photo_id") or "")
            if not photo_id or photo_id in seen:
                # No photo is delivered twice, even if the caller replays.
                continue
            seen.add(photo_id)
            card = photo_card(
                photo,
                media_url_template="/api/v1/media/{key}",
                developing=bool(photo.get("developing", False)),
            )
            event = self.log.append(
                self.run_id,
                "photo",
                {
                    "photo_id": photo_id,
                    "node_id": photo.get("node_id", ""),
                    "caption": photo.get("caption", ""),
                    "activity": photo.get("activity", ""),
                    "moment": photo.get("moment", ""),
                    "framing": photo.get("framing", ""),
                    "subjects": list(photo.get("subjects") or []),
                    "blob_key": (photo.get("blob") or {}).get("key", ""),
                    "image_url": card["image_url"],
                    # The card travels with the event, so a streaming client
                    # renders exactly what a settled client renders. The bare
                    # fields travel too, so a settling reader can rebuild the
                    # card independently instead of trusting this one.
                    "shoot_id": str(photo.get("shoot_id") or ""),
                    "created_at": str(photo.get("created_at") or ""),
                    "parent_id": photo.get("parent_id"),
                    "developing": bool(photo.get("developing", False)),
                    "card": card,
                },
            )
            appended.append(event)
            self.events.append(event)
        return appended

    def publish_failures(self, failures: Iterable[dict[str, Any]]) -> list[RunEvent]:
        """Append one `photo_failed` event per failed node."""
        appended: list[RunEvent] = []
        seen = {
            str(event.data.get("node_id"))
            for event in self.log.events_after(self.run_id, 0)
            if event.kind == "photo_failed"
        }
        for failure in failures:
            node_id = str(failure.get("node_id") or "")
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            event = self.log.append(
                self.run_id,
                "photo_failed",
                {
                    "node_id": node_id,
                    "reason": failure.get("reason", ""),
                    "detail": failure.get("detail", ""),
                    "refunded": True,
                },
            )
            self.events.append(event)
            appended.append(event)
        return appended


def _run_brief(payload: dict[str, Any]) -> dict[str, Any]:
    """The `run` event from design.md §10.2: status, plan, total."""
    return {
        "run_id": payload.get("run_id", ""),
        "status": payload.get("status", ""),
        "plan_id": payload.get("plan_id", ""),
        "total": payload.get("requested", 0),
        "spent_credits": payload.get("spent_credits", 0),
        "warnings": list(payload.get("warnings") or []),
    }


def _done_brief(payload: dict[str, Any]) -> dict[str, Any]:
    """The `run_done` event: what the user's one-click retry needs."""
    return {
        "run_id": payload.get("run_id", ""),
        "status": payload.get("status", ""),
        "generated": len(payload.get("photo_ids") or []),
        "failed": max(
            0,
            int(payload.get("requested", 0)) - len(payload.get("photo_ids") or []),
        ),
        "spent_credits": payload.get("spent_credits", 0),
        "refunded_credits": payload.get("refunded_credits", 0),
        "error": payload.get("error", ""),
    }


def _image_url(blob_key: str) -> str:
    """Media URL for a stored photo. Empty key -> empty URL, never a broken one.

    Kept for the `image_url` field on events built outside `photo_card`.
    """
    if not blob_key:
        return ""
    return f"/api/v1/media/{blob_key}"


__all__ = [
    "EVENTS_VERSION",
    "EventKind",
    "EventLog",
    "EventRecorder",
    "InMemoryEventLog",
    "RunEvent",
    "TERMINAL_KINDS",
    "parse_last_event_id",
    "sse_frame",
]
