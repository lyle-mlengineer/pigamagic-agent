"""One feed card contract, shared by the live stream and the settled feed.

The product shell already has a card shape (`app/api/common.py::serialize_photo`
in the sibling repo) and the frontend already renders it. Steps.md Step 14 asks
for exactly one thing: **a streamed photo and a settled photo must produce
identical card payload shapes.** If the stream invents its own shape, the UI
needs a second card component and a second component is a second set of bugs
(is this one likeable? does it have counts? is `id` the same key?).

So `photo_card()` is the *only* builder. Both paths call it:

    SSE `photo` event  ─┐
                        ├─→ photo_card() ─→ {id, image_url, caption, owner, ...}
    GET /feed /runs   ─┘

Two properties are deliberate:

1. **Fields are always present, never omitted.** A card for a photo with no
   likes still carries `like_count: 0` and `liked_by_me: false`. Optional keys
   are how two shapes drift apart in practice — the frontend then needs
   `card.like_count ?? 0` in one place and `card.like_count` in another.

2. **`developing` is a field, not a different shape.** A streamed card that is
   still being safety-checked sets `developing: true`; it is the same shape, so
   the same component renders it and simply disables the actions
   (design.md §10.3.4).

Cursor pagination is `(created_at, id)` — matching the shell's existing
`pagination_filter`. A single-key cursor on `created_at` alone breaks when two
photos share a timestamp (very likely here: one run renders several photos in
the same second), which is what "stable under inserts" is really testing.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

FEED_VERSION = "feed-v1"

#: Page size bounds, matching the shell's `DEFAULT_PAGE_SIZE`/`MAX_PAGE_SIZE`.
DEFAULT_PAGE_SIZE = 24
MAX_PAGE_SIZE = 60

#: Card states. `developing` cards exist in the feed but are not interactive.
CARD_STATES = ("developing", "ready", "failed")


class FeedError(ValueError):
    """A cursor or card that cannot be honoured."""


# --------------------------------------------------------------------------
# The card
# --------------------------------------------------------------------------


def photo_card(
    photo: Mapping[str, Any],
    *,
    owner: Mapping[str, Any] | None = None,
    location: str = "",
    media_url_template: str = "/api/v1/media/{key}",
    like_count: int = 0,
    comment_count: int = 0,
    liked_by_me: bool = False,
    album_ids: Sequence[str] = (),
    is_public: bool = True,
    developing: bool = False,
) -> dict[str, Any]:
    """Build one feed card from a stored photo (or a streamed photo payload).

    Accepts either a `RenderedPhoto.as_dict()` mapping or the trimmed dict the
    SSE recorder publishes, because both must yield the same card. Missing
    fields degrade to well-defined defaults rather than to absent keys.
    """
    photo_id = str(photo.get("photo_id") or photo.get("id") or "")
    if not photo_id:
        raise FeedError("a feed card needs a photo_id")

    blob = photo.get("blob") or {}
    blob_key = str(blob.get("key") or photo.get("blob_key") or "")
    # An explicit URL wins (a streamed event may carry a signed one); else derive.
    image_url = str(photo.get("image_url") or "")
    if not image_url and blob_key:
        image_url = media_url_template.format(key=blob_key)

    state = "developing" if developing else str(photo.get("state") or "ready")
    if state not in CARD_STATES:
        state = "ready"

    # The shell's feed card is keyed on `media_id` (`serialize_photo`). In this
    # repo a photo *is* its content-addressed blob, so the blob key plays that
    # role; carrying both names makes the card a drop-in for the shell.
    media_id = str(photo.get("media_id") or blob_key)

    return {
        "id": photo_id,
        "photo_id": photo_id,
        "media_id": media_id,
        "shoot_id": str(photo.get("shoot_id") or ""),
        "node_id": str(photo.get("node_id") or ""),
        "image_url": image_url,
        "caption": str(photo.get("caption") or ""),
        "activity": str(photo.get("activity") or ""),
        "moment": str(photo.get("moment") or ""),
        "framing": str(photo.get("framing") or ""),
        "location": location,
        "owner": dict(owner) if owner is not None else None,
        "created_at": str(photo.get("created_at") or ""),
        "like_count": int(like_count),
        "comment_count": int(comment_count),
        "liked_by_me": bool(liked_by_me),
        "album_ids": list(album_ids),
        "is_public": bool(is_public),
        "state": state,
        "developing": state == "developing",
        # Interactive actions are disabled while a card is developing, so the
        # frontend does not have to know which states allow a like.
        "can_interact": state == "ready",
        "parent_id": photo.get("parent_id"),
        "feed_version": FEED_VERSION,
    }


def failed_card(
    *,
    node_id: str,
    reason: str,
    activity: str = "",
    moment: str = "",
    framing: str = "",
    shoot_id: str = "",
    refunded: bool = True,
) -> dict[str, Any]:
    """A card for a shot that will never arrive, so the feed has no silent gap.

    It is *not* a photo card: there is no image and no id to like. It shares the
    `state`/`can_interact` vocabulary plus every photo-card key, so one renderer
    handles both; only `reason` and `refunded` are extra.
    """
    return {
        "id": "",
        "photo_id": "",
        "media_id": "",
        "node_id": node_id,
        "shoot_id": shoot_id,
        "image_url": "",
        "caption": f"Could not develop: {activity or moment or node_id}",
        "activity": activity,
        "moment": moment,
        "framing": framing,
        "location": "",
        "owner": None,
        "created_at": "",
        "like_count": 0,
        "comment_count": 0,
        "liked_by_me": False,
        "album_ids": [],
        "is_public": False,
        "state": "failed",
        "developing": False,
        "can_interact": False,
        "parent_id": None,
        "reason": reason,
        "refunded": bool(refunded),
        "feed_version": FEED_VERSION,
    }


def card_from_event(data: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Card for a streamed `photo` event payload."""
    return photo_card(data, developing=bool(data.get("developing")), **kwargs)


# --------------------------------------------------------------------------
# Cursors
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cursor:
    """A position in the feed: `(created_at, id)`, newest first."""

    created_at: str
    photo_id: str

    def encode(self) -> str:
        payload = json.dumps(
            {"created_at": self.created_at, "id": self.photo_id},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    def as_dict(self) -> dict[str, str]:
        return {"created_at": self.created_at, "id": self.photo_id}


def encode_cursor(created_at: str, photo_id: str) -> str:
    return Cursor(created_at=created_at, photo_id=photo_id).encode()


def decode_cursor(raw: str | None) -> Cursor | None:
    """Decode a cursor. `None`/empty means "from the top".

    A malformed cursor raises `FeedError`, which the route turns into a 422 —
    matching the shell's existing behaviour. Silently restarting from the top
    would make a corrupt cursor look like a refreshed feed and produce
    duplicate cards, which is the bug pagination exists to prevent.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        padded = text + "=" * (-len(text) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        created_at = str(payload["created_at"])
        photo_id = str(payload["id"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, binascii.Error) as exc:
        raise FeedError(f"invalid feed cursor: {exc}") from exc
    if not created_at or not photo_id:
        raise FeedError("invalid feed cursor: empty fields")
    return Cursor(created_at=created_at, photo_id=photo_id)


def _sort_key(card: Mapping[str, Any]) -> tuple[str, str]:
    return (str(card.get("created_at") or ""), str(card.get("id") or ""))


def page_cards(
    cards: Iterable[Mapping[str, Any]],
    *,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """One page of cards, newest first, plus the next cursor.

    Ordering is `(created_at, id)` descending, so two photos created in the same
    second still have a total order. That total order is what makes the cursor
    stable: inserting a photo only ever adds above the cursor, never shifts what
    a later page returns.
    """
    limit = max(1, min(int(limit), MAX_PAGE_SIZE))
    position = decode_cursor(cursor)

    ordered = sorted(
        (dict(card) for card in cards),
        key=_sort_key,
        reverse=True,
    )

    if position is not None:
        boundary = (position.created_at, position.photo_id)
        # Strictly after the cursor in descending order == strictly less.
        ordered = [card for card in ordered if _sort_key(card) < boundary]

    page = ordered[:limit]
    next_cursor = None
    if len(ordered) > limit and page:
        next_cursor = encode_cursor(*_sort_key(page[-1]))
    return page, next_cursor


# --------------------------------------------------------------------------
# Live cards derived from a run
# --------------------------------------------------------------------------


def cards_for_outcome(
    photos: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]] = (),
    *,
    location: str = "",
    owner: Mapping[str, Any] | None = None,
    developing: bool = False,
) -> list[dict[str, Any]]:
    """Cards for one run's photos, in node order, with failures appended.

    Node order (not timestamp order) is deliberate: cards must arrive in the
    order the plan intended — the first four cards decide whether the user keeps
    scrolling (design.md §8), so a photo landing early must not jump ahead of
    the opening shot.
    """
    cards = [
        photo_card(
            photo,
            owner=owner,
            location=location,
            developing=developing,
        )
        for photo in photos
    ]
    cards.sort(key=lambda card: str(card.get("node_id") or ""))
    for failure in failures:
        cards.append(
            failed_card(
                node_id=str(failure.get("node_id") or ""),
                reason=str(failure.get("reason") or "render_failed"),
                activity=str(failure.get("activity") or ""),
                moment=str(failure.get("moment") or ""),
                framing=str(failure.get("framing") or ""),
                shoot_id=str(failure.get("shoot_id") or ""),
                refunded=bool(failure.get("refunded", True)),
            )
        )
    return cards


__all__ = [
    "CARD_STATES",
    "Cursor",
    "DEFAULT_PAGE_SIZE",
    "FEED_VERSION",
    "FeedError",
    "MAX_PAGE_SIZE",
    "card_from_event",
    "cards_for_outcome",
    "decode_cursor",
    "encode_cursor",
    "failed_card",
    "page_cards",
    "photo_card",
]
