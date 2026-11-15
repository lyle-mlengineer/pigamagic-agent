"""Step 14 gates: one card contract, and pagination stable under inserts.

steps.md names two assertions:

* a settled photo and a streamed photo produce **identical card payload shapes**
* cursor pagination **stays stable while photos are inserted**

The first is the reason `photo_card()` exists at all: two builders drift, and a
drifted shape means a second frontend component and a second set of bugs.
"""

from __future__ import annotations

import base64
import json

import pytest

from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.events import EventRecorder, InMemoryEventLog
from pigamagic_agent.agent.feed import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    Cursor,
    FeedError,
    card_from_event,
    cards_for_outcome,
    decode_cursor,
    encode_cursor,
    failed_card,
    page_cards,
    photo_card,
)
from pigamagic_agent.agent.ledger import CreditLedger
from pigamagic_agent.agent.run import InMemoryRunStore
from pigamagic_agent.agent.providers.image_fixture import FixtureImageProvider
from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider
from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider
from pigamagic_agent.agent.storage import LocalBlobStore
from pigamagic_agent.api.runs import ProviderBundle, build_dependencies, create_app

ACCOUNT = "acct_feed"
API = "/api/v1"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _stored_photo(photo_id="ph_1", node_id="s00", created_at="2026-01-01T00:00:00+00:00"):
    """A photo mapping in the shape `RenderedPhoto.as_dict()` produces."""
    return {
        "photo_id": photo_id,
        "shoot_id": "golden-wedding",
        "node_id": node_id,
        "blob": {
            "key": f"wedding/aa/bb/{photo_id}.png",
            "sha256": "a" * 64,
            "size": 1234,
            "content_type": "image/png",
            "bucket": "",
            "url": "",
        },
        "moment": "getting ready",
        "activity": "getting ready",
        "subjects": ["p1"],
        "framing": "wide",
        "caption": "Getting Ready",
        "seed": 42,
        "provider": "fixture",
        "model": "fixture",
        "prompt_hash": "abc",
        "prompt_version": "v1",
        "cost": {"credits": 1, "usd": 0.0},
        "width": 1024,
        "height": 1024,
        "parent_id": None,
        "created_at": created_at,
        "attempts": 1,
    }


def _deps(tmp_path, *, credits: int = 100):
    return build_dependencies(
        settings=AgentSettings(
            media_dir=str(tmp_path / "media"),
            target_shot_count=4,
            max_shots=16,
            image_budget_usd=50.0,
        ),
        ledger=CreditLedger(balances={ACCOUNT: credits}),
        store=InMemoryRunStore(),
        events=InMemoryEventLog(),
        providers=ProviderBundle(
            llm=FixtureLLMProvider(),
            vision=FixtureVisionProvider(),
            web=FixtureWebProvider(),
            image=FixtureImageProvider(),
            blobs=LocalBlobStore(tmp_path / "blobs"),
        ),
    )


def _sse_frames(text: str):
    frames = []
    for block in text.split("\n\n"):
        if not block.strip() or block.lstrip().startswith(":"):
            continue
        frame = {}
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            key, _, value = line.partition(": ")
            frame[key] = value
        if "event" in frame:
            frame["data"] = json.loads(frame.get("data", "{}"))
            frames.append(frame)
    return frames


# --------------------------------------------------------------------------
# GATE 1: settled card == streamed card
# --------------------------------------------------------------------------


def test_gate_settled_and_streamed_cards_are_identical(tmp_path):
    """The gate, through the real HTTP surface: stream, then compare."""
    from fastapi.testclient import TestClient

    deps = _deps(tmp_path)
    client = TestClient(create_app(deps))
    run_id = client.post(
        f"{API}/shoots/golden-wedding/generate",
        json={"account_id": ACCOUNT, "brief": "wedding", "shot_count": 3},
    ).json()["run_id"]

    with client.stream("GET", f"{API}/runs/{run_id}/events") as response:
        streamed = [
            frame["data"]["card"]
            for frame in _sse_frames("".join(response.iter_text()))
            if frame["event"] == "photo"
        ]

    settled = client.get(f"{API}/runs/{run_id}").json()["cards"]

    assert len(streamed) == len(settled) == 3
    # Not just the same keys: the same values, in the same order.
    assert streamed == settled
    assert set(streamed[0]) == set(settled[0])


def test_gate_card_shapes_match_for_a_minimal_event_payload():
    """Both builders accept the different inputs they really receive.

    The stream carries `RenderedPhoto.as_dict()`; a caller building a card from
    an event may only have the trimmed fields. The *shape* must not change.
    """
    full = photo_card(_stored_photo())
    trimmed = card_from_event({"photo_id": "ph_1", "node_id": "s00"})

    assert set(full) == set(trimmed)
    # Absent values degrade to defaults, never to missing keys.
    assert trimmed["like_count"] == 0
    assert trimmed["liked_by_me"] is False
    assert trimmed["album_ids"] == []
    assert trimmed["caption"] == ""


def test_card_requires_a_photo_id():
    with pytest.raises(FeedError):
        photo_card({"node_id": "s00"})


def test_live_card_is_interactive_and_developing_card_is_not():
    ready = photo_card(_stored_photo())
    developing = photo_card(_stored_photo(), developing=True)
    assert ready["can_interact"] is True and ready["developing"] is False
    assert developing["can_interact"] is False and developing["developing"] is True
    # Same shape either way: one component, different affordances.
    assert set(ready) == set(developing)


def test_image_url_is_derived_from_the_blob_key():
    card = photo_card(_stored_photo())
    assert card["image_url"] == "/api/v1/media/wedding/aa/bb/ph_1.png"

    # An explicit URL wins, because a streamed event may carry a signed one.
    signed = photo_card({**_stored_photo(), "image_url": "https://cdn/x.png?sig=1"})
    assert signed["image_url"] == "https://cdn/x.png?sig=1"

    # A configurable template is honoured.
    cdn = photo_card(_stored_photo(), media_url_template="https://cdn.example/{key}")
    assert cdn["image_url"] == "https://cdn.example/wedding/aa/bb/ph_1.png"


def test_streamed_event_carries_the_card(tmp_path):
    """The card must travel with the event, not be rebuilt by the client."""
    log = InMemoryEventLog()
    recorder = EventRecorder(log, "run_1")
    recorder.publish_photos([_stored_photo()])
    event = log.all_events("run_1")[0]
    assert event.data["card"]["id"] == "ph_1"
    assert event.data["image_url"] == event.data["card"]["image_url"]


def test_card_is_a_superset_of_the_shell_card_contract():
    """Every field the product shell's `serialize_photo` returns must exist.

    The shell already renders this shape (app/api/common.py in the sibling
    repo). A missing field means the frontend needs a second component for
    generated photos, which is the whole thing Step 14 avoids.
    """
    shell_fields = {
        "id",
        "shoot_id",
        "media_id",
        "image_url",
        "caption",
        "is_public",
        "created_at",
        "owner",
        "like_count",
        "comment_count",
        "liked_by_me",
        "album_ids",
    }
    card = photo_card(_stored_photo())
    assert shell_fields <= set(card)
    # `media_id` defaults to the blob key: here a photo *is* its content-addressed
    # blob, so the two names refer to the same thing.
    assert card["media_id"] == "wedding/aa/bb/ph_1.png"
    assert card["image_url"] == f"/api/v1/media/{card['media_id']}"


def test_failed_card_matches_the_photo_card_shape():
    """One renderer handles both, so the shapes must not fork.

    Only the failure-specific `reason`/`refunded` are extra; everything else is
    present, so a card renderer never has to branch on card type.
    """
    photo_keys = set(photo_card(_stored_photo()))
    failed_keys = set(failed_card(node_id="s1", reason="render_failed"))
    assert photo_keys <= failed_keys
    assert failed_keys - photo_keys == {"reason", "refunded"}


def test_cards_for_outcome_are_in_node_order_and_place_failures_last():
    photos = [
        _stored_photo("ph_b", "s02"),
        _stored_photo("ph_a", "s00"),
        _stored_photo("ph_c", "s01"),
    ]
    cards = cards_for_outcome(
        photos, [{"node_id": "s03", "reason": "render_failed"}], location="The Pavilion"
    )
    # Node order is the plan's order: the opening shots must not arrive late
    # just because another photo rendered first.
    assert [card["node_id"] for card in cards] == ["s00", "s01", "s02", "s03"]
    assert cards[-1]["state"] == "failed"
    assert cards[-1]["can_interact"] is False
    assert all(card["location"] == "The Pavilion" for card in cards[:3])


def test_failed_card_is_not_a_likeable_photo():
    card = failed_card(node_id="s07", reason="render_failed", activity="the vows")
    assert card["id"] == "" and card["image_url"] == ""
    assert card["state"] == "failed" and card["can_interact"] is False
    assert "the vows" in card["caption"]
    assert card["refunded"] is True


# --------------------------------------------------------------------------
# Cursors
# --------------------------------------------------------------------------


def test_cursor_round_trips():
    encoded = encode_cursor("2026-01-01T00:00:00+00:00", "ph_1")
    cursor = decode_cursor(encoded)
    assert cursor == Cursor("2026-01-01T00:00:00+00:00", "ph_1")


def test_empty_cursor_means_from_the_top():
    assert decode_cursor(None) is None
    assert decode_cursor("") is None
    assert decode_cursor("   ") is None


def test_malformed_cursor_raises_rather_than_restarting():
    """A corrupt cursor must not silently behave like a fresh feed.

    Restarting from the top looks like a refresh and produces duplicate cards —
    the exact failure pagination exists to prevent.
    """
    for bad in ["not-base64!!", "Zm9v", base64.urlsafe_b64encode(b'{"id":"x"}').decode()]:
        with pytest.raises(FeedError):
            decode_cursor(bad)


def test_cursor_encoding_matches_the_shell_convention():
    """Same wire format as `app/api/common.py::page_records` in the shell."""
    encoded = encode_cursor("2026-01-01T00:00:00+00:00", "ph_1")
    padded = encoded + "=" * (-len(encoded) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded))
    assert payload == {"created_at": "2026-01-01T00:00:00+00:00", "id": "ph_1"}
    assert "=" not in encoded  # padding stripped, as the shell does


# --------------------------------------------------------------------------
# GATE 2: pagination stable under inserts
# --------------------------------------------------------------------------


def _card(photo_id, created_at):
    return photo_card(_stored_photo(photo_id, created_at=created_at))


def test_page_is_newest_first_with_a_next_cursor():
    cards = [_card(f"ph_{i}", f"2026-01-0{i}T00:00:00+00:00") for i in range(1, 6)]
    page, next_cursor = page_cards(cards, limit=2)
    assert [c["id"] for c in page] == ["ph_5", "ph_4"]
    assert next_cursor is not None


def test_last_page_has_no_cursor():
    page, next_cursor = page_cards([_card("ph_1", "2026-01-01T00:00:00+00:00")], limit=5)
    assert [c["id"] for c in page] == ["ph_1"]
    assert next_cursor is None


def test_same_timestamp_photos_still_have_a_total_order():
    """One run renders several photos in the same second — the real case."""
    stamp = "2026-01-01T00:00:00+00:00"
    cards = [_card("ph_a", stamp), _card("ph_b", stamp), _card("ph_c", stamp)]
    page, next_cursor = page_cards(cards, limit=2)
    assert [c["id"] for c in page] == ["ph_c", "ph_b"]
    # The tie-break on id is what makes the cursor exact rather than approximate.
    second, _ = page_cards(cards, limit=2, cursor=next_cursor)
    assert [c["id"] for c in second] == ["ph_a"]


def test_gate_inserting_a_photo_does_not_shift_a_page_boundary():
    """Insert photos above the cursor; the next page must not change.

    This is FR-5.6: the user is scrolling while a live run appends cards. If the
    cursor were an offset, every insert would push one card from page 2 to page 1
    (a duplicate) or skip one entirely.
    """
    base = [_card(f"ph_{i}", f"2026-01-0{i}T00:00:00+00:00") for i in range(1, 6)]

    page1, cursor = page_cards(base, limit=2)
    page2_before, _ = page_cards(base, limit=2, cursor=cursor)
    assert [c["id"] for c in page2_before] == ["ph_3", "ph_2"]

    # Two brand-new photos land at the top of the feed while the user reads.
    inserted = base + [
        _card("ph_new_a", "2026-01-09T00:00:00+00:00"),
        _card("ph_new_b", "2026-01-09T00:00:01+00:00"),
    ]
    page2_after, _ = page_cards(inserted, limit=2, cursor=cursor)

    assert page2_after == page2_before, "an insert must not disturb the next page"
    # And page 1 (refetched) shows the new cards, as a live feed should.
    fresh_page1, _ = page_cards(inserted, limit=2)
    assert [c["id"] for c in fresh_page1] == ["ph_new_b", "ph_new_a"]
    # No card appears in both page 1 and page 2.
    assert not ({c["id"] for c in page1} & {c["id"] for c in page2_after})


def test_walking_the_whole_feed_visits_every_card_once():
    stamp_collisions = [
        _card("ph_a", "2026-01-01T00:00:00+00:00"),
        _card("ph_b", "2026-01-01T00:00:00+00:00"),
        _card("ph_c", "2026-01-02T00:00:00+00:00"),
        _card("ph_d", "2026-01-03T00:00:00+00:00"),
        _card("ph_e", "2026-01-03T00:00:00+00:00"),
    ]
    seen: list[str] = []
    cursor = None
    for _ in range(10):
        page, cursor = page_cards(stamp_collisions, limit=2, cursor=cursor)
        seen.extend(c["id"] for c in page)
        if cursor is None:
            break
    assert sorted(seen) == ["ph_a", "ph_b", "ph_c", "ph_d", "ph_e"]
    assert len(seen) == len(set(seen)), "no card may appear twice while paging"


def test_limit_is_bounded():
    cards = [_card(f"ph_{i:03d}", "2026-01-01T00:00:00+00:00") for i in range(200)]
    page, _ = page_cards(cards, limit=10_000)
    assert len(page) == MAX_PAGE_SIZE
    page, _ = page_cards(cards, limit=0)
    assert len(page) == 1


# --------------------------------------------------------------------------
# The /feed endpoint
# --------------------------------------------------------------------------


def test_feed_endpoint_pages_a_real_run(tmp_path):
    from fastapi.testclient import TestClient

    deps = _deps(tmp_path)
    client = TestClient(create_app(deps))
    client.post(
        f"{API}/shoots/golden-wedding/generate",
        json={"account_id": ACCOUNT, "brief": "wedding", "shot_count": 4},
    )

    first = client.get(f"{API}/feed?limit=2")
    assert first.status_code == 200
    body = first.json()
    assert len(body["items"]) == 2
    assert body["has_more"] is True
    assert body["next_cursor"]
    assert first.headers["X-Next-Cursor"] == body["next_cursor"]

    second = client.get(f"{API}/feed?limit=2&cursor={body['next_cursor']}").json()
    assert len(second["items"]) == 2
    assert not ({c["id"] for c in body["items"]} & {c["id"] for c in second["items"]})
    assert second["has_more"] is False
    assert second["next_cursor"] is None


def test_feed_endpoint_rejects_a_bad_cursor(tmp_path):
    from fastapi.testclient import TestClient

    response = TestClient(create_app(_deps(tmp_path))).get(f"{API}/feed?cursor=%%%")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_cursor"


def test_feed_cards_match_the_run_cards(tmp_path):
    """The feed and the run endpoint must not disagree about a photo."""
    from fastapi.testclient import TestClient

    deps = _deps(tmp_path)
    client = TestClient(create_app(deps))
    run_id = client.post(
        f"{API}/shoots/golden-wedding/generate",
        json={"account_id": ACCOUNT, "brief": "wedding", "shot_count": 3},
    ).json()["run_id"]

    from_run = {c["id"]: c for c in client.get(f"{API}/runs/{run_id}").json()["cards"]}
    from_feed = {c["id"]: c for c in client.get(f"{API}/feed?limit=50").json()["items"]}
    assert set(from_run) <= set(from_feed)
    for photo_id, card in from_run.items():
        assert from_feed[photo_id] == card
