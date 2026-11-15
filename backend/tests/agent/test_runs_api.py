"""Step 13 gates: the runs API and its SSE stream.

steps.md names three assertions:

* an SSE client receives `run`, `photo`, and `run_done` events **in order**
* reconnecting with `Last-Event-ID` replays **only** missed events
* **no photo is delivered twice**

Everything runs offline: fixture providers, in-memory stores, tmp blob dir.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.events import (
    EventRecorder,
    InMemoryEventLog,
    RunEvent,
    parse_last_event_id,
    sse_frame,
)
from pigamagic_agent.agent.ledger import CREDITS_PER_PHOTO, CreditLedger
from pigamagic_agent.agent.pipeline import load_brief
from pigamagic_agent.agent.providers.image_fixture import FixtureImageProvider
from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider
from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider
from pigamagic_agent.agent.run import InMemoryRunStore, start_run
from pigamagic_agent.agent.storage import LocalBlobStore
from pigamagic_agent.api.runs import (
    ProviderBundle,
    build_dependencies,
    create_app,
    run_payload,
    stream_events,
)

ACCOUNT = "acct_api"
API = "/api/v1"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _settings(tmp_path, **overrides) -> AgentSettings:
    values = {
        "media_dir": str(tmp_path / "media"),
        "target_shot_count": 6,
        "max_shots": 16,
        "image_budget_usd": 50.0,
    }
    values.update(overrides)
    return AgentSettings(**values)


def _deps(tmp_path, *, credits: int = 100, image=None):
    settings = _settings(tmp_path)
    return build_dependencies(
        settings=settings,
        ledger=CreditLedger(balances={ACCOUNT: credits}),
        store=InMemoryRunStore(),
        events=InMemoryEventLog(),
        providers=ProviderBundle(
            llm=FixtureLLMProvider(),
            vision=FixtureVisionProvider(),
            web=FixtureWebProvider(),
            image=image or FixtureImageProvider(),
            blobs=LocalBlobStore(tmp_path / "blobs"),
        ),
    )


def _client(deps) -> TestClient:
    return TestClient(create_app(deps))


def _generate(client: TestClient, *, brief: str = "wedding", shot_count: int = 3, key: str | None = None):
    headers = {"Idempotency-Key": key} if key else {}
    body: dict[str, Any] = {
        "account_id": ACCOUNT,
        "brief": brief,
        "shot_count": shot_count,
    }
    return client.post(f"{API}/shoots/golden-wedding/generate", json=body, headers=headers)


def _sse_frames(text: str) -> list[dict[str, Any]]:
    """Parse a raw SSE body into frames, skipping comments and blanks."""
    frames: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        frame: dict[str, Any] = {}
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(": ")
            frame[field] = value
        if "event" in frame:
            frame["data"] = json.loads(frame.get("data", "{}"))
            frames.append(frame)
    return frames


# --------------------------------------------------------------------------
# Event log unit tests
# --------------------------------------------------------------------------


def test_event_ids_are_per_run_and_contiguous():
    log = InMemoryEventLog()
    a = [log.append("run_a", "run", {"n": i}) for i in range(3)]
    b = [log.append("run_b", "run", {"n": i}) for i in range(2)]
    assert [event.seq for event in a] == [1, 2, 3]
    # A second run does not continue the first run's sequence: ids are per-run,
    # so a cursor from one run can never skip another run's events.
    assert [event.seq for event in b] == [1, 2]


def test_replay_is_exclusive_of_the_cursor():
    log = InMemoryEventLog()
    for i in range(5):
        log.append("run_x", "run", {"n": i})
    assert [e.seq for e in log.events_after("run_x", 0)] == [1, 2, 3, 4, 5]
    # Last-Event-ID: 3 names the last event *received*, so 3 must not repeat.
    assert [e.seq for e in log.events_after("run_x", 3)] == [4, 5]
    assert log.events_after("run_x", 5) == []


def test_parse_last_event_id_never_errors_on_garbage():
    assert parse_last_event_id(None) == 0
    assert parse_last_event_id("") == 0
    assert parse_last_event_id("not-a-number") == 0
    assert parse_last_event_id("-4") == 0
    assert parse_last_event_id("7") == 7
    assert parse_last_event_id(7) == 7


def test_sse_frame_shape_and_newline_safety():
    frame = sse_frame(event="photo", data={"caption": "a\nb"}, event_id=4)
    assert frame.startswith("id: 4\nevent: photo\ndata: ")
    assert frame.endswith("\n\n")
    # A raw newline in a data value would be read as a second data line.
    assert frame.count("data: ") == 1
    assert "\n\n" not in frame[:-2]


def test_recorder_dedupes_photo_ids():
    log = InMemoryEventLog()
    recorder = EventRecorder(log, "run_1")
    photo = {"photo_id": "ph_a", "node_id": "s00", "blob": {"key": "k/a.png"}}
    assert len(recorder.publish_photos([photo])) == 1
    # Replaying the same photo must not append a second event: the feed card
    # would appear twice.
    assert recorder.publish_photos([photo]) == []
    assert [e.kind for e in log.all_events("run_1")] == ["photo"]


# --------------------------------------------------------------------------
# POST /generate
# --------------------------------------------------------------------------


def test_generate_returns_a_run_and_persists_it(tmp_path):
    deps = _deps(tmp_path)
    response = _generate(_client(deps), shot_count=3)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] in {"done", "partial"}
    assert body["requested"] == 3
    assert len(body["photo_ids"]) == 3
    assert body["events_url"].endswith("/events")

    # GET returns the same persisted state, which is what survives a reload.
    fetched = _client(deps).get(f"{API}/runs/{body['run_id']}").json()
    assert fetched["run_id"] == body["run_id"]
    assert fetched["status"] == body["status"]
    assert set(fetched["photo_urls"]) == set(body["photo_ids"])


def test_generate_rejects_unknown_brief_with_a_typed_error(tmp_path):
    client = _client(_deps(tmp_path))
    response = client.post(
        f"{API}/shoots/s1/generate", json={"account_id": ACCOUNT, "brief": "not-a-brief"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "brief_not_found"


def test_generate_requires_an_account(tmp_path):
    client = _client(_deps(tmp_path))
    response = client.post(f"{API}/shoots/s1/generate", json={"brief": "wedding"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "missing_account"


def test_generate_rejects_malformed_json(tmp_path):
    client = _client(_deps(tmp_path))
    response = client.post(
        f"{API}/shoots/s1/generate",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


def test_generate_refuses_insufficient_credits_before_any_work(tmp_path):
    calls: list[str] = []

    class CountingImage(FixtureImageProvider):
        async def generate(self, **kwargs):  # type: ignore[override]
            calls.append("image")
            return await super().generate(**kwargs)

    deps = _deps(tmp_path, credits=2, image=CountingImage())
    response = _generate(_client(deps), shot_count=5)
    assert response.status_code == 402
    error = response.json()["error"]
    assert error["code"] == "insufficient_credits"
    assert error["required"] == 5 * CREDITS_PER_PHOTO
    assert error["available"] == 2
    # The whole point: an unaffordable request costs zero provider calls.
    assert calls == []
    assert deps.ledger.balance(ACCOUNT) == 2


def test_generate_is_idempotent_on_the_key(tmp_path):
    deps = _deps(tmp_path)
    client = _client(deps)
    first = _generate(client, shot_count=2, key="key-1").json()
    second = _generate(client, shot_count=2, key="key-1").json()

    assert first["run_id"] == second["run_id"]
    assert first["photo_ids"] == second["photo_ids"]
    # Exactly one charge for two photos, despite two POSTs.
    charges = [
        entry
        for entry in deps.ledger.entries(account_id=ACCOUNT)
        if entry.kind == "charge"
    ]
    assert len(charges) == 1
    assert deps.ledger.balance(ACCOUNT) == 100 - 2 * CREDITS_PER_PHOTO


def test_generate_rejects_a_second_run_on_the_same_active_shoot(tmp_path):
    deps = _deps(tmp_path)
    # An active run on this shoot, planted directly: the endpoint must not
    # start a second one and double-charge.
    from pigamagic_agent.agent.run import GenerationRun

    deps.store.put(
        GenerationRun(
            run_id="run_live",
            shoot_id="golden-wedding",
            account_id=ACCOUNT,
            status="rendering",
        )
    )
    if hasattr(deps.store, "track_shoot"):
        deps.store.track_shoot(deps.store.get("run_live"))

    response = _generate(_client(deps), shot_count=2)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "run_in_progress"


def test_shot_count_above_the_cap_is_refused(tmp_path):
    client = _client(_deps(tmp_path))
    response = _generate(client, shot_count=99)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "shot_count_out_of_range"


# --------------------------------------------------------------------------
# GET /runs/{id}
# --------------------------------------------------------------------------


def test_get_unknown_run_is_404(tmp_path):
    response = _client(_deps(tmp_path)).get(f"{API}/runs/run_missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "run_not_found"


def test_run_payload_carries_no_absolute_hosts(tmp_path):
    deps = _deps(tmp_path)
    body = _generate(_client(deps), shot_count=2).json()
    blob = json.dumps(body)
    # A stored run must never freeze a host into it; hosts change, URIs don't.
    assert "http://" not in blob and "https://" not in blob


# --------------------------------------------------------------------------
# SSE gates
# --------------------------------------------------------------------------


def test_sse_stream_emits_run_photo_and_run_done_in_order(tmp_path):
    deps = _deps(tmp_path)
    client = _client(deps)
    run_id = _generate(client, shot_count=3).json()["run_id"]

    with client.stream("GET", f"{API}/runs/{run_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())

    frames = _sse_frames(text)
    kinds = [frame["event"] for frame in frames]

    assert kinds[0] == "run", f"first event must be `run`, got {kinds}"
    assert kinds[-1] == "run_done", f"last event must be `run_done`, got {kinds}"
    assert kinds.count("photo") == 3
    # Every photo arrives before the terminal event, so a client that stops at
    # `run_done` already has every card it will ever get.
    assert kinds.index("run_done") > max(i for i, k in enumerate(kinds) if k == "photo")

    # Ids are contiguous from 1 and strictly increasing.
    ids = [int(frame["id"]) for frame in frames]
    assert ids == sorted(ids) == list(range(1, len(ids) + 1))

    photo_ids = [f["data"]["photo_id"] for f in frames if f["event"] == "photo"]
    assert len(photo_ids) == len(set(photo_ids)), "no photo may be delivered twice"


def test_sse_stream_ends_at_the_terminal_event(tmp_path):
    deps = _deps(tmp_path)
    client = _client(deps)
    run_id = _generate(client, shot_count=2).json()["run_id"]

    with client.stream("GET", f"{API}/runs/{run_id}/events") as response:
        text = "".join(response.iter_text())

    # The connection closed on its own: an open stream per client would leak.
    assert text.rstrip().endswith("}")
    assert _sse_frames(text)[-1]["event"] == "run_done"


def test_last_event_id_replays_only_the_missed_events(tmp_path):
    deps = _deps(tmp_path)
    client = _client(deps)
    run_id = _generate(client, shot_count=3).json()["run_id"]

    with client.stream("GET", f"{API}/runs/{run_id}/events") as response:
        all_frames = _sse_frames("".join(response.iter_text()))

    # Drop the connection *after* the first photo, as a flaky phone would.
    cursor = int(
        next(f for f in all_frames if f["event"] == "photo")["id"]
    )

    with client.stream(
        "GET",
        f"{API}/runs/{run_id}/events",
        headers={"Last-Event-ID": str(cursor)},
    ) as response:
        replay = _sse_frames("".join(response.iter_text()))

    assert [int(f["id"]) for f in replay] == [
        int(f["id"]) for f in all_frames if int(f["id"]) > cursor
    ]
    # Strictly after the cursor: no duplicate id, therefore no duplicate card.
    assert all(int(f["id"]) > cursor for f in replay)
    # The photo the client already has must not arrive again.
    already = {
        f["data"]["photo_id"]
        for f in all_frames
        if f["event"] == "photo" and int(f["id"]) <= cursor
    }
    replayed = {f["data"]["photo_id"] for f in replay if f["event"] == "photo"}
    assert already
    assert not (already & replayed), "a replayed photo must not repeat"
    # And the terminal event is still delivered, so the client knows it is done.
    assert replay[-1]["event"] == "run_done"


def test_last_event_id_via_query_parameter_works_too(tmp_path):
    deps = _deps(tmp_path)
    client = _client(deps)
    run_id = _generate(client, shot_count=3).json()["run_id"]

    with client.stream("GET", f"{API}/runs/{run_id}/events") as response:
        all_frames = _sse_frames("".join(response.iter_text()))
    cursor = int(all_frames[0]["id"])

    # EventSource cannot set headers on reconnect, so this is the real path.
    with client.stream(
        "GET", f"{API}/runs/{run_id}/events?last_event_id={cursor}"
    ) as response:
        replay = _sse_frames("".join(response.iter_text()))
    assert all(int(f["id"]) > cursor for f in replay)
    assert replay


def test_events_for_unknown_run_is_404(tmp_path):
    response = _client(_deps(tmp_path)).get(f"{API}/runs/run_missing/events")
    assert response.status_code == 404


def test_resuming_a_fully_consumed_stream_sends_nothing_new(tmp_path):
    deps = _deps(tmp_path)
    client = _client(deps)
    run_id = _generate(client, shot_count=2).json()["run_id"]

    with client.stream("GET", f"{API}/runs/{run_id}/events") as response:
        all_frames = _sse_frames("".join(response.iter_text()))
    last = int(all_frames[-1]["id"])

    with client.stream(
        "GET", f"{API}/runs/{run_id}/events", headers={"Last-Event-ID": str(last)}
    ) as response:
        replay = _sse_frames("".join(response.iter_text()))
    assert replay == []


# --------------------------------------------------------------------------
# Live streaming (run in progress, events arriving while connected)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_mid_run_delivers_new_events_as_they_land(tmp_path):
    """The stream must keep polling while a run is active, not just replay."""
    deps = _deps(tmp_path)
    log = deps.events
    store = deps.store

    from pigamagic_agent.agent.run import GenerationRun

    run = GenerationRun(
        run_id="run_live", shoot_id="s1", account_id=ACCOUNT, status="rendering"
    )
    store.put(run)
    log.append("run_live", "run", {"run_id": "run_live", "status": "rendering"})

    collected: list[RunEvent] = []

    async def consume() -> None:
        async for frame in stream_events(
            deps=deps, run=run, last_event_id=0, poll_seconds=0.01, keepalive_polls=1000
        ):
            if frame.startswith("id: "):
                seq = int(frame.split("\n", 1)[0][4:])
                collected.append(log.all_events("run_live")[seq - 1])

    task = __import__("asyncio").create_task(consume())
    await __import__("asyncio").sleep(0.05)

    # Events that land *after* the connection opened must be delivered.
    recorder = EventRecorder(log, "run_live")
    recorder.publish_photos([{"photo_id": "ph_live", "node_id": "s00"}])
    recorder("run_done", {"run_id": "run_live", "status": "done", "photo_ids": ["ph_live"]})
    run.status = "done"

    await __import__("asyncio").wait_for(task, timeout=2.0)
    assert [event.kind for event in collected] == ["run", "photo", "run_done"]


def test_media_url_template_is_configurable(tmp_path):
    deps = _deps(tmp_path)
    deps.media_url_template = "https://cdn.example/{key}"
    body = run_payload(
        deps.store.get(_generate(_client(deps), shot_count=1).json()["run_id"]), deps
    )
    assert all(url.startswith("https://cdn.example/") for url in body["photo_urls"].values())


def test_start_run_photos_are_published_exactly_once(tmp_path):
    """`execute_run` publishes photos from the outcome, not from transitions."""
    deps = _deps(tmp_path)
    client = _client(deps)
    run_id = _generate(client, shot_count=4, key="dedupe-key").json()["run_id"]
    photo_events = [
        event for event in deps.events.all_events(run_id) if event.kind == "photo"
    ]
    assert len(photo_events) == 4
    assert len({event.data["photo_id"] for event in photo_events}) == 4

    # A replay must not append anything.
    before = len(deps.events.all_events(run_id))
    _generate(client, shot_count=4, key="dedupe-key")
    assert len(deps.events.all_events(run_id)) == before
