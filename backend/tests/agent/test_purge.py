"""Step 20 gates: the safety review console and the account purge (SR-1 → SR-7).

steps.md names one done-when:

> deleting an account removes media, analyses, photos, runs, and ledger rows,
> and a flagged photo is unreachable via any public route.

Both halves are proven here, and both are proven *against the real routes* rather
than against the helpers, because a purge that only clears the objects in memory
has not been tested — what matters is that the bytes are gone and every public
route says 404.

Everything runs offline: fixture providers, in-memory stores, tmp blob dir.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pigamagic_agent.agent.analyze import InMemoryAnalysisCache, ReferenceAnalysis
from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.events import EventRecorder, InMemoryEventLog
from pigamagic_agent.agent.ledger import CreditLedger
from pigamagic_agent.agent.lineage import InMemoryPhotoStore, LineageError
from pigamagic_agent.agent.pipeline import load_brief
from pigamagic_agent.agent.providers.image_fixture import FixtureImageProvider
from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider
from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider
from pigamagic_agent.agent.purge import (
    FlagEntry,
    FlagRegistry,
    PurgeError,
    purge_account,
    review_card,
    review_queue,
)
from pigamagic_agent.agent.run import InMemoryRunStore, start_run
from pigamagic_agent.agent.storage import LocalBlobStore
from pigamagic_agent.api.runs import ProviderBundle, build_dependencies, create_app, execute_run

ACCOUNT = "acct_purge"
OTHER = "acct_other"
API = "/api/v1"
TOKEN = "test-admin-token"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _settings(tmp_path, **overrides) -> AgentSettings:
    values = {
        "media_dir": str(tmp_path / "media"),
        "target_shot_count": 4,
        "max_shots": 16,
        "image_budget_usd": 50.0,
        "admin_token": TOKEN,
    }
    values.update(overrides)
    return AgentSettings(**values)


def _deps(
    tmp_path,
    *,
    credits: int = 100,
    admin_token: str = TOKEN,
    analyses: InMemoryAnalysisCache | None = None,
    flags: FlagRegistry | None = None,
):
    settings = _settings(tmp_path, admin_token=admin_token)
    return build_dependencies(
        settings=settings,
        ledger=CreditLedger(balances={ACCOUNT: credits, OTHER: credits}),
        store=InMemoryRunStore(),
        events=InMemoryEventLog(),
        photos=InMemoryPhotoStore(),
        flags=flags if flags is not None else FlagRegistry(),
        analyses=analyses,
        providers=ProviderBundle(
            llm=FixtureLLMProvider(),
            vision=FixtureVisionProvider(),
            web=FixtureWebProvider(),
            image=FixtureImageProvider(),
            blobs=LocalBlobStore(tmp_path / "blobs"),
        ),
    )


async def _run_once(deps, *, account_id: str = ACCOUNT, shoot_id: str = "golden-wedding", shots: int = 3):
    """Run the pipeline for real and publish events, exactly as the API does."""
    brief = load_brief("wedding")
    payload = {
        "shoot_id": shoot_id,
        "title": brief.title,
        "event": brief.event,
        "venue_name": brief.venue_name,
        "activities": list(brief.activities),
        "story": brief.story,
        "style_preset": brief.style_preset,
        "shot_count": shots,
    }
    from pigamagic_agent.agent.compile import ShootBrief

    active = ShootBrief(**payload)
    return await execute_run(
        deps=deps,
        brief=active,
        account_id=account_id,
        shot_count=shots,
        idempotency_key=f"key-{account_id}-{shoot_id}",
    )


def _flag(
    photo_id: str,
    *,
    account_id: str = ACCOUNT,
    state: str = "flagged",
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> FlagEntry:
    return FlagEntry(
        photo_id=photo_id,
        account_id=account_id,
        codes=("sexual_content",),
        reason="asks for intimate content ('nude')",
        run_id="run_1",
        shoot_id="golden-wedding",
        node_id="s00",
        plan_node={"node_id": "s00", "moment": "ceremony", "framing": "wide"},
        prompt="[STYLE] editorial... [ACTION] something",
        negative_prompt="no text, no watermark",
        blob_key="wedding/aa/bb/deadbeef.png",
        state=state,
        created_at=created_at,
    )


# --------------------------------------------------------------------------
# FlagRegistry
# --------------------------------------------------------------------------


def test_registry_blocks_flagged_and_removed_only():
    registry = FlagRegistry()
    registry.flag(_flag("ph_flagged"))
    registry.flag(_flag("ph_cleared", state="cleared"))
    registry.flag(_flag("ph_removed", state="removed"))

    assert registry.blocked("ph_flagged") is True
    assert registry.blocked("ph_removed") is True
    assert registry.blocked("ph_cleared") is False
    # A photo that never tripped a rule is not blocked.
    assert registry.blocked("ph_unknown") is False
    assert registry.state("ph_unknown") == "unflagged"


def test_flag_without_a_code_is_rejected():
    """A flag with no reason cannot be reviewed, so it is not accepted."""
    registry = FlagRegistry()
    with pytest.raises(PurgeError, match="no policy code"):
        registry.flag(FlagEntry(photo_id="ph_1", account_id=ACCOUNT))


def test_removed_photo_cannot_be_reinstated():
    """Removal is terminal: a late duplicate flag must not resurrect a photo."""
    registry = FlagRegistry()
    registry.flag(_flag("ph_1"))
    registry.remove("ph_1", reviewer="ops")

    assert registry.state("ph_1") == "removed"
    with pytest.raises(PurgeError, match="cannot be reinstated"):
        registry.clear("ph_1", reviewer="ops")

    # A duplicate flag arriving afterwards keeps the removed state.
    registry.flag(_flag("ph_1"))
    assert registry.state("ph_1") == "removed"


def test_clear_makes_a_photo_public_again():
    registry = FlagRegistry()
    registry.flag(_flag("ph_1"))
    cleared = registry.clear("ph_1", reviewer="ops", note="no visible nudity")
    assert cleared.state == "cleared"
    assert cleared.public is True
    assert registry.blocked("ph_1") is False
    assert "reviewed by ops" in cleared.reason


def test_review_queue_is_stable_and_oldest_first():
    registry = FlagRegistry()
    registry.flag(_flag("ph_b", created_at="2026-01-03T00:00:00+00:00"))
    registry.flag(_flag("ph_a", created_at="2026-01-01T00:00:00+00:00"))
    # Same instant as `ph_a`: the photo_id tie-break keeps the order stable so a
    # paginated console never shows the same row twice or skips one.
    registry.flag(_flag("ph_c", created_at="2026-01-01T00:00:00+00:00"))

    queue = review_queue(registry)
    assert [item["photo_id"] for item in queue["items"]] == ["ph_a", "ph_c", "ph_b"]
    assert queue["count"] == 3
    # Re-reading the queue must not reshuffle it.
    assert [item["photo_id"] for item in review_queue(registry)["items"]] == [
        "ph_a",
        "ph_c",
        "ph_b",
    ]


def test_review_card_carries_the_plan_and_the_prompt():
    """The console's whole point: a reviewer sees what was asked for."""
    card = review_card(_flag("ph_1"))
    assert card["plan_node"]["moment"] == "ceremony"
    assert card["prompt"].startswith("[STYLE]")
    assert card["negative_prompt"]
    assert card["actions"] == ["clear", "remove"]
    assert card["public"] is False


def test_review_card_offers_no_actions_once_reviewed():
    card = review_card(_flag("ph_1", state="cleared"))
    assert card["actions"] == []


# --------------------------------------------------------------------------
# GATE — the purge removes everything
# --------------------------------------------------------------------------


async def test_gate_account_purge_removes_every_artefact(tmp_path):
    """The steps.md gate: media, analyses, photos, runs, ledger rows all gone."""
    analyses = InMemoryAnalysisCache()
    deps = _deps(tmp_path, analyses=analyses)

    # --- seed real state ---------------------------------------------------
    result = await _run_once(deps)
    assert result.run.status == "done", result.run.error

    photo_ids = list(result.run.photo_ids)
    assert photo_ids, "the fixture run must produce photos to purge"

    blob_keys = []
    for photo_id in photo_ids:
        record = deps.photo_store.get(photo_id)
        assert record is not None
        blob_keys.append(record["blob"]["key"])
    for key in blob_keys:
        assert await deps.providers.blobs.exists(key) is True

    analyses.put(
        "deadbeef",
        ReferenceAnalysis(media_id="med_1", scene_quality="usable"),
        account_id=ACCOUNT,
    )
    assert len(analyses) == 1

    before_entries = len(deps.ledger.entries(account_id=ACCOUNT))
    assert before_entries > 0
    assert deps.ledger.balance(ACCOUNT) > 0 or before_entries > 0

    # An unrelated account must survive untouched.
    other = await _run_once(deps, account_id=OTHER, shoot_id="other-shoot", shots=2)
    other_photo_ids = list(other.run.photo_ids)

    # --- purge -------------------------------------------------------------
    report = await deps.purge(ACCOUNT)

    assert report.media_keys, "the report must name the media it removed"
    assert report.photos == len(photo_ids)
    assert report.blobs == len(set(blob_keys))
    assert report.analyses == 1
    assert report.runs >= 1
    assert report.events > 0
    assert report.ledger_entries == before_entries

    # --- nothing of the account survives -----------------------------------
    for key in blob_keys:
        assert await deps.providers.blobs.exists(key) is False, f"blob {key} outlived the purge"
    for photo_id in photo_ids:
        assert deps.photo_store.get(photo_id) is None
    assert deps.store.get(result.run.run_id) is None
    assert deps.events.has_run(result.run.run_id) is False
    assert deps.ledger.entries(account_id=ACCOUNT) == []
    assert deps.ledger.balance(ACCOUNT) == 0
    assert len(analyses) == 0

    # --- the other account is intact ---------------------------------------
    assert deps.ledger.balance(OTHER) > 0
    for photo_id in other_photo_ids:
        assert deps.photo_store.get(photo_id) is not None


async def test_purge_retains_a_blob_another_account_still_uses(tmp_path):
    """Content addressing means two accounts can share one blob. Do not break it.

    A blunt delete-per-account would empty a file the other account's feed still
    displays, which is a data-loss bug wearing a compliance hat.
    """
    deps = _deps(tmp_path)
    mine = await _run_once(deps, account_id=ACCOUNT, shoot_id="golden-wedding", shots=2)

    shared_key = deps.photo_store.get(mine.run.photo_ids[0])["blob"]["key"]
    # A second account's photo pointing at the same bytes, exactly as the
    # content-addressed store produces for a duplicated frame.
    deps.photo_store.put(
        {
            "photo_id": "ph_other_shared",
            "account_id": OTHER,
            "prompt_hash": "other-hash",
            "blob": {"key": shared_key},
        }
    )

    report = await deps.purge(ACCOUNT)

    assert shared_key in report.media_keys
    assert report.blobs_retained == 1, "a shared blob must be retained, not deleted"
    assert await deps.providers.blobs.exists(shared_key) is True
    # The other account's photo row itself is untouched.
    assert deps.photo_store.get("ph_other_shared") is not None


async def test_purging_an_unknown_account_is_a_no_op_not_an_error(tmp_path):
    """Deleting an account twice must be safe: retries are how deletes are done."""
    deps = _deps(tmp_path)
    report = await deps.purge("acct_never_existed")
    assert report.empty is True
    assert report.blobs == 0


async def test_purge_requires_a_store_that_can_actually_delete(tmp_path):
    """A purge that silently does nothing is the failure this guards against."""
    deps = _deps(tmp_path)
    await _run_once(deps)

    class _NoDelete:
        def put(self, run): ...
        def get(self, run_id): return None
        def active_for_shoot(self, shoot_id): return None
        def by_idempotency_key(self, key): return None
        def link_idempotency(self, key, run_id): ...
        def list_runs(self): return []

    with pytest.raises(PurgeError, match="cannot delete"):
        await purge_account(
            ACCOUNT,
            ledger=deps.ledger,
            runs=_NoDelete(),
            events=deps.events,
            photos=deps.photo_store,
            blobs=deps.providers.blobs,
        )


def test_ledger_purge_drops_the_balance_and_the_reservation(tmp_path):
    """A surviving hold would refuse a future run for a deleted account."""
    ledger = CreditLedger(balances={ACCOUNT: 10})
    ledger.reserve(ACCOUNT, 3, run_id="run_x")

    counts = ledger.purge_account(ACCOUNT)

    assert counts["entries"] >= 1
    assert counts["reservations"] == 1
    assert ledger.balance(ACCOUNT) == 0
    assert ledger.entries(account_id=ACCOUNT) == []
    assert ledger.reservation_for_run("run_x") is None


def test_analysis_cache_purge_keeps_a_key_another_account_still_owns():
    """Derived data is shared by content hash, so it is dropped only when unowned."""
    cache = InMemoryAnalysisCache()
    analysis = ReferenceAnalysis(media_id="med_1", scene_quality="usable")
    cache.put("shared", analysis, account_id=ACCOUNT)
    cache.put("shared", analysis, account_id=OTHER)
    cache.put("mine", analysis, account_id=ACCOUNT)

    removed = cache.purge_account(ACCOUNT)

    assert removed == 1
    assert cache.get("mine") is None
    assert cache.get("shared") is not None, "the other account still owns these bytes"


# --------------------------------------------------------------------------
# GATE — a flagged photo is unreachable on every public route
# --------------------------------------------------------------------------


async def test_gate_flagged_photo_is_unreachable_on_every_public_route(tmp_path):
    """SR-4, asserted against the real HTTP routes rather than the helper."""
    deps = _deps(tmp_path)
    result = await _run_once(deps, shots=3)
    photo_id = result.run.photo_ids[0]

    client = TestClient(create_app(deps))

    # Before flagging: the photo is reachable on the routes that serve it.
    assert client.get(f"{API}/photos/{photo_id}/lineage").status_code == 200
    feed_before = client.get(f"{API}/feed").json()
    assert any(card["id"] == photo_id for card in feed_before["items"])

    deps.flag_registry.flag(
        FlagEntry(
            photo_id=photo_id,
            account_id=ACCOUNT,
            codes=("sexual_content",),
            reason="flagged by the safety gate",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )

    # After flagging: every public route behaves as if it does not exist.
    lineage = client.get(f"{API}/photos/{photo_id}/lineage")
    assert lineage.status_code == 404
    assert lineage.json()["error"]["code"] == "photo_not_found"

    compare = client.get(f"{API}/photos/{photo_id}/compare")
    assert compare.status_code == 404

    feed_after = client.get(f"{API}/feed").json()
    assert not any(card["id"] == photo_id for card in feed_after["items"])
    # The run's other photos are unaffected.
    assert len(feed_after["items"]) == len(feed_before["items"]) - 1

    run_body = client.get(f"{API}/runs/{result.run.run_id}").json()
    assert not any(card["id"] == photo_id for card in run_body["cards"])

    # Reaching it through the stream must not work either.
    events = client.get(f"{API}/runs/{result.run.run_id}/events")
    assert photo_id not in _photo_ids_in(events.text)


def _photo_ids_in(body: str) -> set[str]:
    found: set[str] = set()
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        card = payload.get("card") or {}
        if card.get("id"):
            found.add(card["id"])
        if payload.get("photo_id"):
            found.add(payload["photo_id"])
    return found


def test_clearing_a_flag_restores_public_access(tmp_path):
    """Clearing must actually clear: a photo stuck hidden after review is a bug."""
    deps = _deps(tmp_path)
    deps.photo_store.put(
        {"photo_id": "ph_manual", "account_id": ACCOUNT, "prompt_hash": "h1", "blob": {"key": "k"}}
    )
    deps.flag_registry.flag(_flag("ph_manual"))

    client = TestClient(create_app(deps))
    assert client.get(f"{API}/photos/ph_manual/lineage").status_code == 404

    response = client.post(
        f"{API}/admin/flags/ph_manual",
        headers={"X-Admin-Token": TOKEN},
        json={"action": "clear", "reviewer": "ops"},
    )
    assert response.status_code == 200
    assert response.json()["state"] == "cleared"

    assert client.get(f"{API}/photos/ph_manual/lineage").status_code == 200


# --------------------------------------------------------------------------
# Admin routes: authentication and fail-closed behaviour
# --------------------------------------------------------------------------


def test_admin_routes_are_disabled_without_a_configured_token(tmp_path):
    """An unset secret must never mean 'open to everyone'."""
    deps = _deps(tmp_path, admin_token="")
    client = TestClient(create_app(deps))

    response = client.get(f"{API}/admin/flags")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "admin_disabled"


def test_admin_routes_reject_a_missing_or_wrong_token(tmp_path):
    deps = _deps(tmp_path)
    deps.flag_registry.flag(_flag("ph_1"))
    client = TestClient(create_app(deps))

    assert client.get(f"{API}/admin/flags").status_code == 401
    assert (
        client.get(f"{API}/admin/flags", headers={"X-Admin-Token": "wrong"}).status_code == 401
    )
    # A wrong token must not leak the queue body.
    body = client.get(f"{API}/admin/flags", headers={"X-Admin-Token": "wrong"}).json()
    assert "items" not in body


def test_build_dependencies_keeps_the_caller_supplied_flag_registry(tmp_path):
    """Regression: `FlagRegistry` defines `__len__`, so an *empty* registry is
    falsy. `build_dependencies` used `flags or FlagRegistry()`, which silently
    threw the caller's registry away — any flag written to the caller's object
    was never seen by the app, so flagged photos stayed public (SR-4)."""
    settings = _settings(tmp_path)
    supplied = FlagRegistry()
    assert not supplied, "an empty registry must be falsy for this test to mean anything"

    deps = build_dependencies(
        settings=settings,
        flags=supplied,
        providers=ProviderBundle(
            llm=FixtureLLMProvider(),
            vision=FixtureVisionProvider(),
            web=FixtureWebProvider(),
            image=FixtureImageProvider(),
            blobs=LocalBlobStore(tmp_path / "blobs"),
        ),
    )
    assert deps.flag_registry is supplied

    # A flag written through the caller's handle must actually block the route.
    supplied.flag(_flag("ph_1"))
    client = TestClient(create_app(deps))
    assert client.get(f"{API}/photos/ph_1/lineage").status_code == 404


def test_admin_queue_lists_flags_with_plan_and_prompt(tmp_path):
    deps = _deps(tmp_path)
    deps.flag_registry.flag(_flag("ph_1"))
    client = TestClient(create_app(deps))

    body = client.get(f"{API}/admin/flags", headers={"X-Admin-Token": TOKEN}).json()
    assert body["count"] == 1
    item = body["items"][0]
    assert item["photo_id"] == "ph_1"
    assert item["plan_node"]["moment"] == "ceremony"
    assert "[STYLE]" in item["prompt"]
    assert item["public"] is False


def test_admin_review_rejects_an_unknown_action(tmp_path):
    deps = _deps(tmp_path)
    deps.flag_registry.flag(_flag("ph_1"))
    client = TestClient(create_app(deps))

    response = client.post(
        f"{API}/admin/flags/ph_1",
        headers={"X-Admin-Token": TOKEN},
        json={"action": "publish-everywhere"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_action"


def test_admin_review_of_an_unknown_photo_is_a_404(tmp_path):
    deps = _deps(tmp_path)
    client = TestClient(create_app(deps))

    assert (
        client.get(f"{API}/admin/flags/ph_nope", headers={"X-Admin-Token": TOKEN}).status_code
        == 404
    )


def test_admin_purge_route_returns_a_verifiable_report(tmp_path):
    deps = _deps(tmp_path)
    client = TestClient(create_app(deps))

    response = client.post(
        f"{API}/admin/purge/{ACCOUNT}", headers={"X-Admin-Token": TOKEN}
    )
    assert response.status_code == 200
    body = response.json()
    # Counts, not a bare `{"ok": true}` — an unverifiable delete is not a delete.
    for field in ("runs", "events", "photos", "analyses", "blobs", "ledger_entries"):
        assert field in body
    assert body["account_id"] == ACCOUNT


# --------------------------------------------------------------------------
# Store-level deletions the purge depends on
# --------------------------------------------------------------------------


def test_run_store_delete_clears_the_idempotency_index():
    """A stale key would let a replay resurrect a purged run."""
    store = InMemoryRunStore()
    deps_run = _FakeRun(run_id="run_1", shoot_id="shoot_1", key="idem-1")
    store.put(deps_run)
    store.link_idempotency("idem-1", "run_1")
    store.track_shoot(deps_run)

    assert store.delete("run_1") is True
    assert store.get("run_1") is None
    assert store.by_idempotency_key("idem-1") is None
    assert store.active_for_shoot("shoot_1") is None
    assert store.delete("run_1") is False


def test_photo_store_delete_also_clears_the_parent_index():
    store = InMemoryPhotoStore()
    store.put({"photo_id": "ph_1", "prompt_hash": "h"})
    store.put({"photo_id": "ph_2", "prompt_hash": "h", "parent_id": "ph_1"})

    assert store.children_of("ph_1")[0]["photo_id"] == "ph_2"
    assert store.delete("ph_2") is True
    assert store.children_of("ph_1") == []
    assert store.get("ph_2") is None


def test_event_log_delete_removes_the_whole_stream():
    log = InMemoryEventLog()
    log.append("run_1", "run", {"run_id": "run_1"})
    log.append("run_1", "run_done", {"run_id": "run_1"})

    assert log.delete("run_1") is True
    assert log.has_run("run_1") is False
    assert log.events_after("run_1", 0) == []
    assert log.delete("run_1") is False


class _FakeRun:
    """Minimal run object for store-level tests."""

    def __init__(self, *, run_id: str, shoot_id: str, key: str):
        self.run_id = run_id
        self.shoot_id = shoot_id
        self.account_id = ACCOUNT
        self.status = "done"
        self.idempotency_key = key
        self.created_at = "2026-01-01T00:00:00+00:00"

    @property
    def active(self) -> bool:
        return False

    @property
    def terminal(self) -> bool:
        return True
