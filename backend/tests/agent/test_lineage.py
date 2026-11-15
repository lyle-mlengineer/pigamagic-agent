"""Step 18 gates: lineage, compare, and per-edit credits.

steps.md names two assertions:

* **a photo has ≥2 siblings retrievable**
* **each edit appears as its own ledger line**

The invariants behind them (design.md §11.2) are that a delivered photo is never
mutated, and that lineage refuses to invent a parent rather than drawing a
plausible-looking tree.
"""

from __future__ import annotations

import pytest

from pigamagic_agent.agent.comments import Comment, compile_edit
from pigamagic_agent.agent.ledger import CREDITS_PER_PHOTO, CreditLedger
from pigamagic_agent.agent.lineage import (
    COMPARE_FIELDS,
    InMemoryPhotoStore,
    LineageError,
    compare,
    compare_with_parent,
    delete_take,
    edit_ledger_lines,
    edit_run_id,
    lineage_tree,
    reserve_edit,
    root_of,
    settle_edit,
    siblings_of,
)
from pigamagic_agent.agent.plan import ShotSpec

ACCOUNT = "acct_lineage"


def _shot(node_id="s03") -> ShotSpec:
    return ShotSpec(
        node_id=node_id,
        phase="ceremony",
        moment="vow exchange",
        activity="vows",
        description="The couple exchange vows.",
        subjects=["p1", "p2"],
        framing="wide",
        camera={"focal_length_mm": 85},
        lighting="soft window light",
        venue_anchors=["stone arches"],
        wardrobe_notes=["ivory gown"],
        must_include=["the rings"],
        must_avoid=["flash"],
        style_preset="editorial",
        caption="The vows",
        why="the defining moment",
    )


def _comment(body: str, comment_id: str) -> Comment:
    return Comment(
        comment_id=comment_id, photo_id="ph_1", account_id=ACCOUNT, body=body, mode="edit"
    )


def _instruction(body: str, comment_id: str):
    return compile_edit(_comment(body, comment_id), shot=_shot())


def _photo(photo_id, *, parent_id=None, prompt_hash=None, framing="wide", seed=11, created="2026-01-01T00:00:00+00:00"):
    return {
        "photo_id": photo_id,
        "node_id": "s03",
        "prompt_hash": prompt_hash or f"hash_{photo_id}",
        "framing": framing,
        "seed": seed,
        "created_at": created,
        "parent_id": parent_id,
        "caption": "The vows",
    }


def _store_with_family() -> InMemoryPhotoStore:
    """One original and two edits of it — the ≥2 siblings case."""
    store = InMemoryPhotoStore()
    store.put(_photo("ph_1", created="2026-01-01T00:00:00+00:00"))
    store.put(
        _photo("ph_2", parent_id="ph_1", framing="close", seed=12, created="2026-01-01T00:00:01+00:00")
    )
    store.put(
        _photo("ph_3", parent_id="ph_1", framing="medium", seed=13, created="2026-01-01T00:00:02+00:00")
    )
    return store


# --------------------------------------------------------------------------
# GATE 1: a photo has >= 2 siblings retrievable
# --------------------------------------------------------------------------


def test_gate_two_or_more_siblings_are_retrievable():
    store = _store_with_family()
    siblings = siblings_of(store, "ph_2")
    ids = [item["photo_id"] for item in siblings]

    assert len(siblings) >= 2, "FR-7.5 needs at least two takes to compare"
    assert ids == ["ph_1", "ph_2", "ph_3"]
    # Asking from any member of the family returns the same family.
    assert [p["photo_id"] for p in siblings_of(store, "ph_1")] == ids
    assert [p["photo_id"] for p in siblings_of(store, "ph_3")] == ids


def test_sibling_order_is_stable_across_calls():
    store = _store_with_family()
    first = [p["photo_id"] for p in siblings_of(store, "ph_1")]
    second = [p["photo_id"] for p in siblings_of(store, "ph_1")]
    # A compare view must not reshuffle between two identical requests.
    assert first == second


def test_lineage_tree_reports_root_depth_and_siblings():
    store = _store_with_family()
    tree = lineage_tree(store, "ph_3")
    assert tree["root_id"] == "ph_1"
    assert tree["sibling_count"] == 3
    assert tree["depth"] == 1
    assert tree["tree"]["photo_id"] == "ph_1"
    assert [c["photo_id"] for c in tree["tree"]["children"]] == ["ph_2", "ph_3"]


def test_nested_lineage_is_a_tree_not_a_list():
    store = _store_with_family()
    # An edit of an edit.
    store.put(
        _photo("ph_4", parent_id="ph_2", framing="close", seed=14, created="2026-01-01T00:00:03+00:00")
    )
    tree = lineage_tree(store, "ph_4")
    assert tree["root_id"] == "ph_1"
    assert tree["depth"] == 2
    grandchild = tree["tree"]["children"][0]["children"][0]
    assert grandchild["photo_id"] == "ph_4"


def test_root_of_walks_to_the_original():
    store = _store_with_family()
    store.put(_photo("ph_4", parent_id="ph_2", created="2026-01-01T00:00:03+00:00"))
    assert root_of(store, "ph_4")["photo_id"] == "ph_1"
    assert root_of(store, "ph_1")["photo_id"] == "ph_1"


def test_missing_parent_raises_rather_than_pretending_it_is_a_root():
    """A dangling parent is a data problem, not an original photo."""
    store = InMemoryPhotoStore()
    store.put(_photo("ph_9", parent_id="ph_missing"))
    with pytest.raises(LineageError, match="missing parent"):
        root_of(store, "ph_9")


def test_a_cycle_is_detected_instead_of_hanging():
    store = InMemoryPhotoStore()
    store.put(_photo("ph_a", parent_id="ph_b", prompt_hash="h1"))
    store.put(_photo("ph_b", parent_id="ph_a", prompt_hash="h2"))
    with pytest.raises(LineageError, match="cyclic"):
        root_of(store, "ph_a")


def test_unknown_photo_raises():
    with pytest.raises(LineageError, match="no photo"):
        siblings_of(InMemoryPhotoStore(), "ph_nope")


# --------------------------------------------------------------------------
# Delivered photos are immutable
# --------------------------------------------------------------------------


def test_put_is_idempotent_for_identical_content():
    store = InMemoryPhotoStore()
    store.put(_photo("ph_1"))
    store.put(_photo("ph_1"))  # same prompt_hash
    assert len(store.all_photos()) == 1


def test_put_refuses_to_overwrite_a_delivered_photo():
    """§11.2: never mutate a delivered photo. A conflicting put is loud."""
    store = InMemoryPhotoStore()
    store.put(_photo("ph_1", prompt_hash="original"))
    with pytest.raises(LineageError, match="immutable"):
        store.put(_photo("ph_1", prompt_hash="different"))
    assert store.get("ph_1")["prompt_hash"] == "original"


def test_delete_take_refuses_to_orphan_children():
    store = _store_with_family()
    result = delete_take(store, "ph_1")
    assert result.deleted == []
    assert result.refused == ["ph_1"]
    assert "edit(s)" in result.reason
    # The original still exists, so the family still has a root.
    assert store.get("ph_1") is not None


def test_delete_take_removes_a_leaf():
    store = _store_with_family()
    result = delete_take(store, "ph_3")
    assert result.deleted == ["ph_3"]
    assert store.get("ph_3") is None
    assert [p["photo_id"] for p in siblings_of(store, "ph_1")] == ["ph_1", "ph_2"]


def test_delete_take_of_unknown_photo_is_refused_not_raised():
    result = delete_take(InMemoryPhotoStore(), "ph_nope")
    assert result.deleted == []
    assert "no photo" in result.reason


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------


def test_compare_reports_only_what_changed():
    store = _store_with_family()
    payload = compare_with_parent(store, "ph_2")
    assert payload["parent_id"] == "ph_1"
    assert payload["child_id"] == "ph_2"
    # framing, seed and prompt_hash differ; moment/activity/caption do not.
    assert set(payload["changed"]) == {"framing", "seed", "prompt_hash"}
    assert payload["changed"]["framing"] == {"from": "wide", "to": "close"}
    assert payload["parent_unchanged"] is True
    # Every reported field is one we actually compare.
    assert set(payload["changed"]) <= set(COMPARE_FIELDS)


def test_compare_states_the_instruction_and_preserves():
    store = _store_with_family()
    instruction = _instruction("make it brighter", "c1")
    store.put(
        {
            **_photo("ph_5", parent_id="ph_1", framing="close", created="2026-01-01T00:00:05+00:00"),
            "edit_instruction": instruction.model_dump(),
            "preserves": list(instruction.preserves),
        }
    )
    payload = compare_with_parent(store, "ph_5")
    assert payload["instruction"]["change"] == "lighting"
    assert "identity" in payload["preserves"]
    assert "venue" in payload["preserves"]


def test_compare_refuses_a_non_parent():
    store = _store_with_family()
    with pytest.raises(LineageError, match="not its parent"):
        compare(store.get("ph_1"), store.get("ph_1"))


def test_compare_of_an_original_raises():
    store = _store_with_family()
    with pytest.raises(LineageError, match="is an original"):
        compare_with_parent(store, "ph_1")


# --------------------------------------------------------------------------
# GATE 2: each edit appears as its own ledger line
# --------------------------------------------------------------------------


def test_gate_each_edit_has_its_own_ledger_lines():
    ledger = CreditLedger(balances={ACCOUNT: 20})
    brighter = _instruction("make it brighter", "c1")
    closer = _instruction("move closer", "c2")

    first = reserve_edit(ledger, account_id=ACCOUNT, instruction=brighter)
    first = settle_edit(ledger, first, delivered=True, node_id="s03")
    second = reserve_edit(ledger, account_id=ACCOUNT, instruction=closer)
    second = settle_edit(ledger, second, delivered=True, node_id="s03")

    # Different edits are different runs, so each has its own line.
    assert edit_run_id(brighter) != edit_run_id(closer)
    first_lines = edit_ledger_lines(ledger, account_id=ACCOUNT, instruction=brighter)
    second_lines = edit_ledger_lines(ledger, account_id=ACCOUNT, instruction=closer)

    assert first_lines and second_lines
    assert {line["entry_id"] for line in first_lines}.isdisjoint(
        {line["entry_id"] for line in second_lines}
    )
    # Each edit reserved exactly one photo's worth.
    assert [line["delta"] for line in first_lines if line["kind"] == "reservation"] == [
        -CREDITS_PER_PHOTO
    ]
    assert first.charged == CREDITS_PER_PHOTO
    assert second.charged == CREDITS_PER_PHOTO
    assert ledger.balance(ACCOUNT) == 20 - 2 * CREDITS_PER_PHOTO


def test_a_failed_edit_is_refunded_and_costs_nothing():
    ledger = CreditLedger(balances={ACCOUNT: 5})
    instruction = _instruction("make it brighter", "c1")
    charge = reserve_edit(ledger, account_id=ACCOUNT, instruction=instruction)
    assert ledger.balance(ACCOUNT) == 5 - CREDITS_PER_PHOTO

    charge = settle_edit(ledger, charge, delivered=False, node_id="s03")
    assert charge.charged == 0
    assert charge.refunded == CREDITS_PER_PHOTO
    assert ledger.balance(ACCOUNT) == 5
    refunds = [
        line
        for line in edit_ledger_lines(ledger, account_id=ACCOUNT, instruction=instruction)
        if line["kind"] == "refund"
    ]
    assert len(refunds) == 1


def test_retrying_the_same_edit_does_not_charge_twice():
    """A retried POST after a network error must reconcile, not re-charge."""
    ledger = CreditLedger(balances={ACCOUNT: 10})
    instruction = _instruction("make it brighter", "c1")

    first = reserve_edit(ledger, account_id=ACCOUNT, instruction=instruction)
    settle_edit(ledger, first, delivered=True, node_id="s03")
    balance_after = ledger.balance(ACCOUNT)

    again = reserve_edit(ledger, account_id=ACCOUNT, instruction=instruction)
    assert again.reservation.reservation_id == first.reservation.reservation_id
    assert ledger.balance(ACCOUNT) == balance_after, "a retry must not hold credits again"

    reservations = [
        line
        for line in edit_ledger_lines(ledger, account_id=ACCOUNT, instruction=instruction)
        if line["kind"] == "reservation"
    ]
    assert len(reservations) == 1


def test_an_edit_cannot_overspend():
    ledger = CreditLedger(balances={ACCOUNT: 0})
    from pigamagic_agent.agent.ledger import InsufficientCredits

    with pytest.raises(InsufficientCredits):
        reserve_edit(
            ledger,
            account_id=ACCOUNT,
            instruction=_instruction("make it brighter", "c1"),
        )
    assert ledger.balance(ACCOUNT) == 0


def test_settling_without_a_reservation_raises():
    from pigamagic_agent.agent.lineage import EditCharge

    charge = EditCharge(run_id="edit_x", instruction_key="k", reserved=0, charged=0, refunded=0)
    with pytest.raises(LineageError, match="never reserved"):
        settle_edit(lambda *a, **k: None, charge, delivered=True)


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------


def _client(tmp_path):
    from fastapi.testclient import TestClient

    from pigamagic_agent.agent.config import AgentSettings
    from pigamagic_agent.agent.events import InMemoryEventLog
    from pigamagic_agent.agent.providers.image_fixture import FixtureImageProvider
    from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider
    from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
    from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider
    from pigamagic_agent.agent.run import InMemoryRunStore
    from pigamagic_agent.agent.storage import LocalBlobStore
    from pigamagic_agent.api.runs import ProviderBundle, build_dependencies, create_app

    deps = build_dependencies(
        settings=AgentSettings(
            media_dir=str(tmp_path / "media"),
            target_shot_count=3,
            max_shots=16,
            image_budget_usd=50.0,
        ),
        ledger=CreditLedger(balances={ACCOUNT: 100}),
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
    return TestClient(create_app(deps)), deps


def test_rendered_photos_are_registered_for_lineage(tmp_path):
    client, deps = _client(tmp_path)
    body = client.post(
        "/api/v1/shoots/golden-wedding/generate",
        json={"account_id": ACCOUNT, "brief": "wedding", "shot_count": 3},
    ).json()
    assert len(deps.photo_store.all_photos()) == len(body["photo_ids"])

    photo_id = body["photo_ids"][0]
    response = client.get(f"/api/v1/photos/{photo_id}/lineage")
    assert response.status_code == 200
    assert response.json()["root_id"] == photo_id


def test_lineage_endpoint_returns_two_siblings(tmp_path):
    client, deps = _client(tmp_path)
    body = client.post(
        "/api/v1/shoots/golden-wedding/generate",
        json={"account_id": ACCOUNT, "brief": "wedding", "shot_count": 3},
    ).json()
    root_id = body["photo_ids"][0]

    # Two edits of the same original, as the feedback loop produces.
    for index, framing in enumerate(("close", "medium"), start=1):
        deps.photo_store.put(
            _photo(
                f"ph_edit{index}",
                parent_id=root_id,
                framing=framing,
                created=f"2026-02-0{index}T00:00:00+00:00",
            )
        )

    payload = client.get(f"/api/v1/photos/{root_id}/lineage").json()
    assert payload["sibling_count"] == 3
    assert len(payload["siblings"]) >= 2


def test_lineage_endpoint_404s_for_an_unknown_photo(tmp_path):
    client, _ = _client(tmp_path)
    response = client.get("/api/v1/photos/ph_missing/lineage")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "photo_not_found"


def test_compare_endpoint_404s_and_422s(tmp_path):
    client, deps = _client(tmp_path)
    assert client.get("/api/v1/photos/ph_missing/compare").status_code == 404

    deps.photo_store.put(_photo("ph_root"))
    response = client.get("/api/v1/photos/ph_root/compare")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "compare_unavailable"


def test_compare_of_a_realistic_edit_reports_only_its_own_changes(tmp_path):
    """A realistic child differs from its parent in framing/seed/hash only.

    A compare view that also reports `moment`, `activity`, `provider` or `model`
    as changed is noise — those do not vary between takes of one plan node.
    """
    from fastapi.testclient import TestClient

    client, deps = _client(tmp_path)
    body = client.post(
        "/api/v1/shoots/golden-wedding/generate",
        json={"account_id": ACCOUNT, "brief": "wedding", "shot_count": 2},
    ).json()
    root_id = body["photo_ids"][0]
    parent = dict(deps.photo_store.get(root_id))
    deps.photo_store.put(
        {
            **parent,
            "photo_id": "ph_real_child",
            "parent_id": root_id,
            "framing": "close",
            "seed": 999,
            "prompt_hash": "newhash",
            "created_at": "2026-03-01T00:00:00+00:00",
        }
    )

    payload = client.get("/api/v1/photos/ph_real_child/compare").json()
    assert set(payload["changed"]) == {"framing", "seed", "prompt_hash"}


def test_compare_endpoint_returns_the_delta(tmp_path):
    client, deps = _client(tmp_path)
    deps.photo_store.put(_photo("ph_root", framing="wide", seed=11))
    deps.photo_store.put(
        _photo("ph_child", parent_id="ph_root", framing="close", seed=99, created="2026-01-02T00:00:00+00:00")
    )
    payload = client.get("/api/v1/photos/ph_child/compare").json()
    assert payload["parent_id"] == "ph_root"
    assert payload["changed"]["framing"] == {"from": "wide", "to": "close"}
