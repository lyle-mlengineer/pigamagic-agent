"""Step 11 gates: run state machine, reserve-then-settle, idempotency.

The three assertions steps.md names explicitly, each written as its own test:

* a 12-node run with 1 forced failure ends `partial`, charges 11, one refund line
* an insufficient-credit request is rejected before any provider call
* re-posting the same idempotency key returns the same run

Everything runs offline through the fixture providers.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from pigamagic_agent.agent.compile import ShootBrief
from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.ledger import (
    CREDITS_PER_PHOTO,
    CreditLedger,
    InsufficientCredits,
    LedgerConflict,
    LedgerError,
)
from pigamagic_agent.agent.pipeline import load_brief
from pigamagic_agent.agent.providers.base import ProviderError
from pigamagic_agent.agent.providers.image_fixture import FixtureImageProvider
from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider
from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider
from pigamagic_agent.agent.run import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    InMemoryRunStore,
    RunInProgress,
    RunStateError,
    GenerationRun,
    run_digest,
    start_run,
    transition,
)
from pigamagic_agent.agent.storage import LocalBlobStore

ACCOUNT = "acct_test"


def _settings(tmp_path, **overrides) -> AgentSettings:
    values = {
        "media_dir": str(tmp_path / "media"),
        "target_shot_count": 8,
        "max_shots": 16,
        "image_budget_usd": 50.0,
    }
    values.update(overrides)
    return AgentSettings(**values)


def _brief(name: str = "wedding") -> ShootBrief:
    return load_brief(name)


WEDDING_ACTIVITIES_12 = (
    "ceremony",
    "first dance",
    "toast",
    "getting ready",
    "group portrait",
    "cake cutting",
    "vows",
    "recessional",
    "dinner",
    "speeches",
    "bouquet toss",
    "departure",
)


def _brief_with_shots(shot_count: int) -> ShootBrief:
    """A brief whose plan actually contains `shot_count` nodes.

    The golden briefs pin their own plan size, and `shot_count` only ever
    *trims* a plan — so asking the wedding golden for 12 silently gives 10. The
    steps.md gate needs a genuine 12-node run, so this brief names twelve
    activities, which the fixture turns into twelve distinct shots.
    """
    return ShootBrief(
        shoot_id="golden-wedding",
        title="Golden wedding",
        event="wedding",
        venue_name="The Grand Pavilion",
        activities=WEDDING_ACTIVITIES_12[:shot_count],
        shot_count=shot_count,
    )


def _harness(tmp_path, *, credits: int = 100, llm=None, image=None):
    settings = _settings(tmp_path)
    ledger = CreditLedger(balances={ACCOUNT: credits})
    store = InMemoryRunStore()
    return {
        "settings": settings,
        "ledger": ledger,
        "store": store,
        "llm": llm or FixtureLLMProvider(),
        "vision": FixtureVisionProvider(),
        "web": FixtureWebProvider(),
        "image": image or FixtureImageProvider(),
        "blobs": LocalBlobStore(tmp_path / "blobs"),
    }


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------


def test_legal_transition_is_recorded():
    run = GenerationRun(run_id="run_a", shoot_id="s1", account_id=ACCOUNT)
    transition(run, "researching", note="start")
    assert run.status == "researching"
    assert run.history == [{"status": "researching", "at": run.history[0]["at"], "note": "start"}]


def test_illegal_transition_raises_and_names_legal_options():
    run = GenerationRun(run_id="run_a", shoot_id="s1", account_id=ACCOUNT)
    with pytest.raises(RunStateError) as excinfo:
        transition(run, "rendering")
    message = str(excinfo.value)
    # The error must say what *is* possible, so it is actionable.
    assert "researching" in message and "failed" in message


def test_terminal_states_cannot_move():
    for status in TERMINAL_STATUSES:
        assert ALLOWED_TRANSITIONS[status] == frozenset()

    run = GenerationRun(run_id="run_a", shoot_id="s1", account_id=ACCOUNT)
    transition(run, "failed")
    with pytest.raises(RunStateError):
        transition(run, "done")


def test_same_status_transition_is_a_noop():
    run = GenerationRun(run_id="run_a", shoot_id="s1", account_id=ACCOUNT)
    before = len(run.history)
    transition(run, "queued")
    assert len(run.history) == before


def test_terminal_transition_stamps_finished_at():
    run = GenerationRun(run_id="run_a", shoot_id="s1", account_id=ACCOUNT)
    assert run.finished_at == ""
    transition(run, "failed")
    assert run.finished_at != ""
    assert run.terminal


# --------------------------------------------------------------------------
# The 12-node / 1-failure gate from steps.md
# --------------------------------------------------------------------------


class _OneShotFailsImageProvider(FixtureImageProvider):
    """Fails exactly one call, deterministically.

    `render_plan` walks the plan in node order, so failing by call index pins the
    failure to one shot without relying on the prompt carrying an id it does not
    carry.
    """

    name: ClassVar[str] = "fixture-one-fail"

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at
        self.calls: list[int] = []

    async def generate(self, **kwargs):  # type: ignore[override]
        index = len(self.calls)
        self.calls.append(index)
        if index == self.fail_at:
            raise ProviderError("simulated render failure", retryable=False)
        return await super().generate(**kwargs)


@pytest.mark.asyncio
async def test_twelve_node_run_with_one_failure_ends_partial_and_charges_eleven(tmp_path):
    harness = _harness(tmp_path)
    brief = _brief_with_shots(12)

    # Learn the real node ids from a first run so the forced failure is genuinely
    # one node, not a guess.
    probe = await start_run(brief, account_id=ACCOUNT, shot_count=12, **harness)
    assert probe.outcome is not None
    assert len(probe.outcome.photos) == 12, "the brief must yield 12 nodes"

    failing = _OneShotFailsImageProvider(fail_at=3)
    harness2 = _harness(tmp_path / "second", image=failing)

    result = await start_run(
        brief,
        account_id=ACCOUNT,
        shot_count=12,
        **harness2,
    )

    run = result.run
    assert run.status == "partial", f"expected partial, got {run.status}: {run.error}"
    assert run.requested == 12
    assert len(run.photo_ids) == 11
    assert run.spent_credits == 11
    assert run.refunded_credits == 1
    assert run.reserved_credits == 12

    # Exactly one refund line, naming the failed node.
    refunds = [
        entry
        for entry in harness2["ledger"].entries(account_id=ACCOUNT, run_id=run.run_id)
        if entry.kind == "refund"
    ]
    assert len(refunds) == 1
    assert refunds[0].reason == "refund: render_failed"
    assert refunds[0].delta == 1

    # The money adds up: 12 held, 11 charged, 1 returned.
    assert harness2["ledger"].balance(ACCOUNT) == 100 - 11


@pytest.mark.asyncio
async def test_partial_run_reports_the_shortfall_to_the_user(tmp_path):
    harness = _harness(tmp_path)
    brief = _brief_with_shots(12)
    probe = await start_run(brief, account_id=ACCOUNT, shot_count=12, **harness)
    assert probe.outcome is not None

    failing = _OneShotFailsImageProvider(fail_at=0)
    harness2 = _harness(tmp_path / "b", image=failing)
    result = await start_run(brief, account_id=ACCOUNT, shot_count=12, **harness2)

    # A partial run must never look like a clean one.
    assert result.run.status == "partial"
    assert any("failed" in warning for warning in result.run.warnings)


# --------------------------------------------------------------------------
# Insufficient credits — rejected before any provider call
# --------------------------------------------------------------------------


class _CountingImageProvider(FixtureImageProvider):
    name: ClassVar[str] = "fixture-counting"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def generate(self, **kwargs):  # type: ignore[override]
        self.calls += 1
        return await super().generate(**kwargs)


class _CountingWebProvider(FixtureWebProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def search(self, query, *, limit=5, budget=None):  # type: ignore[override]
        self.calls += 1
        return await super().search(query, limit=limit, budget=budget)


@pytest.mark.asyncio
async def test_insufficient_credits_rejected_before_any_provider_call(tmp_path):
    image = _CountingImageProvider()
    web = _CountingWebProvider()
    harness = _harness(tmp_path, credits=3, image=image)
    harness["web"] = web

    result = await start_run(
        _brief("wedding"),
        account_id=ACCOUNT,
        shot_count=12,
        **harness,
    )

    assert result.run.status == "failed"
    assert "needs 12 credit(s)" in result.run.error
    assert "has 3" in result.run.error

    # The whole point: nothing was spent, nothing was called.
    assert image.calls == 0
    assert web.calls == 0
    assert harness["ledger"].balance(ACCOUNT) == 3
    assert harness["ledger"].entries(account_id=ACCOUNT) == []


@pytest.mark.asyncio
async def test_refusal_costs_nothing(tmp_path):
    image = _CountingImageProvider()
    harness = _harness(tmp_path, image=image)
    brief = ShootBrief(
        shoot_id="shoot_refused",
        title="Nude portrait session",
        event="birthday",
        venue_name="A studio",
        activities=("nude portrait",),
        shot_count=4,
    )

    result = await start_run(brief, account_id=ACCOUNT, shot_count=4, **harness)

    assert result.run.status == "failed"
    assert "nude" in result.run.error
    assert image.calls == 0
    assert harness["ledger"].balance(ACCOUNT) == 100
    assert harness["ledger"].entries(account_id=ACCOUNT) == []


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_idempotency_key_returns_the_same_run(tmp_path):
    harness = _harness(tmp_path)
    key = "idem-abc-123"
    brief = _brief("wedding")

    first = await start_run(brief, account_id=ACCOUNT, shot_count=6, idempotency_key=key, **harness)
    second = await start_run(brief, account_id=ACCOUNT, shot_count=6, idempotency_key=key, **harness)

    assert first.run.run_id == second.run.run_id
    # Replayed, not re-charged.
    assert second.run.status == first.run.status
    assert harness["ledger"].balance(ACCOUNT) == 100 - len(first.run.photo_ids)
    charges = [
        entry
        for entry in harness["ledger"].entries(account_id=ACCOUNT)
        if entry.kind == "charge"
    ]
    assert len(charges) == 1


@pytest.mark.asyncio
async def test_different_keys_create_different_runs(tmp_path):
    harness = _harness(tmp_path)
    brief = _brief("wedding")
    a = await start_run(brief, account_id=ACCOUNT, shot_count=4, idempotency_key="k1", **harness)
    b = await start_run(brief, account_id=ACCOUNT, shot_count=4, idempotency_key="k2", **harness)
    assert a.run.run_id != b.run.run_id


@pytest.mark.asyncio
async def test_second_active_run_for_a_shoot_is_rejected(tmp_path):
    harness = _harness(tmp_path)
    brief = _brief("wedding")
    await start_run(brief, account_id=ACCOUNT, shot_count=4, **harness)

    # The first run finished, so a new one is allowed; but mark a live one to
    # prove concurrency is blocked.
    live = GenerationRun(run_id="run_live", shoot_id=brief.shoot_id, account_id=ACCOUNT)
    harness["store"].put(live)
    harness["store"].track_shoot(live)

    with pytest.raises(RunInProgress) as excinfo:
        await start_run(brief, account_id=ACCOUNT, shot_count=4, **harness)
    assert "already has an active run" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_is_deterministic_for_the_same_key(tmp_path):
    brief = _brief("wedding")
    a = await start_run(brief, account_id=ACCOUNT, shot_count=6, idempotency_key="stable", **_harness(tmp_path / "a"))
    b = await start_run(brief, account_id=ACCOUNT, shot_count=6, idempotency_key="stable", **_harness(tmp_path / "b"))
    assert a.run.run_id == b.run.run_id
    assert run_digest(a) == run_digest(b)


# --------------------------------------------------------------------------
# Ledger mechanics
# --------------------------------------------------------------------------


def test_reserve_holds_credits_and_settle_charges_delivered():
    ledger = CreditLedger(balances={ACCOUNT: 10})
    reservation = ledger.reserve(ACCOUNT, 4, run_id="run_1")
    assert ledger.balance(ACCOUNT) == 6
    assert reservation.outstanding == 4

    ledger.settle(reservation, delivered=3, file_failed=["s03"])
    assert reservation.charged == 3
    assert reservation.refunded == 1
    assert reservation.outstanding == 0
    # 4 reserved, 3 delivered, 1 refunded: the hold funded the charge, so the
    # balance moves only by the one credit that came back.
    assert ledger.balance(ACCOUNT) == 7


def test_cache_hits_are_refunded_with_their_own_reason():
    ledger = CreditLedger(balances={ACCOUNT: 10})
    reservation = ledger.reserve(ACCOUNT, 3, run_id="run_c")
    ledger.settle(reservation, delivered=3, cache_hits=1)

    assert reservation.charged == 2
    assert ledger.balance(ACCOUNT) == 8
    reasons = [entry.reason for entry in ledger.entries(account_id=ACCOUNT)]
    assert "refund: cache_hit" in reasons
    assert "charge: rendered" in reasons

def test_reserve_below_balance_raises_with_shortfall():
    ledger = CreditLedger(balances={ACCOUNT: 5})
    with pytest.raises(InsufficientCredits) as excinfo:
        ledger.reserve(ACCOUNT, 12, run_id="run_x")
    assert excinfo.value.shortfall == 7
    assert ledger.balance(ACCOUNT) == 5
    assert ledger.entries(account_id=ACCOUNT) == []


def test_reserve_is_idempotent_per_key():
    ledger = CreditLedger(balances={ACCOUNT: 10})
    first = ledger.reserve(ACCOUNT, 4, run_id="run_1", idempotency_key="k")
    second = ledger.reserve(ACCOUNT, 4, run_id="run_1", idempotency_key="k")
    assert first.reservation_id == second.reservation_id
    assert ledger.balance(ACCOUNT) == 6


def test_reusing_a_key_for_a_different_amount_conflicts():
    ledger = CreditLedger(balances={ACCOUNT: 10})
    ledger.reserve(ACCOUNT, 4, run_id="run_1", idempotency_key="k")
    with pytest.raises(LedgerConflict):
        ledger.reserve(ACCOUNT, 9, run_id="run_1", idempotency_key="k")


def test_settling_twice_does_not_double_charge():
    ledger = CreditLedger(balances={ACCOUNT: 10})
    reservation = ledger.reserve(ACCOUNT, 4, run_id="run_1")
    ledger.settle(reservation, delivered=4)
    balance_after = ledger.balance(ACCOUNT)
    ledger.settle(reservation, delivered=4)
    assert ledger.balance(ACCOUNT) == balance_after


def test_settle_never_charges_more_than_was_reserved():
    ledger = CreditLedger(balances={ACCOUNT: 10})
    reservation = ledger.reserve(ACCOUNT, 2, run_id="run_1")
    ledger.settle(reservation, delivered=9)
    # Capped at the reservation: the user agreed to 2, so 2 it is.
    assert reservation.charged == 2
    assert ledger.balance(ACCOUNT) == 8


def test_cancel_returns_the_whole_hold():
    ledger = CreditLedger(balances={ACCOUNT: 10})
    reservation = ledger.reserve(ACCOUNT, 6, run_id="run_1")
    ledger.cancel(reservation, reason="run refused")
    assert ledger.balance(ACCOUNT) == 10
    assert reservation.settled


def test_every_refund_is_a_visible_line():
    ledger = CreditLedger(balances={ACCOUNT: 12})
    reservation = ledger.reserve(ACCOUNT, 4, run_id="run_1")
    ledger.settle(reservation, delivered=2, file_failed=["s01", "s02"])
    refunds = [e for e in ledger.entries(account_id=ACCOUNT) if e.kind == "refund"]
    assert len(refunds) == 2
    assert all(entry.delta == 1 for entry in refunds)
    statement = ledger.statement(ACCOUNT, run_id="run_1")
    assert statement["charged"] == 2
    assert statement["refunded"] == 2
    assert statement["balance"] == 10


def test_zero_cost_run_settles_without_charging():
    ledger = CreditLedger(balances={ACCOUNT: 5})
    reservation = ledger.reserve(ACCOUNT, 0, run_id="run_free")
    ledger.settle(reservation, delivered=0)
    assert ledger.balance(ACCOUNT) == 5
    assert ledger.entries(account_id=ACCOUNT, run_id="run_free") != []


def test_credits_per_photo_is_one():
    assert CREDITS_PER_PHOTO == 1


def test_grant_is_idempotent():
    ledger = CreditLedger()
    ledger.grant(ACCOUNT, 10, reason="purchase", idempotency_key="pay-1")
    ledger.grant(ACCOUNT, 10, reason="purchase", idempotency_key="pay-1")
    assert ledger.balance(ACCOUNT) == 10


# --------------------------------------------------------------------------
# End-to-end through the real pipeline
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_run_charges_exactly_the_photo_count(tmp_path):
    harness = _harness(tmp_path)
    result = await start_run(_brief("wedding"), account_id=ACCOUNT, shot_count=6, **harness)

    assert result.run.status == "done"
    assert result.artifact is not None
    assert result.outcome is not None
    assert result.outcome.complete
    assert result.run.spent_credits == 6
    assert result.run.refunded_credits == 0
    assert harness["ledger"].balance(ACCOUNT) == 94


@pytest.mark.asyncio
async def test_photos_are_actually_stored(tmp_path):
    harness = _harness(tmp_path)
    result = await start_run(_brief("wedding"), account_id=ACCOUNT, shot_count=3, **harness)
    blobs = harness["blobs"]
    for photo in result.outcome.photos:
        assert await blobs.exists(photo.blob.key), f"missing blob {photo.blob.key}"


@pytest.mark.asyncio
async def test_memorial_runs_with_care_flag_not_refusal(tmp_path):
    harness = _harness(tmp_path)
    result = await start_run(_brief("memorial"), account_id=ACCOUNT, shot_count=4, **harness)
    assert result.run.status == "done"
    assert result.run.photo_ids != []


@pytest.mark.asyncio
async def test_state_history_tells_the_whole_story(tmp_path):
    harness = _harness(tmp_path)
    result = await start_run(_brief("wedding"), account_id=ACCOUNT, shot_count=2, **harness)
    statuses = [step["status"] for step in result.run.history]
    assert statuses == [
        "researching",
        "compiling",
        "rendering",
        "done",
    ], statuses


@pytest.mark.asyncio
async def test_events_are_emitted_in_order(tmp_path):
    harness = _harness(tmp_path)
    seen: list[tuple[str, str]] = []

    def on_event(kind, payload):
        seen.append((kind, payload["status"]))

    await start_run(
        _brief("wedding"),
        account_id=ACCOUNT,
        shot_count=2,
        on_event=on_event,
        **harness,
    )

    kinds = [kind for kind, _ in seen]
    assert kinds[0] == "run"
    assert kinds[-1] == "run_done"
    statuses = [status for _, status in seen]
    # Statuses only ever move forward through the machine.
    order = ["queued", "researching", "compiling", "rendering", "done"]
    indices = [order.index(status) for status in statuses]
    assert indices == sorted(indices), statuses


@pytest.mark.asyncio
async def test_run_result_serialises(tmp_path):
    harness = _harness(tmp_path)
    result = await start_run(_brief("wedding"), account_id=ACCOUNT, shot_count=2, **harness)
    payload = result.as_dict()
    assert payload["run"]["status"] == "done"
    assert payload["artifact_digest"]
    assert payload["render"]["complete"] is True
    assert payload["reservation"]["settled"] is True


@pytest.mark.asyncio
async def test_reservations_lists_the_charge_for_a_settled_run(tmp_path):
    """The *charged* figure lives on the reservation, not on the entry delta.

    Reserve-then-settle records a zero-delta `charge` audit line, so a host that
    reads only `entries()` cannot tell what a run cost and silently gives the
    photos away. `reservations()` is the read that makes the amount reachable.
    """
    harness = _harness(tmp_path)
    ledger = harness["ledger"]
    result = await start_run(_brief("wedding"), account_id=ACCOUNT, shot_count=2, **harness)

    settled = ledger.reservations(account_id=ACCOUNT)
    assert len(settled) == 1
    reservation = settled[0]
    assert reservation.run_id == result.run.run_id
    assert reservation.charged == 2
    assert reservation.settled is True

    # The audit line itself carries no money, which is exactly the trap.
    charges = [e for e in ledger.entries(account_id=ACCOUNT) if e.kind == "charge"]
    assert charges and all(entry.delta == 0 for entry in charges)


@pytest.mark.asyncio
async def test_run_result_carries_a_host_persisted_payload(tmp_path):
    """A host attaches its own rows without mutating the frozen result.

    The shell persists `GeneratedPhoto` rows and needs to hand them back from the
    same call. `RunResult` is frozen and slotted, so this must be a replace.
    """
    from dataclasses import replace

    harness = _harness(tmp_path)
    result = await start_run(_brief("wedding"), account_id=ACCOUNT, shot_count=2, **harness)
    assert result.persisted is None

    carried = replace(result, persisted={"photos": ["a", "b"]})
    assert carried.persisted == {"photos": ["a", "b"]}
    assert result.persisted is None, "the original must be untouched"
